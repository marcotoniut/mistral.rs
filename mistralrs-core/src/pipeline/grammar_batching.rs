use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::{
    device_map::DeviceMapper,
    kv_cache::{KvCache, NormalCache},
    prefix_cacher::PrefixCacheManagerV2,
    sequence::{Sequence, SequenceRecognizer},
};

use super::{
    inputs_processor::text_models_inputs_processor::{get_completion_input_windowed, ModelInputs},
    GeneralMetadata, ModelKind, Pipeline, RecurrentBatchKind,
};

const MAX_PROPOSED_TOKENS: usize = 16;

fn proposed_tokens(
    matcher: &llguidance::Matcher,
    limit: usize,
    stop_tokens: &[u32],
) -> Result<Vec<u32>> {
    let mut matcher = matcher.deep_clone();
    let env = matcher.tok_env().map_err(candle_core::Error::msg)?;
    if limit == 0
        || matcher.is_stopped()
        || matcher.is_accepting().map_err(candle_core::Error::msg)?
    {
        return Ok(Vec::new());
    }
    // Forced bytes can have several valid tokenizations; the sampler must verify each proposal.
    let tokens = matcher.compute_ff_tokens();
    if let Some(error) = matcher.get_error() {
        candle_core::bail!("{error}");
    }
    Ok(tokens
        .into_iter()
        .take(limit)
        .take_while(|token| {
            !stop_tokens.contains(token) && !env.tok_trie().is_special_token(*token)
        })
        .collect())
}

pub(super) fn plan(
    seq: &Sequence,
    metadata: &GeneralMetadata,
    device: &Device,
) -> Result<Vec<u32>> {
    if !seq.is_completion()
        || seq.has_prefill_toks()
        || seq.return_logprobs()
        || !seq.sampler().can_batch_grammar_tokens()
        || seq.tool_call_state.is_some()
        || seq.has_reasoning_state()
        || !seq.stop_strings().is_empty()
        || seq.has_images()
        || seq.has_audios()
        || seq.has_videos()
        || !seq.mm_features().is_empty()
        || seq.adapter_generation().is_some()
        || seq.active_staged_speculative_len() != 0
        || seq.speculative_depth_hint().is_some()
        || metadata.no_kv_cache
        || metadata.is_xlora
        || !matches!(
            metadata.kind,
            ModelKind::Normal | ModelKind::GgufQuantized { .. }
        )
        || metadata.cache_config.is_some()
        || metadata.sliding_window.is_some()
        || metadata.activation_dtype != DType::F32
        || !(device.is_cpu() || device.is_metal())
    {
        return Ok(Vec::new());
    }
    let cached_len = seq.get_toks().len().saturating_sub(1);
    if seq.normal_cache_ref().is_empty()
        || !seq.normal_cache_ref().iter().all(|cache| {
            matches!(cache, Some(KvCache::Normal { k, v })
                if k.current_seq_len() == cached_len && v.current_seq_len() == cached_len)
        })
    {
        return Ok(Vec::new());
    }
    let limit = MAX_PROPOSED_TOKENS
        .min(
            seq.max_generation_len(metadata.max_seq_len)
                .saturating_sub(seq.generated_len()),
        )
        .min(metadata.max_seq_len.saturating_sub(seq.get_toks().len()));
    let mut stop_tokens = seq.stop_tokens().to_vec();
    stop_tokens.extend_from_slice(&metadata.eos_tok);
    let SequenceRecognizer::Llguidance(matcher) = &seq.recognizer else {
        return Ok(Vec::new());
    };
    let proposed = proposed_tokens(matcher, limit, &stop_tokens)?;
    Ok(if proposed.len() > 1 {
        proposed
    } else {
        Vec::new()
    })
}

pub(super) fn inputs(
    seqs: &[&mut Sequence],
    proposed: &[u32],
    device: &Device,
    mapper: Option<&dyn DeviceMapper>,
) -> Result<ModelInputs> {
    let mut tokens = seqs[0].get_toks().to_vec();
    tokens.extend_from_slice(&proposed[..proposed.len() - 1]);
    let input = get_completion_input_windowed(
        vec![&tokens],
        seqs,
        device,
        false,
        None,
        false,
        None,
        mapper,
        None,
        proposed.len(),
    )
    .map_err(candle_core::Error::msg)?
    .inputs;
    Ok(ModelInputs {
        input_ids: input.input,
        input_ids_full: None,
        seqlen_offsets: input.positions,
        seqlen_offsets_full: None,
        context_lens: input.context_lens,
        position_ids: input.position_ids,
        paged_attn_meta: None,
        flash_meta: input.flash_meta,
        flash_meta_full: None,
        recurrent_batch_kind: RecurrentBatchKind::Prefill,
        adapter_leases: vec![None].into(),
    })
}

pub(super) struct SamplingContext<'a> {
    pub prefix_cacher: &'a mut PrefixCacheManagerV2,
    pub disable_eos_stop: bool,
    pub rng: Arc<Mutex<Isaac64Rng>>,
}

fn set_execution_len(cache: &mut NormalCache, seq: &mut Sequence, len: usize) -> Result<()> {
    for layer in cache
        .0
        .iter_mut()
        .chain(seq.normal_cache().iter_mut().flatten())
    {
        layer.set_len(len)?;
    }
    Ok(())
}

pub(super) async fn sample<P: Pipeline + ?Sized>(
    pipeline: &P,
    seqs: &mut [&mut Sequence],
    logits: Tensor,
    proposed: &[u32],
    context: SamplingContext<'_>,
) -> Result<()> {
    let original_len = seqs[0].get_toks().len();
    set_execution_len(&mut pipeline.cache().normal(), seqs[0], original_len - 1)?;
    let mut accepted = 0;
    for (index, expected) in proposed.iter().enumerate() {
        if seqs[0].is_finished_paged_attn() {
            break;
        }
        // Finalization may retain the cache, so expose only the prefix behind this logits row.
        set_execution_len(
            &mut pipeline.cache().normal(),
            seqs[0],
            original_len + index,
        )?;
        pipeline
            .sample_causal_gen(
                seqs,
                vec![logits.narrow(1, index, 1)?],
                context.prefix_cacher,
                context.disable_eos_stop,
                context.rng.clone(),
            )
            .await?;
        if seqs[0].get_toks().last() != Some(expected) {
            break;
        }
        accepted += 1;
    }
    tracing::debug!(
        proposed_tokens = proposed.len(),
        accepted_tokens = accepted,
        "Batched grammar token proposals"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(device: &Device, max_len: Option<usize>, return_logprobs: bool) -> Sequence {
        use crate::{
            sampler::Sampler,
            sequence::{SeqStepType, SequenceGroup, SequenceState},
        };
        use std::{collections::HashMap, sync::Arc};
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            1,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            1, false, true, None,
        )));
        let mut seq = Sequence::new_waiting(
            vec![10, 11, 12],
            "prompt".into(),
            0,
            0,
            1,
            tx,
            sampler,
            vec![],
            vec![],
            max_len,
            return_logprobs,
            false,
            group,
            0,
            0,
            SequenceRecognizer::Llguidance(Box::new(matcher("start: \"abcdef\""))),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
            None,
        );
        let mut cache = KvCache::new_normal(2, 64, 16);
        let data = candle_core::Tensor::zeros((1, 1, 2, 1), DType::F32, device).unwrap();
        cache.append(&data, &data).unwrap();
        seq.normal_cache()[0] = Some(cache);
        seq.set_state(SequenceState::RunningCompletion);
        seq
    }

    fn metadata() -> GeneralMetadata {
        GeneralMetadata {
            max_seq_len: 64,
            llg_factory: None,
            no_kv_cache: false,
            no_prefix_cache: false,
            num_hidden_layers: 1,
            eos_tok: vec![],
            kind: super::super::ModelKind::Normal,
            is_xlora: false,
            activation_dtype: DType::F32,
            sliding_window: None,
            cache_config: None,
            cache_engine: None,
            model_metadata: None,
            modalities: super::super::Modalities {
                input: vec![],
                output: vec![],
            },
            loaded_for_uqff_write: false,
        }
    }

    fn assert_inputs(device: &Device) {
        let mut seq = sequence(device, None, false);
        let original = seq.get_toks().to_vec();
        let input = inputs(&[&mut seq], &[b'a' as u32, b'b' as u32], device, None).unwrap();
        assert_eq!(
            input.input_ids.to_vec2::<u32>().unwrap(),
            vec![vec![12, b'a' as u32]]
        );
        assert_eq!(input.seqlen_offsets, [2]);
        assert_eq!(input.context_lens, [(0, 2)]);
        assert_eq!(input.position_ids, [4]);
        assert_eq!(seq.get_toks(), original);
        assert!(!seq.has_prefill_toks());
        assert_eq!(
            seq.normal_cache_ref()[0]
                .as_ref()
                .unwrap()
                .current_seq_len(),
            2
        );
    }

    #[test]
    fn batches_uncached_suffix_with_absolute_positions_on_cpu() {
        assert_inputs(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn batches_uncached_suffix_with_absolute_positions_on_metal() {
        assert_inputs(&Device::new_metal(0).unwrap());
    }

    fn assert_rejected_tail_can_be_replaced(device: &Device) {
        let mut seq = sequence(device, None, false);
        let mut forwarded = KvCache::new_normal(2, 64, 16);
        let values =
            Tensor::from_slice(&[10_f32, 11., 12., 97., 98.], (1, 1, 5, 1), device).unwrap();
        forwarded.append(&values, &values).unwrap();
        let mut retained = KvCache::new_normal(2, 64, 16);
        retained.append(&values, &values).unwrap();
        let mut cache = NormalCache(vec![forwarded.clone()]);
        seq.normal_cache()[0] = Some(forwarded);
        set_execution_len(&mut cache, &mut seq, 3).unwrap();
        assert_eq!(cache.0[0].current_seq_len(), 3);
        assert_eq!(
            seq.normal_cache_ref()[0]
                .as_ref()
                .unwrap()
                .current_seq_len(),
            3
        );
        assert_eq!(retained.current_seq_len(), 5);
        assert_eq!(
            retained
                .k()
                .unwrap()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [10., 11., 12., 97., 98.]
        );
        let actual = Tensor::from_slice(&[42_f32], (1, 1, 1, 1), device).unwrap();
        cache.0[0].append(&actual, &actual).unwrap();
        assert_eq!(
            cache.0[0]
                .k()
                .unwrap()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [10., 11., 12., 42.]
        );
        assert_eq!(
            retained
                .k()
                .unwrap()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [10., 11., 12., 97., 98.]
        );
    }

    #[test]
    fn rejected_tail_does_not_extend_the_executed_prefix_on_cpu() {
        assert_rejected_tail_can_be_replaced(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn rejected_tail_does_not_extend_the_executed_prefix_on_metal() {
        assert_rejected_tail_can_be_replaced(&Device::new_metal(0).unwrap());
    }

    #[tokio::test]
    async fn sampler_can_reject_a_canonical_proposal_without_advancing_its_tail() {
        use rand::SeedableRng;
        let base = toktrie::ApproximateTokEnv::single_byte_env();
        let mut words = (0..base.tok_trie().vocab_size())
            .map(|id| base.tok_trie().token(u32::try_from(id).unwrap()).to_vec())
            .collect::<Vec<_>>();
        let merged_id = u32::try_from(words.len()).unwrap();
        words.push(b"ab".to_vec());
        let info = toktrie::TokRxInfo::new(u32::try_from(words.len()).unwrap(), base.eos_token());
        let env: toktrie::TokEnv = Arc::new(toktrie::TokEnvWithTrie::new(
            base,
            toktrie::TokTrie::from(&info, &words),
        ));
        let factory = Arc::new(llguidance::ParserFactory::new_simple(&env).unwrap());
        let parser = super::super::llg::constraint_from_llg_grammar(
            &factory,
            llguidance::api::TopLevelGrammar::from_lark("start: \"abcdefghi\"".to_string()),
        )
        .unwrap();
        let mut seq = sequence(&Device::Cpu, None, false);
        seq.recognizer = SequenceRecognizer::Llguidance(Box::new(parser));
        let proposed = plan(&seq, &metadata(), &Device::Cpu).unwrap();
        assert_eq!(proposed.first(), Some(&(b'a' as u32)));
        let mut logits = vec![0_f32; words.len()];
        logits[merged_id as usize] = 10.;
        let actual = super::super::sampling::sample_sequence(
            Tensor::from_vec(logits, (1, 1, words.len()), &Device::Cpu).unwrap(),
            &mut seq,
            false,
            None,
            Some(factory),
            64,
            Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0))),
            false,
            false,
            false,
        )
        .await
        .unwrap();
        assert_eq!(actual.token, merged_id);
        assert_ne!(actual.token, proposed[0]);
        let SequenceRecognizer::Llguidance(parser) = &mut seq.recognizer else {
            unreachable!()
        };
        assert_eq!(parser.validate_tokens(&[b'c' as u32]).unwrap(), 1);
        assert_eq!(parser.validate_tokens(&[b'b' as u32]).unwrap(), 0);
    }

    #[test]
    fn admits_deterministic_top_one_sampling_but_not_stochastic_sampling() {
        let sampler = |temperature, top_k, bias| {
            crate::sampler::Sampler::new(
                temperature,
                0,
                None,
                None,
                None,
                None,
                None,
                top_k,
                1.,
                0.,
                bias,
                vec![],
            )
            .unwrap()
        };
        assert!(sampler(None, -1, std::collections::HashMap::new()).can_batch_grammar_tokens());
        assert!(sampler(Some(1.), 1, std::collections::HashMap::new()).can_batch_grammar_tokens());
        assert!(!sampler(Some(1.), 2, std::collections::HashMap::new()).can_batch_grammar_tokens());
        assert!(
            !sampler(Some(1.), 1, std::collections::HashMap::from([(42, 1.)]))
                .can_batch_grammar_tokens()
        );
    }

    #[test]
    fn parser_errors_are_not_treated_as_empty_proposals() {
        let parser = llguidance::Matcher::new(Err(anyhow::anyhow!("failed grammar")));
        assert!(proposed_tokens(&parser, 16, &[])
            .unwrap_err()
            .to_string()
            .contains("failed grammar"));
    }

    #[test]
    fn respects_generation_and_model_limits() {
        let meta = metadata();
        assert_eq!(
            plan(&sequence(&Device::Cpu, Some(3), false), &meta, &Device::Cpu).unwrap(),
            vec![b'a' as u32, b'b' as u32, b'c' as u32]
        );
        assert!(
            plan(&sequence(&Device::Cpu, Some(1), false), &meta, &Device::Cpu)
                .unwrap()
                .is_empty()
        );
        let mut meta = metadata();
        meta.max_seq_len = 5;
        assert_eq!(
            plan(&sequence(&Device::Cpu, None, false), &meta, &Device::Cpu).unwrap(),
            vec![b'a' as u32, b'b' as u32]
        );
    }

    #[test]
    fn excludes_unsupported_execution_and_incomplete_caches() {
        let mut seq = sequence(&Device::Cpu, None, true);
        let mut meta = metadata();
        assert!(plan(&seq, &meta, &Device::Cpu).unwrap().is_empty());
        seq = sequence(&Device::Cpu, None, false);
        for dtype in [DType::F16, DType::BF16] {
            meta.activation_dtype = dtype;
            assert!(plan(&seq, &meta, &Device::Cpu).unwrap().is_empty());
        }
        meta = metadata();
        meta.no_kv_cache = true;
        assert!(plan(&seq, &meta, &Device::Cpu).unwrap().is_empty());
        meta = metadata();
        meta.sliding_window = Some(16);
        assert!(plan(&seq, &meta, &Device::Cpu).unwrap().is_empty());
        meta = metadata();
        seq.normal_cache()[0] = None;
        assert!(plan(&seq, &meta, &Device::Cpu).unwrap().is_empty());
        seq = sequence(&Device::Cpu, None, false);
        seq.set_state(crate::sequence::SequenceState::Done(
            crate::sequence::StopReason::Canceled,
        ));
        assert!(plan(&seq, &meta, &Device::Cpu).unwrap().is_empty());
    }

    fn matcher(grammar: &str) -> llguidance::Matcher {
        let env = toktrie::ApproximateTokEnv::single_byte_env();
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        super::super::llg::constraint_from_llg_grammar(
            &factory,
            llguidance::api::TopLevelGrammar::from_lark(grammar.to_string()),
        )
        .unwrap()
    }

    #[test]
    fn stops_at_first_token_choice_without_advancing_origin() {
        let mut parser = matcher("start: \"abc\" (\"x\" | \"y\")");
        assert_eq!(
            proposed_tokens(&parser, 16, &[]).unwrap(),
            b"abc".iter().map(|b| *b as u32).collect::<Vec<_>>()
        );
        assert_eq!(parser.validate_tokens(&[b'a' as u32]).unwrap(), 1);
        parser
            .consume_tokens(&[b'a' as u32, b'b' as u32, b'c' as u32])
            .unwrap();
        assert!(proposed_tokens(&parser, 16, &[]).unwrap().is_empty());
    }

    #[test]
    fn does_not_propose_tokens_for_noncanonical_tokenizers() {
        let env = toktrie::ApproximateTokEnv::single_byte_env();
        let mut words = (0..env.tok_trie().vocab_size())
            .map(|id| env.tok_trie().token(u32::try_from(id).unwrap()).to_vec())
            .collect::<Vec<_>>();
        words.push(b"ab".to_vec());
        let info = toktrie::TokRxInfo::new(
            u32::try_from(words.len()).unwrap(),
            env.tok_trie().eos_token(),
        );
        let env: toktrie::TokEnv = std::sync::Arc::new(toktrie::ApproximateTokEnv::new(
            toktrie::TokTrie::from(&info, &words),
        ));
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = super::super::llg::constraint_from_llg_grammar(
            &factory,
            llguidance::api::TopLevelGrammar::from_lark("start: \"abc\"".to_string()),
        )
        .unwrap();
        assert!(proposed_tokens(&parser, 16, &[]).unwrap().is_empty());
    }

    #[test]
    fn respects_stop_tokens_and_limits() {
        let parser = matcher("start: \"abc\"");
        assert!(proposed_tokens(&parser, 0, &[]).unwrap().is_empty());
        assert_eq!(
            proposed_tokens(&parser, 2, &[]).unwrap(),
            vec![b'a' as u32, b'b' as u32]
        );
        assert_eq!(
            proposed_tokens(&parser, 16, &[b'b' as u32]).unwrap(),
            vec![b'a' as u32]
        );
    }

    #[test]
    fn does_not_extend_an_accepting_grammar() {
        let parser = matcher("start: \"a\"?");
        assert!(proposed_tokens(&parser, 16, &[]).unwrap().is_empty());
        let mut parser = matcher("start: \"abc\"");
        let proposed = proposed_tokens(&parser, 16, &[]).unwrap();
        parser.consume_tokens(&proposed).unwrap();
        assert!(proposed_tokens(&parser, 16, &[]).unwrap().is_empty());
    }
}
