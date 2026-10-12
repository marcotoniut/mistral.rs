use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor};

use super::{CacheElement, CacheKey, CachedRecurrentState, MatchingCache, PrefixCacheManagerV2};
use crate::{
    device_map::DeviceMapper,
    kv_cache::{
        CacheManager, EitherCache, KvCache, NormalCache, NormalCacheManager,
        RecurrentStateSnapshot, RotatingCache, SingleCache,
    },
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
    pipeline::{CacheManagerMixin, GeneralMetadata, MetadataMixin, Modalities, ModelKind},
    sampler::Sampler,
    sequence::{SeqStepType, Sequence, SequenceGroup, SequenceRecognizer},
    AdapterGenerationId,
};

const TEST_MAX_SEQ_LEN: usize = 1024;
const TEST_KV_HEADS: usize = 2;
const TEST_KV_HEAD_DIM: usize = 2;
const TEST_PROMPT_LEN: usize = 4;
const TEST_HIDDEN_SIZE: usize = 4;
const TEST_INTERMEDIATE_SIZE: usize = 8;
const TEST_VOCAB_SIZE: usize = 8;
const LOGITS_TOLERANCE: f32 = 1e-5;

fn element(len: usize, device: &Device) -> Result<CacheElement> {
    let data_len = u16::try_from(len).unwrap();
    let data = Tensor::from_vec(
        (1..=data_len).map(f32::from).collect::<Vec<_>>(),
        (1, 1, len, 1),
        device,
    )?;
    let mut k = SingleCache::new(2, TEST_MAX_SEQ_LEN, len);
    let mut v = SingleCache::new(2, TEST_MAX_SEQ_LEN, len);
    k.append(&data)?;
    v.append(&data)?;
    Ok(CacheElement {
        cache: vec![Some(KvCache::Normal { k, v })],
        recurrent_snapshots: None,
        audio_hashes: None,
        image_hashes: None,
        video_hashes: None,
    })
}

fn manager(len: usize, device: &Device) -> Result<PrefixCacheManagerV2> {
    let mut manager = PrefixCacheManagerV2::new(16, false, false);
    manager.caches.insert(
        CacheKey::new((1..=u32::try_from(len).unwrap()).collect(), None),
        element(len, device)?,
    );
    Ok(manager)
}

fn search(manager: &mut PrefixCacheManagerV2, tokens: &[u32]) -> Result<Option<MatchingCache>> {
    manager.search_for_matching_cache(tokens, None, &[], None, None, None)
}

fn hit(manager: &mut PrefixCacheManagerV2, tokens: &[u32]) -> Result<MatchingCache> {
    Ok(search(manager, tokens)?.expect("expected normal text prefix hit"))
}

fn values(cache: &KvCache) -> Result<Vec<f32>> {
    let k = cache
        .k()?
        .expect("normal K data")
        .flatten_all()?
        .to_vec1()?;
    let v: Vec<f32> = cache
        .v()?
        .expect("normal V data")
        .flatten_all()?
        .to_vec1()?;
    assert_eq!(k, v);
    Ok(k)
}

#[test]
fn exact_match_replays_one_token_and_preserves_retained_state() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    for _ in 0..3 {
        let MatchingCache::Normal {
            normal,
            toks,
            offset,
            ..
        } = hit(&mut manager, &[1, 2, 3, 4])?;
        assert_eq!(offset, 3);
        assert_eq!(toks, [4]);
        assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3.]);
        let retained = manager.caches.values().next().unwrap();
        assert_eq!(retained.coverage_len(), 4);
        assert_eq!(
            values(retained.cache[0].as_ref().unwrap())?,
            [1., 2., 3., 4.]
        );
    }
    Ok(())
}

#[test]
fn empty_and_single_token_prompts_skip_zero_reuse() -> Result<()> {
    let mut manager = manager(1, &Device::Cpu)?;
    assert!(search(&mut manager, &[])?.is_none());
    assert!(search(&mut manager, &[1])?.is_none());
    assert!(search(&mut manager, &[9])?.is_none());
    let MatchingCache::Normal { offset, toks, .. } = hit(&mut manager, &[1, 2])?;
    assert_eq!(offset, 1);
    assert_eq!(toks, [2]);
    Ok(())
}

fn shorter_fallback(exact: CacheElement, shorter: CacheElement) -> Result<MatchingCache> {
    let mut manager = PrefixCacheManagerV2::new(16, false, false);
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], None), exact);
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3], None), shorter);
    let result = hit(&mut manager, &[1, 2, 3, 4])?;
    let MatchingCache::Normal {
        normal,
        offset,
        toks,
        ..
    } = &result;
    assert_eq!(*offset, 3);
    assert_eq!(toks, &[4]);
    for layer in normal {
        assert_eq!(values(layer.as_ref().unwrap())?, [1., 2., 3.]);
    }
    Ok(result)
}

#[test]
fn rotating_exact_candidate_does_not_hide_shorter_reuse() -> Result<()> {
    let mut exact = element(4, &Device::Cpu)?;
    let mut shorter = element(3, &Device::Cpu)?;
    for cached in [&mut exact, &mut shorter] {
        let len = cached.coverage_len();
        let data = cached.cache[0].as_ref().unwrap().k()?.unwrap();
        let mut k = RotatingCache::new(2, TEST_MAX_SEQ_LEN, len);
        let mut v = RotatingCache::new(2, TEST_MAX_SEQ_LEN, len);
        k.append(&data)?;
        v.append(&data)?;
        cached.cache = vec![Some(KvCache::Rotating { k, v })];
    }
    let MatchingCache::Normal {
        recurrent_snapshots,
        ..
    } = shorter_fallback(exact, shorter)?;
    assert!(recurrent_snapshots.is_none());
    Ok(())
}

#[test]
fn recurrent_exact_candidate_restores_shorter_state() -> Result<()> {
    let mut exact = element(4, &Device::Cpu)?;
    let mut shorter = element(3, &Device::Cpu)?;
    for cached in [&mut exact, &mut shorter] {
        let len = cached.coverage_len();
        let value = f32::from(u16::try_from(len).unwrap());
        cached.recurrent_snapshots = Some(CachedRecurrentState {
            len,
            snapshots: vec![RecurrentStateSnapshot {
                conv_state: Tensor::from_vec(vec![value], (1, 1, 1), &Device::Cpu)?,
                recurrent_state: Tensor::from_vec(vec![-value], (1, 1, 1), &Device::Cpu)?,
                state_layout: crate::kv_cache::RecurrentStateLayout::Opaque,
            }],
        });
    }
    let MatchingCache::Normal {
        recurrent_snapshots,
        ..
    } = shorter_fallback(exact, shorter)?;
    let snapshots = recurrent_snapshots.expect("shorter recurrent state must be restored");
    assert_eq!(snapshots.len(), 1);
    assert_eq!(
        snapshots[0].conv_state.flatten_all()?.to_vec1::<f32>()?,
        [3.]
    );
    assert_eq!(
        snapshots[0]
            .recurrent_state
            .flatten_all()?
            .to_vec1::<f32>()?,
        [-3.]
    );
    assert_eq!(
        snapshots[0].state_layout,
        crate::kv_cache::RecurrentStateLayout::Opaque
    );
    Ok(())
}

#[test]
fn incomplete_exact_candidate_does_not_hide_complete_shorter_reuse() -> Result<()> {
    let mut exact = element(4, &Device::Cpu)?;
    let mut shorter = element(3, &Device::Cpu)?;
    exact.cache.push(None);
    shorter.cache.extend(element(3, &Device::Cpu)?.cache);
    assert_eq!(exact.cache.len(), shorter.cache.len());
    let MatchingCache::Normal {
        normal,
        recurrent_snapshots,
        ..
    } = shorter_fallback(exact, shorter)?;
    assert_eq!(normal.len(), 2);
    assert!(recurrent_snapshots.is_none());
    Ok(())
}

#[test]
fn shorter_keys_do_not_hide_a_later_longer_prefix() -> Result<()> {
    let mut manager = manager(3, &Device::Cpu)?;
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2], None), element(2, &Device::Cpu)?);
    manager.caches.insert(
        CacheKey::new(vec![1, 2, 9], None),
        element(3, &Device::Cpu)?,
    );
    manager.caches.insert(
        CacheKey::new(vec![1, 2, 3, 4], None),
        element(4, &Device::Cpu)?,
    );
    let MatchingCache::Normal {
        normal,
        offset,
        toks,
        ..
    } = hit(&mut manager, &[1, 2, 3, 4, 5])?;
    assert_eq!(offset, 4);
    assert_eq!(toks, [5]);
    assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3., 4.]);
    Ok(())
}

#[test]
fn partial_prefix_reuse_stops_at_forwarded_tokens() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &[1, 2, 3, 9])?;
    assert_eq!(offset, 3);
    assert_eq!(toks, [9]);
    assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3.]);
    let cached = manager.caches.shift_remove_index(0).unwrap().1;
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4, 5], None), cached);
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &[1, 2, 3, 4, 5])?;
    assert_eq!(offset, 4);
    assert_eq!(toks, [5]);
    assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3., 4.]);
    Ok(())
}

#[test]
fn exact_replay_isolates_adapter_generations_and_shared_layers() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    let mut cached = manager.caches.shift_remove_index(0).unwrap().1;
    cached.cache.push(Some(KvCache::Shared { owner: 0 }));
    let a = AdapterGenerationId::from_bytes([1; 32]);
    let b = AdapterGenerationId::from_bytes([2; 32]);
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], Some(a)), cached);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    assert!(manager
        .search_for_matching_cache(&[1, 2, 3, 4], Some(b), &[], None, None, None)?
        .is_none());
    let Some(MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    }) = manager.search_for_matching_cache(&[1, 2, 3, 4], Some(a), &[], None, None, None)?
    else {
        panic!("matching adapter missing")
    };
    assert_eq!(offset, 3);
    assert_eq!(toks, [4]);
    assert!(matches!(normal[1], Some(KvCache::Shared { owner: 0 })));
    Ok(())
}

#[test]
fn unsupported_cache_states_skip_exact_replay() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    manager.has_paged_attention = true;
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    manager.has_paged_attention = false;
    manager.no_prefix_cache = true;
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    manager.no_prefix_cache = false;
    for kind in [MultimodalKind::Image, MultimodalKind::Audio] {
        let features = [MultiModalFeature {
            kind,
            offset: 1,
            length: 2,
            hashes: vec![],
            item_range: 0..1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }];
        assert!(manager
            .search_for_matching_cache(&[1, 2, 3, 4], None, &features, None, None, None)?
            .is_none());
    }
    for kind in 0..3 {
        let mut cached = element(4, &Device::Cpu)?;
        match kind {
            0 => cached.image_hashes = Some(vec![]),
            1 => cached.audio_hashes = Some(vec![]),
            _ => cached.video_hashes = Some(vec![]),
        }
        manager
            .caches
            .insert(CacheKey::new(vec![1, 2, 3, 4], None), cached);
        assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    }
    manager.caches.insert(
        CacheKey::new(vec![1, 2, 3, 4], None),
        element(4, &Device::Cpu)?,
    );
    for kind in 0..3 {
        let image = (kind == 0).then_some([].as_slice());
        let audio = (kind == 1).then_some([].as_slice());
        let video = (kind == 2).then_some([].as_slice());
        assert!(manager
            .search_for_matching_cache(&[1, 2, 3, 4], None, &[], image, audio, video)?
            .is_none());
    }
    let mut recurrent = element(4, &Device::Cpu)?;
    recurrent.recurrent_snapshots = Some(CachedRecurrentState {
        len: 4,
        snapshots: vec![RecurrentStateSnapshot {
            conv_state: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?,
            recurrent_state: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?,
            state_layout: crate::kv_cache::RecurrentStateLayout::Opaque,
        }],
    });
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], None), recurrent);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    let mut missing = element(4, &Device::Cpu)?;
    missing.cache.push(None);
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], None), missing);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    let data = Tensor::zeros((1, 1, 4, 1), DType::F32, &Device::Cpu)?;
    let mut k = RotatingCache::new(2, 8, 8);
    let mut v = RotatingCache::new(2, 8, 8);
    k.append(&data)?;
    v.append(&data)?;
    let mut rotating = element(4, &Device::Cpu)?;
    rotating.cache = vec![Some(KvCache::Rotating { k, v })];
    manager
        .caches
        .insert(CacheKey::new(vec![1, 2, 3, 4], None), rotating);
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    Ok(())
}

struct TestPipeline {
    cache: EitherCache,
    device: Device,
}

impl TestPipeline {
    fn new(device: Device) -> Self {
        Self {
            cache: EitherCache::Normal(NormalCache::new(1, TEST_MAX_SEQ_LEN)),
            device,
        }
    }
}

impl CacheManagerMixin for TestPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> Result<()> {
        NormalCacheManager.clone_in_cache(self, seqs, false)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        NormalCacheManager.clone_out_cache(self, seqs, false);
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        _: bool,
        draft: bool,
        preallocated: bool,
    ) -> Result<()> {
        NormalCacheManager.set_none_cache(self, seqs, draft, preallocated)
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
}

impl MetadataMixin for TestPipeline {
    fn device(&self) -> Device {
        self.device.clone()
    }
    fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
        None
    }
    fn name(&self) -> String {
        "replay-test".into()
    }
    fn reset_non_granular_state(&self) {}
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        Arc::new(GeneralMetadata {
            max_seq_len: TEST_MAX_SEQ_LEN,
            llg_factory: None,
            no_kv_cache: false,
            no_prefix_cache: false,
            num_hidden_layers: 1,
            eos_tok: vec![],
            kind: ModelKind::Normal,
            is_xlora: false,
            activation_dtype: DType::F32,
            sliding_window: None,
            cache_config: None,
            cache_engine: None,
            model_metadata: None,
            modalities: Modalities {
                input: vec![],
                output: vec![],
            },
            loaded_for_uqff_write: false,
        })
    }
}

fn sequence(normal: Vec<Option<KvCache>>, toks: Vec<u32>, offset: usize) -> Sequence {
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let sampler = Sampler::new(
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        32,
        1.,
        0.,
        HashMap::new(),
        vec![],
    )
    .unwrap();
    let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
        1, false, true, None,
    )));
    let original_tokens = (1..=u32::try_from(offset).unwrap())
        .chain(toks.iter().copied())
        .collect();
    Sequence::new_waiting(
        original_tokens,
        "prompt".into(),
        0,
        0,
        1,
        tx,
        sampler,
        vec![],
        vec![],
        None,
        false,
        false,
        group,
        0,
        0,
        SequenceRecognizer::None,
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
    )
    .prefill_v2_normal(normal, toks, offset)
}

fn append(pipeline: &TestPipeline, value: f32) -> Result<()> {
    let source = Tensor::full(value, (1, 1, 1, 1), &pipeline.device)?;
    append_tensor(pipeline, &source)
}

fn append_tensor(pipeline: &TestPipeline, source: &Tensor) -> Result<()> {
    let mut cache = pipeline.cache.normal();
    if pipeline.device.is_metal() {
        cache.0[0].append(source, source)?;
        return Ok(());
    }
    let KvCache::Normal { k, v } = &mut cache.0[0] else {
        panic!("normal cache required")
    };
    k.append(source)?;
    v.append(source)?;
    Ok(())
}

fn short_prefix_from_long_retained_cache(device: Device) -> Result<()> {
    let retained_len = NormalCache::CACHE_GROW_SIZE + 88;
    let mut manager = manager(retained_len, &device)?;
    let tokens: Vec<_> = (1..=20).collect();
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &tokens)?;
    assert_eq!(offset, tokens.len() - 1);
    let mut sequence = sequence(normal, toks, offset);
    let pipeline = TestPipeline::new(device);
    NormalCacheManager.clone_in_cache(&pipeline, &mut [&mut sequence], false)?;
    append(&pipeline, 1234.)?;
    NormalCacheManager.clone_out_cache(&pipeline, &mut [&mut sequence], false);
    let cache = sequence.normal_cache();
    let KvCache::Normal { k, v } = cache[0].as_ref().unwrap() else {
        panic!("normal cache required")
    };
    assert_eq!(k.capacity_seq_len, NormalCache::CACHE_GROW_SIZE);
    assert_eq!(v.capacity_seq_len, NormalCache::CACHE_GROW_SIZE);
    let mut expected: Vec<f32> = (1u16..20).map(f32::from).collect();
    expected.push(1234.);
    assert_eq!(values(cache[0].as_ref().unwrap())?, expected);
    let retained = manager.caches.values().next().unwrap();
    assert_eq!(retained.coverage_len(), retained_len);
    assert_eq!(
        values(retained.cache[0].as_ref().unwrap())?.len(),
        retained_len
    );
    assert_eq!(values(retained.cache[0].as_ref().unwrap())?[19], 20.);
    let restored = hit(
        &mut manager,
        &(1..=u32::try_from(retained_len).unwrap()).collect::<Vec<_>>(),
    )?;
    let MatchingCache::Normal { normal, offset, .. } = restored;
    assert_eq!(offset, retained_len - 1);
    assert_eq!(values(normal[0].as_ref().unwrap())?[19], 20.);
    Ok(())
}

#[test]
fn restoring_a_short_prefix_does_not_inherit_unused_capacity_on_cpu() -> Result<()> {
    short_prefix_from_long_retained_cache(Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn restoring_a_short_prefix_does_not_inherit_unused_capacity_on_metal() -> Result<()> {
    short_prefix_from_long_retained_cache(Device::new_metal(0)?)
}

fn batched_replay(device: Device) -> Result<()> {
    let block = NormalCache::CACHE_GROW_SIZE;
    for capacities in [
        [block, 2 * block],
        [2 * block, block],
        [block, block],
        [2 * block, 2 * block],
    ] {
        for draft in [false, true] {
            let mut manager = PrefixCacheManagerV2::new(16, false, false);
            for (row, capacity) in capacities.into_iter().enumerate() {
                let token = u32::try_from(row + 1).unwrap();
                let data = Tensor::full(
                    [1f32, 2.][row],
                    (1, TEST_KV_HEADS, TEST_PROMPT_LEN, TEST_KV_HEAD_DIM),
                    &device,
                )?;
                let mut k = SingleCache::new(2, TEST_MAX_SEQ_LEN, capacity);
                let mut v = SingleCache::new(2, TEST_MAX_SEQ_LEN, capacity);
                k.append(&data)?;
                v.append(&data)?;
                manager.caches.insert(
                    CacheKey::new(vec![token; TEST_PROMPT_LEN], None),
                    CacheElement {
                        cache: vec![Some(KvCache::Normal { k, v })],
                        recurrent_snapshots: None,
                        audio_hashes: None,
                        image_hashes: None,
                        video_hashes: None,
                    },
                );
            }
            let mut seqs = Vec::new();
            for token in [1, 2] {
                let MatchingCache::Normal {
                    normal,
                    toks,
                    offset,
                    ..
                } = hit(&mut manager, &[token; TEST_PROMPT_LEN])?;
                assert_eq!(offset, TEST_PROMPT_LEN - 1);
                let mut seq = sequence(normal, toks, offset);
                if draft {
                    *seq.normal_draft_cache() = seq.normal_cache_ref().to_vec();
                }
                seqs.push(seq);
            }
            let pipeline = TestPipeline::new(device.clone());
            let mut refs = seqs.iter_mut().collect::<Vec<_>>();
            NormalCacheManager.clone_in_cache(&pipeline, &mut refs, draft)?;
            let data = Tensor::from_vec(
                [
                    vec![9f32; TEST_KV_HEADS * TEST_KV_HEAD_DIM],
                    vec![7.; TEST_KV_HEADS * TEST_KV_HEAD_DIM],
                ]
                .concat(),
                (2, TEST_KV_HEADS, 1, TEST_KV_HEAD_DIM),
                &device,
            )?;
            append_tensor(&pipeline, &data)?;
            drop(refs);
            for extra_tokens in [0, block] {
                if extra_tokens != 0 {
                    let extension = Tensor::full(
                        5f32,
                        (2, TEST_KV_HEADS, extra_tokens, TEST_KV_HEAD_DIM),
                        &device,
                    )?;
                    append_tensor(&pipeline, &extension)?;
                }
                let mut refs = seqs.iter_mut().collect::<Vec<_>>();
                NormalCacheManager.clone_out_cache(&pipeline, &mut refs, draft);
                drop(refs);
                for (row, seq) in seqs.iter_mut().enumerate() {
                    let cache = if draft {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    let KvCache::Normal { k, v } = cache[0].as_ref().unwrap() else {
                        panic!("normal cache required")
                    };
                    assert_eq!(k.capacity_seq_len, block + extra_tokens);
                    assert_eq!(v.capacity_seq_len, block + extra_tokens);
                    let mut head = vec![[1., 2.][row]; (TEST_PROMPT_LEN - 1) * TEST_KV_HEAD_DIM];
                    head.extend(vec![[9., 7.][row]; TEST_KV_HEAD_DIM]);
                    head.extend(vec![5.; extra_tokens * TEST_KV_HEAD_DIM]);
                    assert_eq!(
                        values(cache[0].as_ref().unwrap())?,
                        head.repeat(TEST_KV_HEADS)
                    );
                }
            }
            for (row, retained) in manager.caches.values().enumerate() {
                let KvCache::Normal { k, v } = retained.cache[0].as_ref().unwrap() else {
                    panic!("normal cache required")
                };
                assert_eq!(k.capacity_seq_len, capacities[row]);
                assert_eq!(v.capacity_seq_len, capacities[row]);
                assert_eq!(retained.coverage_len(), TEST_PROMPT_LEN);
                assert_eq!(
                    values(retained.cache[0].as_ref().unwrap())?,
                    vec![[1., 2.][row]; TEST_KV_HEADS * TEST_PROMPT_LEN * TEST_KV_HEAD_DIM]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn batched_replay_with_different_capacities_on_cpu() -> Result<()> {
    batched_replay(Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn batched_replay_with_different_capacities_on_metal() -> Result<()> {
    batched_replay(Device::new_metal(0)?)
}

fn concurrent_replay(device: Device) -> Result<()> {
    let mut manager = manager(4, &device)?;
    let mut requests = Vec::new();
    for value in [9., 7.] {
        let MatchingCache::Normal {
            normal,
            toks,
            offset,
            ..
        } = hit(&mut manager, &[1, 2, 3, 4])?;
        requests.push((sequence(normal, toks, offset), value));
    }
    std::thread::scope(|scope| -> Result<()> {
        let handles = requests
            .into_iter()
            .map(|(mut seq, value)| {
                let device = device.clone();
                scope.spawn(move || -> Result<()> {
                    let pipeline = TestPipeline::new(device);
                    pipeline.clone_in_cache(&mut [&mut seq])?;
                    append(&pipeline, value)?;
                    assert_eq!(
                        values(seq.normal_cache_ref()[0].as_ref().unwrap())?,
                        [1., 2., 3.]
                    );
                    pipeline.clone_out_cache(&mut [&mut seq]);
                    assert_eq!(
                        values(seq.normal_cache_ref()[0].as_ref().unwrap())?,
                        [1., 2., 3., value]
                    );
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap()?;
        }
        Ok(())
    })?;
    let retained = manager.caches.values().next().unwrap();
    assert_eq!(retained.coverage_len(), 4);
    assert_eq!(
        values(retained.cache[0].as_ref().unwrap())?,
        [1., 2., 3., 4.]
    );
    let MatchingCache::Normal {
        normal,
        offset,
        toks,
        ..
    } = hit(&mut manager, &[1, 2, 3, 4])?;
    assert_eq!(offset, 3);
    assert_eq!(toks, [4]);
    assert_eq!(values(normal[0].as_ref().unwrap())?, [1., 2., 3.]);
    Ok(())
}

#[test]
fn concurrent_replay_clone_in_isolates_tensor_writes() -> Result<()> {
    concurrent_replay(Device::Cpu)
}

fn forward_prompt(model: &crate::models::llama::Llama, seq: &mut Sequence) -> Result<Tensor> {
    use crate::pipeline::{text_models_inputs_processor::get_prompt_input, ModelForwardContext};

    let tokens = seq.get_toks().to_vec();
    let inputs = get_prompt_input(
        vec![tokens.as_slice()],
        &[seq],
        &Device::Cpu,
        None,
        false,
        None,
        None,
        None,
    )
    .map_err(candle_core::Error::msg)?
    .inputs;
    let mut ctx = ModelForwardContext::new(
        &inputs.positions,
        &inputs.context_lens,
        &inputs.position_ids,
        None,
        &inputs.flash_meta,
    );
    model.forward(&inputs.input, &mut ctx)
}

#[test]
fn exact_replay_logits_match_fresh_prompt() -> Result<()> {
    use crate::{
        device_map::DummyDeviceMapper,
        layers::Activation,
        models::llama::{Config, Llama},
        paged_attention::AttentionImplementation,
        pipeline::{NormalLoadingMetadata, NormalModel},
    };

    let h = TEST_HIDDEN_SIZE;
    let shapes = [
        ("model.embed_tokens.weight", vec![TEST_VOCAB_SIZE, h]),
        ("model.norm.weight", vec![h]),
        ("model.layers.0.input_layernorm.weight", vec![h]),
        ("model.layers.0.post_attention_layernorm.weight", vec![h]),
        ("model.layers.0.self_attn.q_proj.weight", vec![h, h]),
        ("model.layers.0.self_attn.k_proj.weight", vec![h, h]),
        ("model.layers.0.self_attn.v_proj.weight", vec![h, h]),
        ("model.layers.0.self_attn.o_proj.weight", vec![h, h]),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![TEST_INTERMEDIATE_SIZE, h],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![TEST_INTERMEDIATE_SIZE, h],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![h, TEST_INTERMEDIATE_SIZE],
        ),
        ("lm_head.weight", vec![TEST_VOCAB_SIZE, h]),
    ];
    let tensors = shapes
        .into_iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            let count = shape.iter().product::<usize>();
            let data = (0..count)
                .map(|i| {
                    if shape.len() == 1 {
                        1.
                    } else {
                        f32::from(u16::try_from(i + index + 1).unwrap()).sin()
                    }
                })
                .collect::<Vec<_>>();
            Ok((
                name.to_string(),
                Tensor::from_vec(data, shape, &Device::Cpu)?,
            ))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let model = Llama::new(
        &Config {
            hidden_act: Activation::Silu,
            hidden_size: h,
            intermediate_size: TEST_INTERMEDIATE_SIZE,
            vocab_size: TEST_VOCAB_SIZE,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            max_position_embeddings: TEST_MAX_SEQ_LEN,
            ..Default::default()
        },
        mistralrs_quant::ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu),
        false,
        NormalLoadingMetadata {
            mapper: Box::new(DummyDeviceMapper {
                nm_device: Device::Cpu,
            }),
            loading_isq: false,
            real_device: Device::Cpu,
            multi_progress: Arc::new(indicatif::MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        },
        AttentionImplementation::Eager,
    )?;
    let pipeline = TestPipeline {
        cache: model.cache().clone(),
        device: Device::Cpu,
    };
    let prompt = [1, 2, 3, 4];
    let mut fresh = sequence(vec![None], prompt.to_vec(), 0);
    let expected = forward_prompt(&model, &mut fresh)?;
    pipeline.clone_out_cache(&mut [&mut fresh]);
    let mut manager = PrefixCacheManagerV2::new(1, false, false);
    manager.add_sequence(&mut fresh, None);
    for _ in 0..3 {
        let MatchingCache::Normal {
            normal,
            toks,
            offset,
            ..
        } = hit(&mut manager, &prompt)?;
        let mut replay = sequence(normal, toks, offset);
        pipeline.clone_in_cache(&mut [&mut replay])?;
        let actual = forward_prompt(&model, &mut replay)?;
        let error = (&actual - &expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(error < LOGITS_TOLERANCE, "logits error: {error}");
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn concurrent_replay_clone_in_isolates_metal_tensor_writes() -> Result<()> {
    concurrent_replay(Device::new_metal(0)?)
}

#[test]
fn restored_request_survives_eviction_without_mutating_retained_storage() -> Result<()> {
    let mut manager = manager(4, &Device::Cpu)?;
    let original = manager.caches.values().next().unwrap().clone();
    let MatchingCache::Normal {
        normal,
        toks,
        offset,
        ..
    } = hit(&mut manager, &[1, 2, 3, 4])?;
    let mut seq = sequence(normal, toks, offset);
    assert_eq!(manager.evict_all_caches()?, 1);
    let pipeline = TestPipeline::new(Device::Cpu);
    pipeline.clone_in_cache(&mut [&mut seq])?;
    append(&pipeline, 9.)?;
    drop(seq);
    assert_eq!(original.coverage_len(), 4);
    assert_eq!(
        values(original.cache[0].as_ref().unwrap())?,
        [1., 2., 3., 4.]
    );
    assert!(search(&mut manager, &[1, 2, 3, 4])?.is_none());
    Ok(())
}
