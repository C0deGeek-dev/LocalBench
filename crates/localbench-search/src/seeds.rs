//! Smart seeds: hardware-informed starting candidates for the search, derived
//! from injected host facts (never probed here, so the logic stays pure).

use serde::{Deserialize, Serialize};

use crate::candidate::Profile;
use crate::space::SearchSpace;

/// The host facts seeding reads. Zero/empty means "unknown" and falls back
/// conservatively.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostFacts {
    pub vram_gb: u32,
    pub logical_cores: u32,
    pub available_ram_gb: f64,
    pub gguf_size_gb: f64,
    /// Memory the host can still commit — RAM plus page file on Windows, RAM
    /// plus free swap elsewhere — before the server starts. `0` = unknown.
    pub commit_available_gb: f64,
    /// Model bytes the build keeps memory-mapped even when it loads the model
    /// without mmap (a per-layer embedding table it reads on demand), so they
    /// never become private memory.
    pub lazy_mapped_gb: f64,
    /// Whether the GPU driver backs device memory with host commit (Windows
    /// WDDM), so the part of the model on the GPU also counts against commit.
    pub vram_backed_by_host_commit: bool,
}

/// How VRAM-constrained the run looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VramRisk {
    High,
    Medium,
    Normal,
}

/// Why a memory-pinning candidate is not tried on this host.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "limit", rename_all = "snake_case")]
pub enum PinningLimit {
    /// Less free RAM than the candidate needs to spare.
    Ram { needed_gb: f64, available_gb: f64 },
    /// Loading the model without mmap would commit more memory (RAM plus page
    /// file or swap) than the host has free.
    Commit { needed_gb: f64, available_gb: f64 },
}

/// The memory-mapping recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MmapRecommendation {
    pub mlock: bool,
    pub no_mmap: bool,
    /// Why `mlock` is withheld, when it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlock_limit: Option<PinningLimit>,
    /// Why `no_mmap` is withheld, when it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_mmap_limit: Option<PinningLimit>,
}

/// The seeded starting candidates per axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmartSeeds {
    pub offload_candidates: Vec<i64>,
    pub ubatch_candidates: Vec<i64>,
    pub batch_candidates: Vec<i64>,
    pub thread_candidates: Vec<i64>,
    pub mmap_recommendation: MmapRecommendation,
    pub vram_risk: VramRisk,
    /// Human-readable notes on what the seeding assumed.
    pub assumptions: Vec<String>,
    pub vram_gb: u32,
    pub gguf_size_gb: f64,
}

impl Default for SmartSeeds {
    fn default() -> Self {
        Self {
            offload_candidates: Vec::new(),
            ubatch_candidates: Vec::new(),
            batch_candidates: Vec::new(),
            thread_candidates: Vec::new(),
            mmap_recommendation: MmapRecommendation {
                mlock: true,
                no_mmap: true,
                mlock_limit: None,
                no_mmap_limit: None,
            },
            vram_risk: VramRisk::Normal,
            assumptions: Vec::new(),
            vram_gb: 0,
            gguf_size_gb: 0.0,
        }
    }
}

/// Derive the smart seeds for a search space on a host.
#[must_use]
pub fn resolve_smart_seeds(space: &SearchSpace, host: HostFacts, profile: Profile) -> SmartSeeds {
    let offload = if space.is_moe {
        let base = space.baseline_n_cpu_moe;
        let upper = space.moe_upper;
        // On a small card, bias toward MORE CPU offload first (safer fits).
        let near: Vec<i64> = if host.vram_gb > 0 && host.vram_gb <= 16 {
            vec![base, base + 10, base + 5, base + 15, base - 5]
        } else {
            vec![base, base - 10, base - 5, base + 5, base + 10]
        };
        dedup(near.into_iter().filter(|n| *n >= 0 && *n <= upper))
    } else {
        // Dense models seed no offload ladder here. The only consumer of
        // `offload_candidates` is `moe_candidate_values`, which never runs for a
        // dense space; the dense `-ngl` recovery ladder is owned entirely by
        // `dense_recovery_candidates` in the VRAM-fit phase, anchored on the real
        // layer count. Keeping a second dense ladder here (the old 999-sentinel
        // one) was dead, redundant, and wrong (LocalHub#76).
        Vec::new()
    };

    let (ubatches, batches) = if host.vram_gb > 0 && host.vram_gb <= 12 {
        (vec![256, 512], vec![512, 1024])
    } else {
        (vec![512, 1024, 256], vec![1024, 2048, 512])
    };

    let cores = i64::from(host.logical_cores);
    let threads = match profile {
        // Balanced: always leave headroom for the agent/OS.
        Profile::Balanced => dedup(
            [cores - 2, cores * 3 / 4, cores / 2]
                .into_iter()
                .map(|t| t.max(1))
                .filter(|t| *t < cores),
        ),
        // Pure: sweep up to every core.
        Profile::Pure => dedup(
            [cores / 2, cores * 3 / 4, cores]
                .into_iter()
                .map(|t| t.max(1)),
        ),
    };

    let mmap_recommendation = mmap_recommendation(host);

    let vram_risk = if host.vram_gb > 0 && host.vram_gb <= 12 {
        VramRisk::High
    } else if host.vram_gb > 0 && host.vram_gb <= 16 {
        VramRisk::Medium
    } else {
        VramRisk::Normal
    };

    let mut assumptions = Vec::new();
    if host.vram_gb > 0 {
        assumptions.push(format!("detected VRAM {}GB", host.vram_gb));
    }
    if host.available_ram_gb > 0.0 {
        assumptions.push(format!("available RAM {:.1}GB", host.available_ram_gb));
    }
    if host.gguf_size_gb > 0.0 {
        assumptions.push(format!("GGUF size {:.1}GB", host.gguf_size_gb));
    }
    if host.commit_available_gb > 0.0 {
        assumptions.push(format!(
            "commit headroom {:.1}GB (loading without mmap needs ~{:.1}GB)",
            host.commit_available_gb,
            no_mmap_commit_gb(host)
        ));
    }
    if space.is_moe {
        assumptions.push(format!(
            "MoE expert CPU-offload boundary near NCpuMoe={}",
            space.baseline_n_cpu_moe
        ));
    }

    SmartSeeds {
        offload_candidates: offload,
        ubatch_candidates: ubatches,
        batch_candidates: batches,
        thread_candidates: threads,
        mmap_recommendation,
        vram_risk,
        assumptions,
        vram_gb: host.vram_gb,
        gguf_size_gb: host.gguf_size_gb,
    }
}

/// RAM a host must keep free, beyond what it pins, before the tuner tries
/// loading the model into RAM (`NoMmap`) or locking it there (`Mlock`). The
/// same margin is kept on commit headroom.
const PINNING_HEADROOM_GB: f64 = 8.0;

/// Memory a load without mmap commits for the model's weights, in GB.
///
/// Without mmap the weights become private memory, except the tensors the
/// build keeps mapped anyway (a lazily read per-layer embedding table). The
/// part on the GPU is not private host memory — unless the GPU driver backs
/// device memory with host commit (Windows WDDM), where the whole non-mapped
/// model counts: the CPU part as private pages, the GPU part as the driver's
/// backing. The model's size is the best estimate before placement is known.
#[must_use]
pub fn no_mmap_commit_gb(host: HostFacts) -> f64 {
    let size = host.gguf_size_gb.max(0.0);
    let private = size - host.lazy_mapped_gb.clamp(0.0, size);
    let on_gpu = if host.vram_backed_by_host_commit {
        0.0
    } else {
        f64::from(host.vram_gb)
    };
    (private - on_gpu).max(0.0)
}

/// Which memory-pinning candidates are worth measuring on this host.
///
/// Loading without mmap needs working room in RAM and enough commit headroom
/// for the weights it makes private; locking holds the whole model in RAM, so
/// it is only tried when the full GGUF (every shard) fits beside the headroom.
/// Unknown RAM, commit, or size keeps the candidate: the trial itself is the
/// evidence.
fn mmap_recommendation(host: HostFacts) -> MmapRecommendation {
    let ram_known = host.available_ram_gb > 0.0;
    let size_known = host.gguf_size_gb > 0.0;
    let ram_limit =
        (ram_known && host.available_ram_gb < PINNING_HEADROOM_GB).then_some(PinningLimit::Ram {
            needed_gb: PINNING_HEADROOM_GB,
            available_gb: host.available_ram_gb,
        });
    let commit_needed = no_mmap_commit_gb(host) + PINNING_HEADROOM_GB;
    let commit_limit =
        (host.commit_available_gb > 0.0 && size_known && host.commit_available_gb < commit_needed)
            .then_some(PinningLimit::Commit {
                needed_gb: commit_needed,
                available_gb: host.commit_available_gb,
            });
    let lock_needed = host.gguf_size_gb + PINNING_HEADROOM_GB;
    let lock_limit = ram_limit.or_else(|| {
        (ram_known && size_known && host.available_ram_gb < lock_needed).then_some(
            PinningLimit::Ram {
                needed_gb: lock_needed,
                available_gb: host.available_ram_gb,
            },
        )
    });
    let no_mmap_limit = ram_limit.or(commit_limit);
    MmapRecommendation {
        mlock: lock_limit.is_none(),
        no_mmap: no_mmap_limit.is_none(),
        mlock_limit: lock_limit,
        no_mmap_limit,
    }
}

fn dedup(values: impl Iterator<Item = i64>) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    for value in values {
        if !out.contains(&value) {
            out.push(value);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{resolve_search_space, ModelAxes};

    fn moe_space(base: i64) -> SearchSpace {
        resolve_search_space(
            &ModelAxes {
                n_cpu_moe: Some(base),
                ..ModelAxes::default()
            },
            128,
            -1,
        )
    }

    #[test]
    fn small_card_biases_toward_more_cpu_offload_first() {
        let space = moe_space(35);
        let small = resolve_smart_seeds(
            &space,
            HostFacts {
                vram_gb: 12,
                logical_cores: 16,
                ..HostFacts::default()
            },
            Profile::Pure,
        );
        assert_eq!(small.offload_candidates, vec![35, 45, 40, 50, 30]);
        assert_eq!(small.vram_risk, VramRisk::High);
        assert_eq!(small.ubatch_candidates, vec![256, 512]);

        let big = resolve_smart_seeds(
            &space,
            HostFacts {
                vram_gb: 24,
                logical_cores: 16,
                ..HostFacts::default()
            },
            Profile::Pure,
        );
        assert_eq!(big.offload_candidates, vec![35, 25, 30, 40, 45]);
        assert_eq!(big.vram_risk, VramRisk::Normal);
        assert_eq!(big.ubatch_candidates, vec![512, 1024, 256]);
    }

    #[test]
    fn dense_models_seed_no_redundant_offload_ladder() {
        // A dense space consumes no `offload_candidates` (that field only feeds
        // the MoE `--n-cpu-moe` sweep); the dense `-ngl` recovery ladder lives in
        // `dense_recovery_candidates`. Seeding a second one here was dead code
        // (LocalHub#76).
        let space = resolve_search_space(
            &ModelAxes {
                n_gpu_layers: Some(48),
                ..ModelAxes::default()
            },
            0,
            65,
        );
        assert!(!space.is_moe, "expert_count 0 ⇒ dense");
        let seeds = resolve_smart_seeds(
            &space,
            HostFacts {
                logical_cores: 8,
                ..HostFacts::default()
            },
            Profile::Pure,
        );
        assert!(
            seeds.offload_candidates.is_empty(),
            "dense models seed no offload ladder: {:?}",
            seeds.offload_candidates
        );
    }

    #[test]
    fn balanced_threads_always_reserve_headroom() {
        let space = moe_space(35);
        let host = HostFacts {
            logical_cores: 16,
            ..HostFacts::default()
        };
        let balanced = resolve_smart_seeds(&space, host, Profile::Balanced);
        assert_eq!(balanced.thread_candidates, vec![14, 12, 8]);
        assert!(balanced.thread_candidates.iter().all(|t| *t < 16));
        let pure = resolve_smart_seeds(&space, host, Profile::Pure);
        assert_eq!(pure.thread_candidates, vec![8, 12, 16]);
    }

    #[test]
    fn a_model_larger_than_ram_is_never_locked_but_may_load_without_mmap() {
        let space = moe_space(35);
        let big = resolve_smart_seeds(
            &space,
            HostFacts {
                available_ram_gb: 58.0,
                gguf_size_gb: 105.6,
                logical_cores: 32,
                ..HostFacts::default()
            },
            Profile::Pure,
        );
        assert!(!big.mmap_recommendation.mlock);
        assert!(big.mmap_recommendation.no_mmap);
        let small = resolve_smart_seeds(
            &space,
            HostFacts {
                available_ram_gb: 58.0,
                gguf_size_gb: 22.8,
                logical_cores: 32,
                ..HostFacts::default()
            },
            Profile::Pure,
        );
        assert!(small.mmap_recommendation.mlock);
        assert!(small.mmap_recommendation.no_mmap);
    }

    /// The live Flash-Next case: a 105.6 GB model whose 50.7 GB per-layer
    /// embedding table stays mapped, on a 24 GB Windows card.
    fn flash_next(commit_available_gb: f64, windows: bool) -> HostFacts {
        HostFacts {
            vram_gb: 24,
            logical_cores: 32,
            available_ram_gb: 58.0,
            gguf_size_gb: 105.6,
            commit_available_gb,
            lazy_mapped_gb: 50.7,
            vram_backed_by_host_commit: windows,
        }
    }

    #[test]
    fn loading_without_mmap_needs_commit_for_everything_not_mapped() {
        // Windows: the GPU part is backed by host commit, so the whole
        // non-mapped model (54.9 GB) counts.
        assert!((no_mmap_commit_gb(flash_next(0.0, true)) - 54.9).abs() < 1e-9);
        // Elsewhere only the part that cannot sit on the GPU is private.
        assert!((no_mmap_commit_gb(flash_next(0.0, false)) - 30.9).abs() < 1e-9);
        // Nothing read lazily: the whole file.
        let eager = HostFacts {
            lazy_mapped_gb: 0.0,
            ..flash_next(0.0, true)
        };
        assert!((no_mmap_commit_gb(eager) - 105.6).abs() < 1e-9);
        // A model that fits the card commits nothing extra off Windows.
        let small = HostFacts {
            gguf_size_gb: 16.0,
            lazy_mapped_gb: 0.0,
            ..flash_next(0.0, false)
        };
        assert_eq!(no_mmap_commit_gb(small), 0.0);
    }

    #[test]
    fn a_host_without_the_commit_headroom_is_not_offered_no_mmap() {
        let space = moe_space(35);
        // The failing live host: ~55 GB of commit left before the server
        // started, while loading without mmap needs 54.9 GB plus the margin.
        let tight = resolve_smart_seeds(&space, flash_next(55.0, true), Profile::Pure);
        assert!(!tight.mmap_recommendation.no_mmap);
        let Some(PinningLimit::Commit {
            needed_gb,
            available_gb,
        }) = tight.mmap_recommendation.no_mmap_limit
        else {
            panic!("{:?}", tight.mmap_recommendation);
        };
        assert!((needed_gb - (54.9 + PINNING_HEADROOM_GB)).abs() < 1e-9);
        assert_eq!(available_gb, 55.0);
        // Locking is a RAM question: the whole model does not fit in 58 GB.
        assert!(!tight.mmap_recommendation.mlock);
        assert!(matches!(
            tight.mmap_recommendation.mlock_limit,
            Some(PinningLimit::Ram { .. })
        ));
        assert!(tight
            .assumptions
            .iter()
            .any(|a| a == "commit headroom 55.0GB (loading without mmap needs ~54.9GB)"));

        // A 128 GB page file leaves room.
        let roomy = resolve_smart_seeds(&space, flash_next(150.0, true), Profile::Pure);
        assert!(roomy.mmap_recommendation.no_mmap);
        assert_eq!(roomy.mmap_recommendation.no_mmap_limit, None);
        // The same headroom suffices off Windows, where the GPU part is not committed.
        let linux = resolve_smart_seeds(&space, flash_next(55.0, false), Profile::Pure);
        assert!(linux.mmap_recommendation.no_mmap);
        // Unknown commit keeps the candidate: the trial is the evidence.
        let unknown = resolve_smart_seeds(&space, flash_next(0.0, true), Profile::Pure);
        assert!(unknown.mmap_recommendation.no_mmap);
    }

    #[test]
    fn locking_a_mapped_model_needs_ram_not_commit() {
        let space = moe_space(35);
        // A 22.8 GB model on a host with RAM to spare but almost no commit:
        // loading it privately is out, locking the mapped file is not.
        let host = HostFacts {
            vram_gb: 24,
            logical_cores: 16,
            available_ram_gb: 40.0,
            gguf_size_gb: 22.8,
            commit_available_gb: 20.0,
            lazy_mapped_gb: 0.0,
            vram_backed_by_host_commit: true,
        };
        let seeds = resolve_smart_seeds(&space, host, Profile::Pure);
        assert!(seeds.mmap_recommendation.mlock);
        assert!(!seeds.mmap_recommendation.no_mmap);
    }

    #[test]
    fn low_ram_disables_the_mlock_recommendation() {
        let space = moe_space(35);
        let tight = resolve_smart_seeds(
            &space,
            HostFacts {
                available_ram_gb: 6.5,
                logical_cores: 8,
                ..HostFacts::default()
            },
            Profile::Pure,
        );
        assert!(!tight.mmap_recommendation.mlock);
        assert!(!tight.mmap_recommendation.no_mmap);
        assert!(tight
            .assumptions
            .iter()
            .any(|assumption| assumption == "available RAM 6.5GB"));
        let roomy = resolve_smart_seeds(
            &space,
            HostFacts {
                available_ram_gb: 32.0,
                logical_cores: 8,
                ..HostFacts::default()
            },
            Profile::Pure,
        );
        assert!(roomy.mmap_recommendation.mlock);
        assert!(roomy.mmap_recommendation.no_mmap);
        assert!(roomy
            .assumptions
            .iter()
            .any(|assumption| assumption == "available RAM 32.0GB"));
    }
}
