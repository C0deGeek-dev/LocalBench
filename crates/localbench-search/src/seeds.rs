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
}

/// How VRAM-constrained the run looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VramRisk {
    High,
    Medium,
    Normal,
}

/// The memory-mapping recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmapRecommendation {
    pub mlock: bool,
    pub no_mmap: bool,
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

    let mmap_recommendation = if host.available_ram_gb > 0.0 && host.available_ram_gb < 8.0 {
        MmapRecommendation {
            mlock: false,
            no_mmap: false,
        }
    } else {
        MmapRecommendation {
            mlock: true,
            no_mmap: true,
        }
    };

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
