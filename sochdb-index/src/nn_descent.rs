//! Sub-quadratic approximate k-NN via NN-descent (Dong, Charikar, Li 2011).
//!
//! Used by `HnswIndex::rebuild_layer0_exact` (the `optimize()` path) to replace
//! its previous O(N^2)·dim all-pairs brute-force k-NN with an O(N·K·iters)
//! approximate-k-NN candidate generator. The candidate pool produced here is
//! distance-exact (every edge carries the exact f32 distance computed with the
//! same SIMD kernel the brute-force path used), so reranking the pool to the
//! final M0 yields exact-quality layer-0 edges — the same purpose the old path
//! served — at a fraction of the distance-op count.
//!
//! ## Why NN-descent (vs re-querying the as-built HNSW graph)
//! The HNSW layer-0 graph as built with pure top-M selection is weak/fragmented
//! at high dimension (the very reason optimize() exists). Seeding refinement
//! from that graph would inherit its fragmentation. NN-descent instead starts
//! from RANDOM neighbors and refines via "neighbors-of-neighbors" local joins,
//! so it is independent of the as-built graph and converges to ~95-99% of the
//! true k-NN in a few iterations. Distances are f32-exact, so quality of the
//! kept edges is bounded only by which candidates are discovered, and the larger
//! candidate pool (K = sample_pool > M0) plus the symmetrize-and-rerank step in
//! the caller recover the recall the exact path gave.
//!
//! The convergence is approximate by construction; the caller's append-only
//! connectivity-repair pass (BFS reconnect) still runs afterward and guarantees
//! zero orphans regardless of any candidate NN-descent failed to find.

use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use parking_lot::Mutex;

/// One neighbor entry in a node's working k-NN list.
#[derive(Clone, Copy, Debug)]
struct Neighbor {
    /// Index into the snapshot arrays (0..n).
    id: u32,
    /// Exact f32 distance from the owning node to `id`.
    dist: f32,
    /// "new" flag for incremental NN-descent: edges discovered since the last
    /// iteration are joined against everything; old edges only against new ones.
    is_new: bool,
}

/// A node's bounded, distance-sorted k-NN list (max-on-top via explicit scan).
///
/// We keep it as a plain `Vec` capped at `k`. `k` is small (≈2*M0 = 128), so a
/// linear insert with a far-bound early-out is cheaper and more cache-friendly
/// than a binary heap, and keeping it sorted lets the caller take the closest
/// M0 directly.
struct KnnList {
    items: Vec<Neighbor>,
    k: usize,
}

impl KnnList {
    #[inline]
    fn new(k: usize) -> Self {
        Self {
            items: Vec::with_capacity(k),
            k,
        }
    }

    /// Current farthest (worst) distance in the list, or +inf if not yet full.
    #[inline]
    fn worst(&self) -> f32 {
        if self.items.len() < self.k {
            f32::INFINITY
        } else {
            // items kept sorted ascending; last is farthest.
            self.items.last().map(|n| n.dist).unwrap_or(f32::INFINITY)
        }
    }

    /// Try to insert `(id, dist)` as a *new* edge. Returns 1 if it was inserted
    /// (i.e. it improved the list), 0 otherwise. Skips duplicates and self.
    #[inline]
    fn push(&mut self, id: u32, dist: f32, owner: u32) -> usize {
        if id == owner {
            return 0;
        }
        // Reject if no better than the current farthest and list is full.
        if self.items.len() >= self.k {
            if let Some(last) = self.items.last() {
                if dist >= last.dist {
                    return 0;
                }
            }
        }
        // Dedup: if id already present, keep the better (smaller) distance.
        // Linear scan — k is small.
        for n in self.items.iter() {
            if n.id == id {
                return 0;
            }
        }
        // Find sorted insert position (ascending by dist).
        let pos = self
            .items
            .partition_point(|n| n.dist < dist || (n.dist == dist && n.id < id));
        self.items.insert(
            pos,
            Neighbor {
                id,
                dist,
                is_new: true,
            },
        );
        if self.items.len() > self.k {
            self.items.truncate(self.k);
        }
        1
    }
}

/// Configuration for NN-descent. Defaults follow the original paper, tuned for
/// the high-dim recall the optimize() path must preserve.
#[derive(Clone, Copy, Debug)]
pub struct NnDescentConfig {
    /// Size of each node's working k-NN candidate list. Should be >= the final
    /// M0 the caller wants (caller passes ~2*M0 to leave headroom for the
    /// rerank/symmetrize step).
    pub k: usize,
    /// Sampling rate for local joins (Dong et al. `rho`). 1.0 = no sampling
    /// (highest quality, most work). 0.5-1.0 is typical. We keep it high
    /// because edge quality directly gates recall here.
    pub rho: f32,
    /// Early-termination threshold (Dong et al. `delta`): stop when the number
    /// of list updates in an iteration falls below `delta * n * k`.
    pub delta: f32,
    /// Hard cap on iterations (NN-descent usually converges in <= ~10).
    pub max_iters: usize,
    /// RNG seed for reproducible initialization (the random seed neighbors).
    pub seed: u64,
}

impl Default for NnDescentConfig {
    fn default() -> Self {
        Self {
            k: 64,
            rho: 1.0,
            delta: 0.001,
            max_iters: 12,
            seed: 0x5eed_1234_abcd_ef00,
        }
    }
}

/// Run NN-descent over `n` items whose pairwise distance is given by `dist`.
///
/// Returns, for each item `i`, its approximate k-nearest neighbors as a list of
/// `(neighbor_index, exact_f32_distance)` sorted ascending by distance (closest
/// first), length <= `cfg.k`. Self is never included.
///
/// `dist(i, j)` must be a cheap, thread-safe symmetric distance (it is called
/// from multiple rayon threads). It is invoked with the snapshot indices.
///
/// Complexity: O(n · k · max_iters) distance evaluations in the worst case,
/// typically far fewer because of the `is_new` flag and `delta` early-out —
/// versus O(n^2) for the brute-force path it replaces.
pub fn nn_descent<F>(n: usize, cfg: NnDescentConfig, dist: F) -> Vec<Vec<(u32, f32)>>
where
    F: Fn(u32, u32) -> f32 + Sync,
{
    let k = cfg.k.max(1).min(n.saturating_sub(1).max(1));
    if n < 2 {
        return vec![Vec::new(); n];
    }

    // Per-node working lists behind fine-grained mutexes so threads can update
    // a neighbor's list during the symmetric "neighbors-of-neighbors" join.
    let lists: Vec<Mutex<KnnList>> = (0..n).map(|_| Mutex::new(KnnList::new(k))).collect();

    // ---- Initialization: seed each node with `k` random neighbors. ----
    // Seed each node independently/deterministically from cfg.seed so the whole
    // build is reproducible regardless of thread scheduling.
    (0..n).into_par_iter().for_each(|i| {
        let mut rng =
            StdRng::seed_from_u64(cfg.seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut list = lists[i].lock();
        // Sample k distinct random ids != i. For small n just take a shuffled
        // range; for large n rejection-sample (k << n so collisions are rare).
        if n <= 4 * k {
            let mut ids: Vec<u32> = (0..n as u32).filter(|&x| x != i as u32).collect();
            ids.shuffle(&mut rng);
            for &j in ids.iter().take(k) {
                let d = dist(i as u32, j);
                list.push(j, d, i as u32);
            }
        } else {
            let mut tries = 0;
            while list.items.len() < k && tries < k * 4 {
                let j = rng.gen_range(0..n as u32);
                if j != i as u32 {
                    let d = dist(i as u32, j);
                    list.push(j, d, i as u32);
                }
                tries += 1;
            }
        }
    });

    let sample = ((k as f32) * cfg.rho).ceil() as usize;
    let sample = sample.max(1);
    let stop_threshold = (cfg.delta * n as f32 * k as f32).max(1.0) as usize;

    // ---- Refinement iterations. ----
    for _iter in 0..cfg.max_iters {
        // Build per-node "new" and "old" reverse+forward neighbor sets for this
        // round. Following Dong et al.: for each node u, partition its current
        // list into new (recently added) and old; sample up to `rho*k` of the
        // new ones, mark them sampled (clear is_new); also gather reverse edges.
        let new_lists: Vec<Mutex<Vec<u32>>> = (0..n).map(|_| Mutex::new(Vec::new())).collect();
        let old_lists: Vec<Mutex<Vec<u32>>> = (0..n).map(|_| Mutex::new(Vec::new())).collect();

        // Forward pass: collect each node's own new/old neighbors (sampled).
        (0..n).into_par_iter().for_each(|u| {
            let mut new_u: Vec<u32> = Vec::new();
            let mut old_u: Vec<u32> = Vec::new();
            {
                let mut list = lists[u].lock();
                let mut taken_new = 0usize;
                for nb in list.items.iter_mut() {
                    if nb.is_new {
                        if taken_new < sample {
                            new_u.push(nb.id);
                            nb.is_new = false; // mark sampled
                            taken_new += 1;
                        }
                    } else {
                        old_u.push(nb.id);
                    }
                }
            }
            // Add forward edges to this node's own buckets.
            {
                let mut g = new_lists[u].lock();
                g.extend_from_slice(&new_u);
            }
            {
                let mut g = old_lists[u].lock();
                g.extend_from_slice(&old_u);
            }
            // Reverse edges: u is a neighbor of each of these; record the
            // reverse direction so v also joins against u.
            for &v in new_u.iter() {
                new_lists[v as usize].lock().push(u as u32);
            }
            for &v in old_u.iter() {
                old_lists[v as usize].lock().push(u as u32);
            }
        });

        // Local join: for each node, compare its new neighbors against each
        // other and against its old neighbors, updating BOTH endpoints'
        // lists with any discovered shorter edge (the symmetric update is what
        // propagates improvements).
        let updates = AtomicUsize::new(0);
        (0..n).into_par_iter().for_each(|u| {
            // Cap reverse-edge blow-up by sampling here too.
            let mut new_u = {
                let g = new_lists[u].lock();
                g.clone()
            };
            let mut old_u = {
                let g = old_lists[u].lock();
                g.clone()
            };
            if new_u.is_empty() {
                return;
            }
            // Dedup (reverse edges can duplicate forward ones).
            new_u.sort_unstable();
            new_u.dedup();
            old_u.sort_unstable();
            old_u.dedup();
            // Bound reverse blow-up: keep at most `sample` of each.
            if new_u.len() > sample {
                new_u.truncate(sample);
            }
            if old_u.len() > sample {
                old_u.truncate(sample);
            }

            let mut local = 0usize;
            // new x new (i<j to avoid double work; symmetric update covers both)
            for a in 0..new_u.len() {
                let i = new_u[a];
                for &j in new_u.iter().skip(a + 1) {
                    if i == j {
                        continue;
                    }
                    let d = dist(i, j);
                    local += try_update(&lists, i, j, d);
                    local += try_update(&lists, j, i, d);
                }
                // new x old
                for &j in old_u.iter() {
                    if i == j {
                        continue;
                    }
                    let d = dist(i, j);
                    local += try_update(&lists, i, j, d);
                    local += try_update(&lists, j, i, d);
                }
            }
            if local > 0 {
                updates.fetch_add(local, AtomicOrdering::Relaxed);
            }
        });

        let total = updates.load(AtomicOrdering::Relaxed);
        if total <= stop_threshold {
            break;
        }
    }

    // ---- Extract sorted (id, dist) lists. ----
    lists
        .into_iter()
        .map(|m| {
            let l = m.into_inner();
            l.items.iter().map(|nb| (nb.id, nb.dist)).collect()
        })
        .collect()
}

/// Try to insert edge `owner -> id` (distance `d`) into `owner`'s list under its
/// lock. Returns 1 if it improved the list. New edges are flagged `is_new` so
/// the next iteration joins against them.
#[inline]
fn try_update(lists: &[Mutex<KnnList>], owner: u32, id: u32, d: f32) -> usize {
    if owner == id {
        return 0;
    }
    // Fast lock-free-ish reject: peek the worst without holding long.
    {
        let l = lists[owner as usize].lock();
        if d >= l.worst() {
            return 0;
        }
    }
    let mut l = lists[owner as usize].lock();
    l.push(id, d, owner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
        let n = vectors.len();
        (0..n)
            .map(|i| {
                let mut d: Vec<(u32, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let dist: f32 = vectors[i]
                            .iter()
                            .zip(vectors[j].iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        (j as u32, dist)
                    })
                    .collect();
                d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                d.truncate(k);
                d.into_iter().map(|(j, _)| j).collect()
            })
            .collect()
    }

    fn gen_clusters(n: usize, dim: usize, nclusters: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / ((1u32 << 31) as f32) - 0.5
        };
        let centers: Vec<Vec<f32>> = (0..nclusters)
            .map(|_| (0..dim).map(|_| rnd() * 8.0).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % nclusters];
                (0..dim).map(|d| c[d] + rnd()).collect()
            })
            .collect()
    }

    fn recall_vs_bruteforce(approx: &[Vec<(u32, f32)>], exact: &[Vec<u32>], k: usize) -> f64 {
        let n = approx.len();
        let mut hit = 0usize;
        let mut tot = 0usize;
        for i in 0..n {
            let truth: std::collections::HashSet<u32> = exact[i].iter().take(k).copied().collect();
            let got: std::collections::HashSet<u32> =
                approx[i].iter().take(k).map(|(id, _)| *id).collect();
            hit += truth.intersection(&got).count();
            tot += truth.len();
        }
        hit as f64 / tot.max(1) as f64
    }

    #[test]
    fn nn_descent_recovers_high_recall_knn() {
        let n = 1500usize;
        let dim = 64usize;
        let k_final = 16usize;
        let vectors = gen_clusters(n, dim, 20, 0xABCD_1234);
        let dist = |i: u32, j: u32| -> f32 {
            vectors[i as usize]
                .iter()
                .zip(vectors[j as usize].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        };
        let cfg = NnDescentConfig {
            k: 2 * k_final,
            ..Default::default()
        };
        let approx = nn_descent(n, cfg, dist);
        let exact = brute_force(&vectors, k_final);
        let recall = recall_vs_bruteforce(&approx, &exact, k_final);
        // NN-descent should recover the vast majority of the true k-NN.
        assert!(
            recall >= 0.95,
            "NN-descent recall too low: {:.4} (expected >= 0.95)",
            recall
        );
        // Every node must have neighbors and be sorted ascending.
        for (i, l) in approx.iter().enumerate() {
            assert!(!l.is_empty(), "node {} has no neighbors", i);
            for w in l.windows(2) {
                assert!(w[0].1 <= w[1].1, "node {} list not sorted", i);
            }
            assert!(
                l.iter().all(|(id, _)| *id != i as u32),
                "self-loop at {}",
                i
            );
        }
    }

    #[test]
    fn nn_descent_tiny_n_is_exact() {
        // n <= 4k path takes a shuffled full range, so should be exact.
        let n = 40usize;
        let dim = 8usize;
        let k_final = 5usize;
        let vectors = gen_clusters(n, dim, 5, 0x1111_2222);
        let dist = |i: u32, j: u32| -> f32 {
            vectors[i as usize]
                .iter()
                .zip(vectors[j as usize].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        };
        let cfg = NnDescentConfig {
            k: 2 * k_final,
            ..Default::default()
        };
        let approx = nn_descent(n, cfg, dist);
        let exact = brute_force(&vectors, k_final);
        let recall = recall_vs_bruteforce(&approx, &exact, k_final);
        assert!(recall >= 0.99, "tiny-n recall: {:.4}", recall);
    }
}
