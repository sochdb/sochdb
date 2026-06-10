//! Per-namespace lazy hydrate/evict — resident memory tracks active tenants.

use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct LazyNamespaceConfig {
    pub max_resident: usize,
    pub idle_evict_after: Duration,
}

impl Default for LazyNamespaceConfig {
    fn default() -> Self {
        Self {
            max_resident: 1024,
            idle_evict_after: Duration::from_secs(300),
        }
    }
}

#[derive(Debug)]
struct NamespaceHandle {
    namespace: String,
    hydrated: bool,
    last_access: Instant,
    epoch: u64,
}

/// Handle table with LRU eviction under epoch guard.
pub struct LazyNamespaceTable {
    config: LazyNamespaceConfig,
    handles: RwLock<HashMap<String, NamespaceHandle>>,
    lru: RwLock<VecDeque<String>>,
    global_epoch: AtomicU64,
    evictions: AtomicU64,
}

impl LazyNamespaceTable {
    pub fn new(config: LazyNamespaceConfig) -> Self {
        Self {
            config,
            handles: RwLock::new(HashMap::new()),
            lru: RwLock::new(VecDeque::new()),
            global_epoch: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    pub fn resolve(&self, namespace: &str) -> u64 {
        let epoch = self.global_epoch.fetch_add(1, Ordering::SeqCst);
        let mut handles = self.handles.write();
        let entry = handles
            .entry(namespace.to_string())
            .or_insert_with(|| NamespaceHandle {
                namespace: namespace.to_string(),
                hydrated: false,
                last_access: Instant::now(),
                epoch,
            });
        entry.last_access = Instant::now();
        entry.epoch = epoch;
        if !entry.hydrated {
            entry.hydrated = true;
        }
        self.touch_lru(namespace);
        epoch
    }

    fn touch_lru(&self, namespace: &str) {
        let mut lru = self.lru.write();
        lru.retain(|n| n != namespace);
        lru.push_front(namespace.to_string());
        while lru.len() > self.config.max_resident {
            if let Some(victim) = lru.pop_back() {
                self.evict(&victim);
            }
        }
    }

    fn evict(&self, namespace: &str) {
        let mut handles = self.handles.write();
        if let Some(h) = handles.get_mut(namespace) {
            h.hydrated = false;
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn evict_idle(&self) {
        let now = Instant::now();
        let victims: Vec<String> = self
            .handles
            .read()
            .iter()
            .filter(|(_, h)| {
                h.hydrated && now.duration_since(h.last_access) > self.config.idle_evict_after
            })
            .map(|(k, _)| k.clone())
            .collect();
        for v in victims {
            self.evict(&v);
        }
    }

    pub fn resident_count(&self) -> usize {
        self.handles
            .read()
            .values()
            .filter(|h| h.hydrated)
            .count()
    }

    pub fn eviction_count(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }
}