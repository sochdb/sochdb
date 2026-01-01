//! Example 3: Plugin Hot-Reload Demo
//!
//! This example demonstrates zero-downtime plugin upgrades using atomic swapping
//! and epoch-based draining.
//!
//! Run with: cargo run --example 03_hot_reload --package toondb-kernel

use std::sync::Arc;
use std::thread;
use std::time::Duration;
use toondb_kernel::plugin_hot_reload::{
    EpochTracker, HotReloadManager, HotReloadState, HotReloadablePlugin,
};
use toondb_kernel::plugin_manifest::ManifestBuilder;
use toondb_kernel::wasm_runtime::{WasmInstanceConfig, WasmPluginInstance};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ToonDB Hot-Reload Plugin System - Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Step 1: Demonstrate Epoch Tracking (core mechanism for safe upgrades)
    println!("Step 1: Epoch-Based Draining Mechanism\n");

    let tracker = EpochTracker::new();
    println!("   Initial epoch: {}", tracker.current());

    // Simulate entering an epoch (like starting a request)
    let guard = tracker.enter();
    println!("   After entering - epoch refs: {}", tracker.refs_for_epoch(0));

    // Advance epoch
    let new_epoch = tracker.advance();
    println!("   After advancing - current epoch: {}", new_epoch);
    println!("   Old epoch (0) still has refs: {}", tracker.refs_for_epoch(0));

    // Drop guard to release epoch
    drop(guard);
    println!("   After dropping guard - epoch 0 refs: {}", tracker.refs_for_epoch(0));

    // Now draining would succeed
    let drained = tracker.wait_drain(0, Duration::from_millis(10));
    println!("   Drain epoch 0 succeeded: {}", drained);
    println!();

    // Step 2: Create a Hot-Reloadable Plugin
    println!("Step 2: Creating Hot-Reloadable Plugin\n");

    let config = WasmInstanceConfig::default();
    let instance = WasmPluginInstance::new("my-plugin", b"wasm v1.0", config).unwrap();
    instance.init().unwrap();

    let manifest = ManifestBuilder::new("my-plugin", "1.0.0")
        .author("ToonDB Team")
        .description("Demo plugin")
        .export("on_insert")
        .export("on_update")
        .build()
        .unwrap();

    let plugin = HotReloadablePlugin::new("my-plugin", Arc::new(instance), manifest);

    println!("   Plugin name: {}", plugin.name());
    println!("   Plugin state: {:?}", plugin.state());
    println!("   Initial version: {}", plugin.stats().version);
    println!();

    // Step 3: Make calls through the hot-reloadable wrapper
    println!("Step 3: Calling Plugin (Epoch-Protected)\n");

    let result = plugin.call("on_insert", &[]).unwrap();
    println!("   on_insert() result: {:?}", result);

    let result = plugin.call("on_update", &[]).unwrap();
    println!("   on_update() result: {:?}", result);

    let stats = plugin.stats();
    println!("   Stats - version: {}, successful upgrades: {}", stats.version, stats.successful_upgrades);
    println!();

    // Step 4: Perform a Hot-Reload Upgrade
    println!("Step 4: Hot-Reload Upgrade (v1.0 → v2.0)\n");

    let new_manifest = ManifestBuilder::new("my-plugin", "2.0.0")
        .author("ToonDB Team")
        .description("Demo plugin v2 with improvements")
        .export("on_insert")
        .export("on_update")
        .export("on_delete")  // New function in v2
        .build()
        .unwrap();

    println!("   State before upgrade: {:?}", plugin.state());

    // Prepare the upgrade
    plugin.prepare_upgrade(b"wasm v2.0 bytes", new_manifest).unwrap();
    println!("   State after prepare: {:?}", plugin.state());

    // Execute the upgrade (drains in-flight, atomically swaps)
    plugin.execute_upgrade().unwrap();
    println!("   State after execute: {:?}", plugin.state());

    let stats = plugin.stats();
    println!("   Stats after upgrade:");
    println!("   - Version: {}", stats.version);
    println!("   - Successful upgrades: {}", stats.successful_upgrades);
    println!("   - Total drain time: {}μs", stats.total_drain_time_us);
    println!();

    // Step 5: Hot-Reload Manager
    println!("Step 5: Hot-Reload Manager (Multiple Plugins)\n");

    let manager = HotReloadManager::new();

    // Register multiple plugins
    for i in 1..=3 {
        let name = format!("service-{}", i);
        let config = WasmInstanceConfig::default();
        let instance = WasmPluginInstance::new(&name, b"wasm bytes", config).unwrap();
        instance.init().unwrap();

        let manifest = ManifestBuilder::new(&name, "1.0.0")
            .export("on_insert")
            .build()
            .unwrap();

        manager.register(&name, Arc::new(instance), manifest).unwrap();
    }

    println!("   Registered plugins: {:?}", manager.list());

    // Upgrade one plugin
    let new_manifest = ManifestBuilder::new("service-2", "2.0.0")
        .export("on_insert")
        .build()
        .unwrap();
    
    manager.upgrade("service-2", b"new wasm", new_manifest).unwrap();
    println!("   Upgraded service-2 to v2.0.0");

    // Check all stats
    for (name, stats) in manager.all_stats() {
        println!("   {} - version: {}, upgrades: {}", name, stats.version, stats.successful_upgrades);
    }
    println!();

    // Step 6: Concurrent calls during upgrade (demonstration)
    println!("Step 6: Concurrent Access During Upgrade\n");

    let config = WasmInstanceConfig::default();
    let instance = WasmPluginInstance::new("concurrent-plugin", b"v1", config).unwrap();
    instance.init().unwrap();
    let manifest = ManifestBuilder::new("concurrent-plugin", "1.0.0")
        .export("on_insert")
        .build()
        .unwrap();

    let plugin = Arc::new(HotReloadablePlugin::new(
        "concurrent-plugin",
        Arc::new(instance),
        manifest,
    ));

    // Spawn threads making calls
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let p = plugin.clone();
            thread::spawn(move || {
                for j in 0..5 {
                    let _ = p.call("on_insert", &[]);
                    thread::sleep(Duration::from_millis(1));
                }
                println!("   Thread {} completed 5 calls", i);
            })
        })
        .collect();

    // Wait a bit then upgrade
    thread::sleep(Duration::from_millis(5));
    println!("   Initiating upgrade while threads are running...");

    let new_manifest = ManifestBuilder::new("concurrent-plugin", "2.0.0")
        .export("on_insert")
        .build()
        .unwrap();

    match plugin.upgrade(b"v2 bytes", new_manifest) {
        Ok(()) => println!("   ✅ Upgrade succeeded during concurrent access!"),
        Err(e) => println!("   ⚠️  Upgrade failed (expected if threads held refs): {}", e),
    }

    // Wait for threads
    for h in handles {
        h.join().unwrap();
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ✅ Hot-Reload Demo completed!");
    println!("═══════════════════════════════════════════════════════════════");
}
