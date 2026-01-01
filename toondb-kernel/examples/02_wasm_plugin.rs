//! Example 2: WASM Plugin Runtime Demo
//!
//! This example demonstrates the WASM sandboxed plugin runtime.
//!
//! Run with: cargo run --example 02_wasm_plugin --package toondb-kernel

use toondb_kernel::wasm_runtime::{
    WasmInstanceConfig, WasmPluginCapabilities, WasmPluginInstance, WasmPluginRegistry,
    WasmPluginState, WasmValue,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ToonDB WASM Plugin Runtime - Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Step 1: Demonstrate Capabilities Configuration
    println!("Step 1: Plugin Capabilities\n");
    
    let default_caps = WasmPluginCapabilities::default();
    println!("   Default capabilities:");
    println!("   - Memory limit: {} MB", default_caps.memory_limit_bytes / (1024 * 1024));
    println!("   - Fuel limit: {} instructions", default_caps.fuel_limit);
    println!("   - Timeout: {}ms", default_caps.timeout_ms);
    println!("   - Can vector search: {}", default_caps.can_vector_search);
    println!();

    let readonly_caps = WasmPluginCapabilities::read_only(vec!["users".to_string(), "logs_*".to_string()]);
    println!("   Read-only analytics capabilities:");
    println!("   - Can read 'users' table: {}", readonly_caps.can_read("users"));
    println!("   - Can read 'logs_2024' table: {}", readonly_caps.can_read("logs_2024"));
    println!("   - Can read 'orders' table: {}", readonly_caps.can_read("orders"));
    println!("   - Can write 'users' table: {}", readonly_caps.can_write("users"));
    println!("   - Can vector search: {}", readonly_caps.can_vector_search);
    println!();

    // Step 2: Create a single WASM plugin instance
    println!("Step 2: Creating WASM Plugin Instance\n");

    let config = WasmInstanceConfig {
        capabilities: WasmPluginCapabilities::observability_only(),
        debug_mode: true,
        enable_fuel: true,
        ..Default::default()
    };

    // Note: In production, this would be actual WASM bytes
    let fake_wasm_bytes = b"fake wasm module bytes for demo";
    let instance = WasmPluginInstance::new("my-analytics-plugin", fake_wasm_bytes, config).unwrap();
    
    println!("   Plugin name: {}", instance.name());
    println!("   Plugin state: {:?}", instance.state());
    
    // Initialize the plugin
    instance.init().unwrap();
    println!("   After init - state: {:?}", instance.state());
    println!();

    // Step 3: Call plugin functions
    println!("Step 3: Calling Plugin Functions\n");

    // Call various hook functions
    let result = instance.call("on_insert", &[WasmValue::I32(42)]).unwrap();
    println!("   on_insert(42) returned: {:?}", result);

    let result = instance.call("on_update", &[WasmValue::I64(100)]).unwrap();
    println!("   on_update(100) returned: {:?}", result);

    let result = instance.call("get_metrics", &[]).unwrap();
    println!("   get_metrics() returned: {:?}", result);

    let result = instance.call("transform", &[WasmValue::F64(3.14)]).unwrap();
    println!("   transform(3.14) returned: {:?}", result);
    println!();

    // Step 4: Check plugin statistics
    println!("Step 4: Plugin Statistics\n");
    
    let stats = instance.stats();
    println!("   Total calls: {}", stats.total_calls);
    println!("   Total fuel consumed: {}", stats.total_fuel_consumed);
    println!("   Total execution time: {}μs", stats.total_execution_us);
    println!("   Trap count: {}", stats.trap_count);
    println!();

    // Step 5: Plugin Registry Demo
    println!("Step 5: WASM Plugin Registry\n");

    let registry = WasmPluginRegistry::new();

    // Load multiple plugins
    registry.load("plugin-a", b"wasm bytes A", WasmInstanceConfig::default()).unwrap();
    registry.load("plugin-b", b"wasm bytes B", WasmInstanceConfig::default()).unwrap();
    registry.load("plugin-c", b"wasm bytes C", WasmInstanceConfig::default()).unwrap();

    println!("   Registered plugins: {:?}", registry.list());
    println!("   Plugin count: {}", registry.count());

    // Call plugins through registry
    let result = registry.call("plugin-a", "on_insert", &[]).unwrap();
    println!("   Called plugin-a.on_insert(): {:?}", result);

    let result = registry.call("plugin-b", "get_metrics", &[]).unwrap();
    println!("   Called plugin-b.get_metrics(): {:?}", result);

    // Get specific plugin
    if let Some(plugin) = registry.get("plugin-c") {
        println!("   Retrieved plugin-c, state: {:?}", plugin.state());
    }

    // Global stats
    let (total_calls, total_traps) = registry.global_stats();
    println!("   Global stats - calls: {}, traps: {}", total_calls, total_traps);

    // Unload a plugin
    registry.unload("plugin-b").unwrap();
    println!("   After unloading plugin-b: {:?}", registry.list());

    // Shutdown all
    registry.shutdown_all().unwrap();
    println!("   After shutdown: {:?}", registry.list());

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ✅ WASM Plugin Runtime Demo completed!");
    println!("═══════════════════════════════════════════════════════════════");
}
