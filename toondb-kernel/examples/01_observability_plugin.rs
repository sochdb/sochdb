//! Example 1: Observability Plugin Demo
//!
//! This example demonstrates how to use the plugin architecture for observability.
//!
//! Run with: cargo run --example 01_observability_plugin --package toondb-kernel

use std::any::Any;
use std::sync::Arc;
use toondb_kernel::{
    Extension, ExtensionCapability, ExtensionInfo, ObservabilityExtension, PluginManager,
};

/// A simple custom metrics collector that prints to stdout
struct SimpleMetricsPlugin {
    name: String,
}

impl SimpleMetricsPlugin {
    fn new(name: &str) -> Self {
        println!("✅ [{}] Plugin created", name);
        Self {
            name: name.to_string(),
        }
    }
}

impl Extension for SimpleMetricsPlugin {
    fn info(&self) -> ExtensionInfo {
        ExtensionInfo {
            name: self.name.clone(),
            version: "1.0.0".to_string(),
            description: "Simple metrics collector for demo".to_string(),
            author: "ToonDB Demo".to_string(),
            capabilities: vec![ExtensionCapability::Observability],
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ObservabilityExtension for SimpleMetricsPlugin {
    fn counter_inc(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        let label_str: Vec<String> = labels.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        println!(
            "📊 [COUNTER] {} += {} {}",
            name,
            value,
            if label_str.is_empty() {
                String::new()
            } else {
                format!("({})", label_str.join(", "))
            }
        );
    }

    fn gauge_set(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let label_str: Vec<String> = labels.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        println!(
            "📈 [GAUGE] {} = {:.2} {}",
            name,
            value,
            if label_str.is_empty() {
                String::new()
            } else {
                format!("({})", label_str.join(", "))
            }
        );
    }

    fn histogram_observe(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        let label_str: Vec<String> = labels.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        println!(
            "📉 [HISTOGRAM] {} observed {:.3}ms {}",
            name,
            value * 1000.0,
            if label_str.is_empty() {
                String::new()
            } else {
                format!("({})", label_str.join(", "))
            }
        );
    }

    fn span_start(&self, name: &str, parent: Option<u64>) -> u64 {
        let span_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            % 100000;
        println!(
            "🔷 [SPAN START] {} (id={}, parent={:?})",
            name, span_id, parent
        );
        span_id
    }

    fn span_end(&self, span_id: u64) {
        println!("🔶 [SPAN END] id={}", span_id);
    }

    fn span_event(&self, span_id: u64, name: &str, attributes: &[(&str, &str)]) {
        let attr_str: Vec<String> = attributes
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        println!(
            "   ├─ [EVENT] {} in span {} {}",
            name,
            span_id,
            if attr_str.is_empty() {
                String::new()
            } else {
                format!("({})", attr_str.join(", "))
            }
        );
    }

    fn log_info(&self, message: &str, fields: &[(&str, &str)]) {
        let field_str: Vec<String> = fields.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        println!(
            "ℹ️  [INFO] {} {}",
            message,
            if field_str.is_empty() {
                String::new()
            } else {
                format!("| {}", field_str.join(", "))
            }
        );
    }

    fn log_error(&self, message: &str, fields: &[(&str, &str)]) {
        let field_str: Vec<String> = fields.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        println!(
            "❌ [ERROR] {} {}",
            message,
            if field_str.is_empty() {
                String::new()
            } else {
                format!("| {}", field_str.join(", "))
            }
        );
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ToonDB Plugin System - Observability Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Step 1: Create the Plugin Manager
    println!("Step 1: Creating PluginManager...");
    let plugins = PluginManager::new();
    println!("   Has observability before registration: {}\n", plugins.has_observability());

    // Step 2: Register our custom plugin
    println!("Step 2: Registering custom observability plugin...");
    let metrics_plugin = Arc::new(SimpleMetricsPlugin::new("demo-metrics"));
    plugins.register_observability(metrics_plugin).unwrap();
    println!("   Has observability after registration: {}\n", plugins.has_observability());

    // Step 3: List all registered extensions
    println!("Step 3: Listing registered extensions...");
    for ext in plugins.list_extensions() {
        println!("   - {} v{} ({})", ext.name, ext.version, ext.description);
    }
    println!();

    // Step 4: Use the plugin for metrics
    println!("Step 4: Recording metrics through the plugin...\n");

    // Simulate database operations with metrics
    plugins.counter_inc("toondb_queries_total", 1, &[("type", "select")]);
    plugins.counter_inc("toondb_queries_total", 1, &[("type", "insert")]);
    plugins.counter_inc("toondb_queries_total", 3, &[("type", "select")]);

    plugins.gauge_set("toondb_connections", 5.0, &[("pool", "primary")]);
    plugins.gauge_set("toondb_buffer_pool_used", 0.73, &[]);

    plugins.histogram_observe("toondb_query_duration", 0.0023, &[("query", "select")]);
    plugins.histogram_observe("toondb_query_duration", 0.0156, &[("query", "insert")]);

    println!();

    // Step 5: Test tracing/spans (simulated through observability extensions)
    println!("Step 5: Testing spans (distributed tracing)...\n");

    // Note: PluginManager doesn't have direct span methods; these would be called on the extension
    // But we can demonstrate the fan-out behavior with metrics
    plugins.counter_inc("toondb_transactions_started", 1, &[]);
    plugins.counter_inc("toondb_rows_read", 42, &[("table", "users")]);
    plugins.counter_inc("toondb_transactions_committed", 1, &[]);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  ✅ Demo completed successfully!");
    println!("═══════════════════════════════════════════════════════════════");
}
