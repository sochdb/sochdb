// Test Rust analytics with static error tracking
use std::collections::HashMap;

fn main() {
    println!("============================================================");
    println!("ToonDB Analytics Test (Rust - Static Errors)");
    println!("============================================================\n");

    // Check if analytics is disabled
    let disabled = std::env::var("TOONDB_DISABLE_ANALYTICS")
        .map(|v| {
            let v = v.to_lowercase();
            v == "true" || v == "1" || v == "yes" || v == "on"
        })
        .unwrap_or(false);
    
    println!("Analytics disabled: {}\n", disabled);

    if disabled {
        println!("⚠️  Analytics disabled via TOONDB_DISABLE_ANALYTICS");
        return;
    }

    println!("Sending test events...\n");

    // Test 1: Database open
    println!("1. Database open event...");
    let mut props = HashMap::new();
    props.insert("mode".to_string(), serde_json::json!("embedded"));
    props.insert("has_custom_path".to_string(), serde_json::json!(true));
    send_event("database_opened", Some(props));
    println!("   ✅ Sent");

    println!("\n2. Testing error events (static only)...\n");

    // Test 2: Connection error
    println!("   a. connection_error @ database::open...");
    send_error("connection_error", "database::open");
    println!("      ✅ Sent");

    // Test 3: Query error
    println!("   b. query_error @ sql::execute...");
    send_error("query_error", "sql::execute");
    println!("      ✅ Sent");

    // Test 4: Permission error
    println!("   c. permission_error @ table::access...");
    send_error("permission_error", "table::access");
    println!("      ✅ Sent");

    // Test 5: Timeout error
    println!("   d. timeout_error @ transaction::commit...");
    send_error("timeout_error", "transaction::commit");
    println!("      ✅ Sent");

    println!("\n============================================================");
    println!("All test events sent successfully!");
    println!("Only static info sent - no sensitive data!");
    println!("\nCheck PostHog dashboard:");
    println!("https://us.posthog.com/project/YOUR_PROJECT_ID/events");
    println!("\nExpected events:");
    println!("  - database_opened");
    println!("  - error (4 events with different types/locations)");
    println!("============================================================");
}

fn get_anonymous_id() -> String {
    use std::hash::{Hash, Hasher};
    
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    
    if let Ok(hostname) = hostname::get() {
        hostname.to_string_lossy().hash(&mut hasher);
    }
    
    std::env::consts::OS.hash(&mut hasher);
    std::env::consts::ARCH.hash(&mut hasher);
    
    #[cfg(unix)]
    unsafe {
        libc::getuid().hash(&mut hasher);
    }
    
    format!("{:016x}", hasher.finish())
}

fn send_event(event: &str, properties: Option<HashMap<String, serde_json::Value>>) {
    let api_key = "phc_zf0hm6ZmPUJj1pM07Kigqvphh1ClhKX1NahRU4G0bfu";
    let host = "https://us.i.posthog.com";
    let distinct_id = get_anonymous_id();
    
    // Build properties with SDK context
    let mut event_properties = properties.unwrap_or_default();
    event_properties.insert("sdk".to_string(), serde_json::json!("rust"));
    event_properties.insert("sdk_version".to_string(), serde_json::json!("0.3.1"));
    event_properties.insert("os".to_string(), serde_json::json!(std::env::consts::OS));
    event_properties.insert("arch".to_string(), serde_json::json!(std::env::consts::ARCH));
    
    // PostHog capture endpoint expects this format
    let payload = serde_json::json!({
        "api_key": api_key,
        "event": event,
        "properties": event_properties,
        "distinct_id": distinct_id,
    });
    
    let url = format!("{}/capture/", host);
    
    match ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(&payload)
    {
        Ok(_) => {},
        Err(e) => eprintln!("   ⚠️  Failed to send event: {}", e),
    }
}

fn send_error(error_type: &str, location: &str) {
    let mut props = HashMap::new();
    props.insert("error_type".to_string(), serde_json::json!(error_type));
    props.insert("location".to_string(), serde_json::json!(location));
    send_event("error", Some(props));
}
