# Test error tracking across all SDKs

import sys
import os

# Ensure we can import toondb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'toondb-python-sdk', 'src'))

from toondb.analytics import capture_error, is_analytics_disabled

print("=" * 60)
print("Testing Error Tracking (Static Only)")
print("=" * 60)
print(f"Analytics disabled: {is_analytics_disabled()}\n")

if not is_analytics_disabled():
    print("Sending test error events (static info only)...\n")
    
    # Test different error types with static locations
    errors = [
        ("connection_error", "database.open"),
        ("query_error", "sql.execute"),
        ("permission_error", "table.access"),
        ("timeout_error", "transaction.commit"),
    ]
    
    for i, (error_type, location) in enumerate(errors, 1):
        print(f"{i}. {error_type} @ {location}...")
        capture_error(error_type, location)
        print(f"   ✅ Sent")
    
    print("\n" + "=" * 60)
    print("All error events sent successfully!")
    print("Only static info sent - no sensitive data!")
    print("=" * 60)
else:
    print("⚠️  Analytics disabled - no events sent")
