#!/usr/bin/env python3
"""
Test script to verify SochDB analytics integration.

This sends several test events to PostHog to verify the integration works.
Events should appear in PostHog dashboard at: https://us.posthog.com
"""

import os
import time

# Ensure analytics is enabled
os.environ['SOCHDB_DISABLE_ANALYTICS'] = 'false'

from sochdb.analytics import (
    capture,
    capture_error,
    track_database_open,
    track_vector_search,
    track_batch_insert,
    is_analytics_disabled,
    _get_anonymous_id
)

def main():
    print("=" * 60)
    print("SochDB Analytics Test")
    print("=" * 60)
    print()
    
    # Check status
    print(f"Analytics disabled: {is_analytics_disabled()}")
    print(f"Anonymous ID: {_get_anonymous_id()}")
    print()
    
    print("Sending test events...")
    print()
    
    # Test 1: Basic event
    print("1. Basic event...")
    capture('analytics_test', {
        'test_number': 1,
        'test_name': 'basic_event',
        'timestamp': time.time()
    })
    print("   ✅ Sent")
    
    # Test 2: Database open
    print("2. Database open event...")
    track_database_open('./test_db', 'embedded')
    print("   ✅ Sent")
    
    # Test 3: Vector search
    print("3. Vector search event...")
    track_vector_search(dimension=1536, k=10, latency_ms=45.7)
    print("   ✅ Sent")
    
    # Test 4: Batch insert
    print("4. Batch insert event...")
    track_batch_insert(count=1000, dimension=768, latency_ms=123.4)
    print("   ✅ Sent")
    
    # Test 5: Error event
    print("5. Error event...")
    capture_error('test_error', 'This is a test error for analytics verification')
    print("   ✅ Sent")
    
    print()
    print("=" * 60)
    print("All test events sent successfully!")
    print()
    print("Check PostHog dashboard:")
    print("https://us.posthog.com/project/YOUR_PROJECT_ID/events")
    print()
    print("Expected events:")
    print("  - analytics_test")
    print("  - database_opened")
    print("  - vector_search")
    print("  - batch_insert")
    print("  - error")
    print("=" * 60)

if __name__ == '__main__':
    main()
