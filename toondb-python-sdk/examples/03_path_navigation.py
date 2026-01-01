#!/usr/bin/env python3
"""
Example 03: Path Navigation
===========================

This example demonstrates ToonDB's path-native API for hierarchical data:
- Storing data at paths (like a filesystem)
- Retrieving data by path
- Organizing data hierarchically
- Real-world hierarchical data patterns

Difficulty: Beginner-Intermediate
Mode: Embedded (FFI)
"""

import os
import shutil
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from toondb import Database
from toondb.errors import DatabaseError

DB_PATH = "./example_03_db"


def cleanup():
    """Clean up database."""
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    print(f"✓ Cleaned up {DB_PATH}")


def example_basic_paths():
    """Basic path operations."""
    print("\n" + "=" * 60)
    print("Example 3.1: Basic Path Operations")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Store data at paths (like a filesystem)
        db.put_path("config/app/name", b"MyApplication")
        db.put_path("config/app/version", b"1.0.0")
        db.put_path("config/app/debug", b"false")
        print("✓ Stored application config")
        
        # Retrieve by path
        name = db.get_path("config/app/name")
        version = db.get_path("config/app/version")
        debug = db.get_path("config/app/debug")
        
        print(f"  App Name: {name.decode('utf-8')}")
        print(f"  Version: {version.decode('utf-8')}")
        print(f"  Debug: {debug.decode('utf-8')}")
        
        # Non-existent path returns None
        missing = db.get_path("config/app/nonexistent")
        print(f"  Nonexistent: {missing}")
    
    print("✓ Basic path operations complete")


def example_hierarchical_user_data():
    """Organizing user data hierarchically."""
    print("\n" + "=" * 60)
    print("Example 3.2: Hierarchical User Data")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Store user profile hierarchically
        user_id = "user_123"
        
        # Profile information
        db.put_path(f"users/{user_id}/profile/name", b"Alice Johnson")
        db.put_path(f"users/{user_id}/profile/email", b"alice@example.com")
        db.put_path(f"users/{user_id}/profile/avatar", b"/avatars/alice.png")
        
        # Preferences
        db.put_path(f"users/{user_id}/preferences/theme", b"dark")
        db.put_path(f"users/{user_id}/preferences/notifications", b"true")
        db.put_path(f"users/{user_id}/preferences/language", b"en-US")
        
        # Account metadata
        db.put_path(f"users/{user_id}/account/created_at", b"2024-01-15T10:30:00Z")
        db.put_path(f"users/{user_id}/account/plan", b"premium")
        
        print(f"✓ Created user {user_id} with hierarchical data")
        
        # Read back
        print(f"\nUser Profile:")
        print(f"  Name: {db.get_path(f'users/{user_id}/profile/name').decode()}")
        print(f"  Email: {db.get_path(f'users/{user_id}/profile/email').decode()}")
        
        print(f"\nPreferences:")
        print(f"  Theme: {db.get_path(f'users/{user_id}/preferences/theme').decode()}")
        print(f"  Lang: {db.get_path(f'users/{user_id}/preferences/language').decode()}")
        
        print(f"\nAccount:")
        print(f"  Plan: {db.get_path(f'users/{user_id}/account/plan').decode()}")
    
    print("\n✓ Hierarchical user data complete")


def example_api_configuration():
    """Managing API configuration with paths."""
    print("\n" + "=" * 60)
    print("Example 3.3: API Configuration Management")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Store API configurations for multiple services
        services = {
            "auth": {
                "endpoint": "https://auth.myapp.com/v1",
                "timeout_ms": "5000",
                "retries": "3"
            },
            "storage": {
                "endpoint": "https://storage.myapp.com/v2",
                "timeout_ms": "30000",
                "retries": "2"
            },
            "analytics": {
                "endpoint": "https://analytics.myapp.com/v1",
                "timeout_ms": "10000",
                "retries": "1"
            }
        }
        
        # Store all configurations
        for service_name, config in services.items():
            for key, value in config.items():
                path = f"api/{service_name}/{key}"
                db.put_path(path, value.encode('utf-8'))
        
        print("✓ Stored API configurations for 3 services")
        
        # Helper function to get service config
        def get_service_config(db, service_name):
            config = {}
            for key in ["endpoint", "timeout_ms", "retries"]:
                value = db.get_path(f"api/{service_name}/{key}")
                if value:
                    config[key] = value.decode('utf-8')
            return config
        
        # Read configurations
        print("\nService Configurations:")
        for service in ["auth", "storage", "analytics"]:
            config = get_service_config(db, service)
            print(f"\n  {service}:")
            for k, v in config.items():
                print(f"    {k}: {v}")
    
    print("\n✓ API configuration example complete")


def example_feature_flags():
    """Feature flags management with paths."""
    print("\n" + "=" * 60)
    print("Example 3.4: Feature Flags")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Define feature flags for different environments
        environments = ["development", "staging", "production"]
        
        features = {
            "new_dashboard": {"development": "true", "staging": "true", "production": "false"},
            "beta_api": {"development": "true", "staging": "true", "production": "false"},
            "dark_mode": {"development": "true", "staging": "true", "production": "true"},
            "analytics_v2": {"development": "true", "staging": "false", "production": "false"},
        }
        
        # Store feature flags
        for feature, env_values in features.items():
            for env, value in env_values.items():
                path = f"features/{env}/{feature}"
                db.put_path(path, value.encode('utf-8'))
        
        print("✓ Stored feature flags")
        
        # Helper to check if feature is enabled
        def is_feature_enabled(db, feature_name, environment):
            value = db.get_path(f"features/{environment}/{feature_name}")
            return value and value.decode('utf-8').lower() == "true"
        
        # Check features for each environment
        print("\nFeature Status:")
        print(f"{'Feature':<20} {'Dev':<8} {'Staging':<8} {'Prod':<8}")
        print("-" * 44)
        
        for feature in features.keys():
            dev = "✓" if is_feature_enabled(db, feature, "development") else "✗"
            stg = "✓" if is_feature_enabled(db, feature, "staging") else "✗"
            prd = "✓" if is_feature_enabled(db, feature, "production") else "✗"
            print(f"{feature:<20} {dev:<8} {stg:<8} {prd:<8}")
    
    print("\n✓ Feature flags example complete")


def example_document_store():
    """Storing JSON documents at paths."""
    print("\n" + "=" * 60)
    print("Example 3.5: Document Store with Paths")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Store product catalog
        products = [
            {"id": "prod_001", "name": "Laptop Pro", "price": 1299.99, "category": "electronics"},
            {"id": "prod_002", "name": "Wireless Mouse", "price": 49.99, "category": "electronics"},
            {"id": "prod_003", "name": "Standing Desk", "price": 599.00, "category": "furniture"},
        ]
        
        for product in products:
            prod_id = product["id"]
            # Store the full document
            db.put_path(f"products/{prod_id}/data", json.dumps(product).encode('utf-8'))
            # Also store an index by category
            db.put_path(f"products/by_category/{product['category']}/{prod_id}", b"1")
        
        print(f"✓ Stored {len(products)} products")
        
        # Retrieve a product
        laptop_data = db.get_path("products/prod_001/data")
        laptop = json.loads(laptop_data.decode('utf-8'))
        print(f"\nProduct Details:")
        print(f"  Name: {laptop['name']}")
        print(f"  Price: ${laptop['price']}")
        print(f"  Category: {laptop['category']}")
    
    print("\n✓ Document store example complete")


def example_path_with_transactions():
    """Combining paths with transactions."""
    print("\n" + "=" * 60)
    print("Example 3.6: Paths with Transactions")
    print("=" * 60)
    
    with Database.open(DB_PATH) as db:
        # Atomic update of related hierarchical data
        order_id = "order_12345"
        
        with db.transaction() as txn:
            # Order details
            txn.put_path(f"orders/{order_id}/status", b"processing")
            txn.put_path(f"orders/{order_id}/total", b"149.99")
            txn.put_path(f"orders/{order_id}/customer_id", b"cust_789")
            
            # Order items
            txn.put_path(f"orders/{order_id}/items/0/product_id", b"prod_002")
            txn.put_path(f"orders/{order_id}/items/0/quantity", b"2")
            txn.put_path(f"orders/{order_id}/items/0/price", b"49.99")
            
            txn.put_path(f"orders/{order_id}/items/1/product_id", b"prod_003")
            txn.put_path(f"orders/{order_id}/items/1/quantity", b"1")
            txn.put_path(f"orders/{order_id}/items/1/price", b"50.01")
            
            print(f"✓ Created order {order_id} atomically")
        
        # Read order
        print(f"\nOrder {order_id}:")
        print(f"  Status: {db.get_path(f'orders/{order_id}/status').decode()}")
        print(f"  Total: ${db.get_path(f'orders/{order_id}/total').decode()}")
        print(f"  Customer: {db.get_path(f'orders/{order_id}/customer_id').decode()}")
    
    print("\n✓ Path transactions example complete")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ToonDB Python SDK - Example 03: Path Navigation")
    print("=" * 60)
    
    cleanup()
    
    example_basic_paths()
    example_hierarchical_user_data()
    example_api_configuration()
    example_feature_flags()
    example_document_store()
    example_path_with_transactions()
    
    cleanup()
    
    print("\n" + "=" * 60)
    print("All path navigation examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
