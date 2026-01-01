#!/bin/bash
# ToonDB MCP Server wrapper for Goose
# This script ensures proper execution with correct working directory

# Set database path (can be overridden by --db argument)
DB_PATH="${TOONDB_DATA:-/Users/sushanth/toon_database/toondb_data}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --db)
            DB_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Create database directory if it doesn't exist
mkdir -p "$DB_PATH"

# Run the MCP server
exec /Users/sushanth/toon_database/target/release/toondb-mcp --db "$DB_PATH"
