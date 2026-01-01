# ToonDB Makefile
# One-command development setup and common tasks

.PHONY: setup build test check clean docs bench help

# Default target
help:
	@echo "ToonDB Development Commands"
	@echo "==========================="
	@echo ""
	@echo "  make setup    - One-command development environment setup"
	@echo "  make build    - Build all crates"
	@echo "  make test     - Run all tests"
	@echo "  make check    - Run full CI checks (format, lint, test)"
	@echo "  make docs     - Build documentation"
	@echo "  make bench    - Run benchmarks"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "Quick Start:"
	@echo "  git clone https://github.com/toondb/toondb && cd toondb && make setup"

# One-command setup for new contributors
setup: check-rust install-tools build test
	@echo ""
	@echo "✅ ToonDB development environment is ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Read CONTRIBUTING.md"
	@echo "  2. Pick an issue from GitHub"
	@echo "  3. Create a branch: git checkout -b feature/my-feature"
	@echo ""

# Check Rust installation
check-rust:
	@echo "Checking Rust installation..."
	@command -v rustc >/dev/null 2>&1 || { \
		echo "Rust not found. Installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		. $$HOME/.cargo/env; \
	}
	@rustc --version
	@cargo --version

# Install development tools
install-tools:
	@echo "Installing development tools..."
	@cargo install cargo-watch 2>/dev/null || true
	@cargo install cargo-criterion 2>/dev/null || true
	@cargo install cargo-deny 2>/dev/null || true
	@rustup component add clippy rustfmt 2>/dev/null || true
	@echo "✅ Development tools installed"

# Build all crates
build:
	@echo "Building all crates..."
	cargo build --all

# Build release version
release:
	@echo "Building release..."
	cargo build --release --all

# Run all tests
test:
	@echo "Running tests..."
	cargo test --all

# Run tests with output
test-verbose:
	cargo test --all -- --nocapture

# Full CI check (run before submitting PR)
check: fmt-check lint test
	@echo ""
	@echo "✅ All checks passed!"

# Format check (no changes)
fmt-check:
	@echo "Checking formatting..."
	cargo fmt --all -- --check

# Format code
fmt:
	@echo "Formatting code..."
	cargo fmt --all

# Run clippy lints
lint:
	@echo "Running clippy..."
	cargo clippy --all -- -D warnings

# Build documentation
docs:
	@echo "Building documentation..."
	cargo doc --no-deps --all
	@echo "Documentation available at: target/doc/toondb/index.html"

# Open documentation in browser
docs-open: docs
	@open target/doc/toondb/index.html 2>/dev/null || \
	 xdg-open target/doc/toondb/index.html 2>/dev/null || \
	 echo "Open target/doc/toondb/index.html in your browser"

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	cargo bench

# Run specific benchmark
bench-%:
	cargo bench -p toondb-$*

# Clean build artifacts
clean:
	cargo clean
	rm -rf coverage/

# Coverage report
coverage:
	@echo "Generating coverage report..."
	@command -v cargo-tarpaulin >/dev/null 2>&1 || cargo install cargo-tarpaulin
	cargo tarpaulin --all --out Html --output-dir coverage/
	@echo "Coverage report: coverage/tarpaulin-report.html"

# Watch mode for development
watch:
	cargo watch -x 'test --all'

watch-build:
	cargo watch -x 'build --all'

# Python SDK
python-build:
	cd toondb-python && maturin build --release

python-develop:
	cd toondb-python && maturin develop

# MCP server
mcp-build:
	cargo build --release -p toondb-mcp

mcp-run:
	cargo run --release -p toondb-mcp -- --db ./test_mcp_db

# Server
server-run:
	cargo run --release -p toondb-grpc -- --config toondb-server-config.toml

# Security audit
audit:
	@command -v cargo-deny >/dev/null 2>&1 || cargo install cargo-deny
	cargo deny check

# Find unused dependencies
udeps:
	@command -v cargo-udeps >/dev/null 2>&1 || cargo install cargo-udeps
	cargo +nightly udeps --all-targets
