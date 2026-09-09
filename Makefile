.PHONY: build
.PHONY: dev
.PHONY: dev-release
.PHONY: release

build:
	maturin build --manifest-path crates/pyhebb/Cargo.toml

release:
	maturin build --release --manifest-path crates/pyhebb/Cargo.toml

dev:
	maturin develop --manifest-path crates/pyhebb/Cargo.toml

dev-release:
	maturin develop --release --manifest-path crates/pyhebb/Cargo.toml
