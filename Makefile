.PHONY: build

build:
	maturin build --manifest-path crates/pyhebb/Cargo.toml
