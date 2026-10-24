# Jean-Pierre

Unsupervised visual feature extractor, bio-inspired and powered by STDP. For now this is very basic, new features and
modalities, such as text, audio, and various motor controls, will be added later. Feedback welcome, but keep your
contributions focused and clean (no low effort slop PR's).

(Btw, if you have a better name for this project, please by all means open an issue)

WARNING: This code is experimental and should not be used in production.

## Usage

### Prerequisites

- [Rust](https://rustup.rs/)
- [Python](https://www.python.org/) and [uv](https://docs.astral.sh/uv/getting-started/installation/)

Clone the project:
```
git clone https://github.com/lazo4/jean-pierre.git
```

### Building

First build the python bindings:
```
make dev
```

Then run one of the examples with:
```
uv run examples/[EXAMPLE].py
```

