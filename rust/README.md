# DiffEqGMRFs Rust workspace

This workspace mirrors the Julia `DiffEqGMRFs.jl` project with crate-per-module parity and example binaries that reproduce the reference assembly flows.

## Module mapping
- **Julia finite-element utilities** → `diffeq-gmrfs-core` provides meshes, constraint handling, metrics, and block-tridiagonal Cholesky utilities built on `ndarray`, `nalgebra`, and `sprs`.
- **Julia dataset loaders** → `diffeq-gmrfs-datasets` loads Burgers and Darcy MATLAB `.mat` files (real-valued) or synthetic fixtures with the same accessor semantics as the Julia helpers.
- **Julia PDE assemblers** → `diffeq-gmrfs-problems` assembles Burgers advection/mass/diffusion systems and Darcy diffusion matrices with optional constraint pruning.
- **Julia SPDE utilities** → `diffeq-gmrfs-spde` constructs shallow-water SPDE discretizations, Matern priors, and constrained/dense GMRF solvers for small systems.
- **Example scripts** → `diffeq-gmrfs-examples` hosts runnable binaries and integration tests that stitch the crates together in the same order as the Julia scripts.

## Quickstart
1. Install Rust 1.78+ with `cargo` available.
2. From `rust/`, run the validation stack:
   - `cargo fmt`
   - `cargo clippy --all-targets --all-features`
   - `cargo test`
3. Run example binaries (use a dataset path to consume `.mat` inputs or omit it to fall back to synthetic fixtures):
   - `cargo run -p diffeq-gmrfs-examples --bin burgers [path/to/burgers.mat]`
   - `cargo run -p diffeq-gmrfs-examples --bin darcy [path/to/darcy.mat]`
   - `cargo run -p diffeq-gmrfs-examples --bin shallow_water`

## Migration notes
- MATLAB datasets must contain real-valued arrays; complex fields are rejected to match the Julia guards.
- Mesh helpers use row-major `ndarray` storage and accept tolerances for constraint detection; pass the same grid sizes as the Julia generators to keep parity.
- Sparse/dense linear algebra relies on `sprs` CSC matrices and `nalgebra` factorizations in place of Julia’s `SparseArrays` and `LinearAlgebra.Cholesky`.
- The examples mirror the Julia scripts but keep error handling explicit via `Result` returns; compose them directly in downstream binaries or tests rather than relying on global state.
- For larger systems where dense factorizations are insufficient, swap in iterative solvers atop `sprs` or `nalgebra-sparse` while preserving the exposed trait boundaries in `diffeq-gmrfs-core` and `diffeq-gmrfs-spde`.
