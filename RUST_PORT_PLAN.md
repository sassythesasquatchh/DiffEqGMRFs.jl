# DiffEqGMRFs.jl → Rust Port Plan

## Repository overview
- Finite-element utilities built on Ferrite.jl to create discretizations (Dirichlet/periodic constraints, 1D/2D grids).
- PDE-specific assemblers:
  - Darcy diffusion matrix assembly with coefficient lookups from data-driven fields.
  - Burgers advection and mass/diffusion matrices with optional lumping and constraint handling.
  - Linear shallow-water SPDE discretization yielding Gaussian Markov Random Fields via implicit Euler state-space models.
- Data loaders for MATLAB `.mat` datasets (Darcy and Burgers) exposing convenience accessors.
- Metrics helpers (RMSE, max error, relative error) and a block-tridiagonal Cholesky factorization helper for banded systems.

## Rust ecosystem equivalents
- **Linear algebra & sparse matrices:** `nalgebra` + `nalgebra_sparse` or `sprs` for CSC matrices; consider `linfa-linalg` for LAPACK bindings when needed.
- **FEM/grid tools:** `fenris` (FEM), `russell_lab`/`russell_sparse` for assembly utilities, or `meshx` for mesh generation; if insufficient, implement minimal mesh + element primitives.
- **Probabilistic models / GMRFs:** use `linfa` for general ML pieces plus custom GMRF structs; `ndarray` for dense arrays; implement precision-operator-based sampling/solves atop `sprs` + iterative solvers (`argmin` or `krylov`).
- **MAT file IO:** `matfile` or `hdf5` crate depending on dataset format version.
- **Special functions:** `special` or `statrs` for gamma functions.

## Incremental port roadmap
1. **Set up Rust workspace**
   - Create Cargo workspace with crates for `core` (math/FEM), `datasets`, `problems`, and `spde` to mirror Julia structure.
   - Add dependencies above plus `thiserror`/`anyhow` for error handling and `serde` for configuration where helpful.

2. **Data layer parity** ✅
   - Implemented `diffeq-gmrfs-datasets` with loaders for Darcy and Burgers `.mat` files (real-valued only), coordinate generation, and accessor helpers matching the Julia interfaces.
   - Added unit tests over synthetic fixtures to verify shapes, accessors, and nearest-index lookup semantics.

3. **Core utilities & metrics** ✅
   - Port metrics (RMSE, max error, relative error) using `ndarray` views.
   - Recreate periodic/Dirichlet constraint helpers and basic discretization builders for 1D/2D domains; leverage `fenris`/`meshx` or custom mesh structs.
   - Completed in `rust/core` with `ndarray` + `num-traits`, providing 1D periodic and 2D Dirichlet-ready discretizations and matching unit tests.

4. **Problem assemblers** ✅
   - Added `diffeq-gmrfs-problems` with Darcy diffusion assembly (supporting inflated boundaries and Dirichlet constraints) and Burgers advection/mass/diffusion assemblers with lumping and periodic handling.
   - Included unit tests on small synthetic meshes to validate stencil construction, constraint handling, and forcing terms.

5. **Numerical linear algebra utilities** ✅
   - Implemented block-tridiagonal Cholesky factor with dense block storage and forward/backward solves using `sprs` extraction and `nalgebra` factorizations.

6. **SPDE / shallow-water model** ✅
   - Added `diffeq-gmrfs-spde` with linear shallow-water assembly, Matern priors, implicit-Euler joint precision construction, and boundary-aware noise handling.
   - Introduced GMRF/constrained-GMRF types plus a Cholesky precision solver abstraction for downstream inference utilities.

7. **Examples and parity tests** ✅
   - Added `diffeq-gmrfs-examples` crate with runnable Burgers, Darcy, and shallow-water binaries that mirror the Julia script flow using either on-disk datasets or synthetic fixtures.
   - Exercised pipelines in integration tests and wired CI to run Rust fmt/clippy/test plus a bench build to keep kernels regression-ready.

8. **Documentation** ✅
   - Documented module mapping from Julia → Rust crates, usage examples, and migration notes with guidance on substituting Julia dependencies in `rust/README.md`.

## Future follow-ups
- Add performance benchmarks that mirror the Julia profiling scripts to evaluate solver scaling.
- Extend linear-solvers to expose iterative options for large sparse systems once inference workloads are ported.
- Package dataset download helpers so CI and downstream users can fetch `.mat` fixtures automatically.
