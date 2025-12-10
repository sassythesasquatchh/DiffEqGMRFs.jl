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

2. **Data layer parity**
   - Implement `datasets` crate: loaders for Darcy and Burgers `.mat` files, including coordinate generation and accessor methods.
   - Validate with sample `.mat` fixtures and unit tests to match Julia loader semantics.

3. **Core utilities & metrics**
   - Port metrics (RMSE, max error, relative error) using `ndarray` views.
   - Recreate periodic/Dirichlet constraint helpers and basic discretization builders for 1D/2D domains; leverage `fenris`/`meshx` or custom mesh structs.

4. **Problem assemblers**
   - **Darcy:** assemble diffusion matrix and load vector with coefficient lookup; support optional inflated boundary handling and constraint-aware assembly.
   - **Burgers:** assemble advection matrix and mass/diffusion matrices with lumping and constraint zeroing.
   - Provide unit/integration tests comparing against small analytic examples.

5. **Numerical linear algebra utilities**
   - Implement block-tridiagonal Cholesky factor type with forward/backward solves using `sprs` + dense block operations (or `ndarray` for blocks).

6. **SPDE / shallow-water model**
   - Port linear shallow-water SPDE discretization: mass/stiffness operators, Matern precision assembly, noise handling, and implicit-Euler state-space construction.
   - Implement GMRF and constrained-GMRF types with square-root factor interfaces; integrate iterative solver abstractions for precision solves.

7. **Examples and parity tests**
   - Reproduce Julia scripts as Rust examples/binaries that load datasets, assemble systems, and generate GMRF outputs.
   - Add CI with cargo fmt/clippy/test and benchmark hooks for key kernels.

8. **Documentation**
   - Document module mapping from Julia → Rust crates, usage examples, and migration notes; include guidance on substituting Julia dependencies.
