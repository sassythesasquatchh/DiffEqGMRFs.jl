//! Core utilities for the DiffEqGMRFs Rust port.
//!
//! This crate mirrors the Julia helpers in `src/metrics.jl` and `src/utils.jl`.
//! It provides:
//! - Metrics for comparing predicted and reference solutions.
//! - Lightweight mesh/discretization builders for 1D/2D domains with constraint bookkeeping.
//! - Numerical linear-algebra utilities for structured factorizations.

pub mod linalg;
pub mod mesh;
pub mod metrics;
