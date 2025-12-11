use std::path::Path;

use diffeq_gmrfs_core::mesh::{
    periodic_unit_interval_discretization, uniform_unit_square_discretization, MeshError,
};
use diffeq_gmrfs_core::metrics::rmse;
use diffeq_gmrfs_datasets::{BurgersDataset, DarcyDataset, DatasetError};
use diffeq_gmrfs_problems::{
    assemble_burgers_advection_matrix, assemble_burgers_mass_diffusion_matrices,
    assemble_darcy_diff_matrix, AssemblyError,
};
use diffeq_gmrfs_spde::shallow_water::{
    discretize_shallow_water, LinearShallowWaterSpde, SpdeError,
};
use ndarray::{array, Array3};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExampleError {
    #[error("dataset error: {0}")]
    Dataset(#[from] DatasetError),
    #[error("assembly error: {0}")]
    Assembly(#[from] AssemblyError),
    #[error("spde error: {0}")]
    Spde(#[from] SpdeError),
    #[error("mesh error: {0}")]
    Mesh(#[from] MeshError),
    #[error("unexpected empty dataset")] // defensive guard for tiny fixtures
    EmptyDataset,
}

pub struct BurgersExampleResult {
    pub rmse_to_first_snapshot: f64,
    pub drift_norm: f64,
}

pub struct DarcyExampleResult {
    pub keep_indices: usize,
    pub load_norm: f64,
}

pub struct ShallowWaterExampleResult {
    pub state_dim: usize,
    pub constrained_dofs: usize,
}

/// Demonstrates the Burgers assembly pipeline over the first trajectory in the dataset.
///
/// If `path` is provided the dataset is loaded from disk; otherwise a synthetic fixture is used.
/// The function returns the RMSE between an implicit Euler predictor and the first saved snapshot
/// alongside the magnitude of the drift term used to advance the solution.
pub fn run_burgers_example(path: Option<&Path>) -> Result<BurgersExampleResult, ExampleError> {
    let dataset = match path {
        Some(p) => BurgersDataset::from_mat_file(p)?,
        None => synthetic_burgers_dataset(),
    };

    let disc = periodic_unit_interval_discretization(dataset.x_coords.len() - 1, 1, 1e-4)?;
    let u0 = dataset.get_initial_condition(0);
    let target = dataset.get_solution(0).row(0).to_owned();
    let dt = dataset.ts.get(1).copied().unwrap_or(1.0) - dataset.ts.get(0).copied().unwrap_or(0.0);

    let (mass, diffusion) = assemble_burgers_mass_diffusion_matrices(&disc, false)?;
    let (advection, forcing) = assemble_burgers_advection_matrix(&disc, &u0)?;

    let mut drift = mass.dot(&u0);
    drift = drift + diffusion.dot(&u0);
    drift = drift + advection.dot(&u0);
    drift = drift + &forcing;

    let predictor = &u0 + &(drift.mapv(|v| dt * v));
    let error = rmse(&predictor, &target);
    let drift_norm = drift.iter().map(|v| v * v).sum::<f64>().sqrt();

    Ok(BurgersExampleResult {
        rmse_to_first_snapshot: error,
        drift_norm,
    })
}

/// Demonstrates Darcy diffusion assembly on the first coefficient grid.
///
/// If `path` is provided the dataset is loaded from disk; otherwise a synthetic coefficient and
/// solution field are generated.
pub fn run_darcy_example(path: Option<&Path>) -> Result<DarcyExampleResult, ExampleError> {
    let dataset = match path {
        Some(p) => DarcyDataset::from_mat_file(p)?,
        None => synthetic_darcy_dataset(),
    };

    let disc = uniform_unit_square_discretization(dataset.x_coords.len() - 1, 0.0, true, 1, 1e-4)?;
    let (sol, coeff) = dataset.get_problem(0);
    let (g, f, keep) = assemble_darcy_diff_matrix(&disc, &coeff, 1.0, false)?;

    let sol_len = sol.len();
    let sol_vec = sol
        .into_shape(sol_len)
        .map_err(|_| ExampleError::EmptyDataset)?;
    let response = g.dot(&sol_vec) + f;
    let load_norm = response.iter().map(|v| v * v).sum::<f64>().sqrt();

    Ok(DarcyExampleResult {
        keep_indices: keep.map(|k| k.len()).unwrap_or(0),
        load_norm,
    })
}

/// Demonstrates construction of the joint GMRF precision for the shallow-water SPDE.
pub fn run_shallow_water_example() -> Result<ShallowWaterExampleResult, ExampleError> {
    let disc = uniform_unit_square_discretization(2, 0.1, true, 1, 1e-3)?;
    let spde = LinearShallowWaterSpde::default();
    let ts = vec![0.0, 0.1, 0.2];

    let gmrf = discretize_shallow_water(&spde, &disc, &ts, 1.0, 0.0)?;
    Ok(ShallowWaterExampleResult {
        state_dim: gmrf.base.mean.len(),
        constrained_dofs: gmrf.constrained_dofs.len(),
    })
}

fn synthetic_burgers_dataset() -> BurgersDataset {
    let input = array![[0.2, 0.4, 0.6, 0.4, 0.2]];
    let output = Array3::from_shape_fn((1, 2, 5), |(_, t, j)| input[[0, j]] + 0.05 * t as f64);
    BurgersDataset::from_parts(input, output, 0.05)
}

fn synthetic_darcy_dataset() -> DarcyDataset {
    let sol = Array3::from_shape_fn((1, 3, 3), |(_, i, j)| (i as f64 + j as f64) / 4.0);
    let coeff = Array3::from_shape_fn((1, 3, 3), |(_, i, j)| 1.0 + 0.2 * (i + j) as f64);
    DarcyDataset::from_parts(sol, coeff)
}
