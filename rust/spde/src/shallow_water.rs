use diffeq_gmrfs_core::mesh::{Constraint, TriangulatedSquare};
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use std::sync::Arc;
use thiserror::Error;

use crate::gmrf::{dense_to_csmat, ConstrainedGMRF, GMRF};

#[derive(Debug, Error)]
pub enum SpdeError {
    #[error("time grid must contain at least two entries")]
    InsufficientTimes,
    #[error("time grid must be strictly increasing")]
    NonMonotoneTimes,
    #[error("matrix inversion failed during implicit Euler step")]
    SingularStep,
}

/// Linear shallow water SPDE parameters mirroring the Julia helper.
pub struct LinearShallowWaterSpde {
    pub depth_fn: Arc<dyn Fn((f64, f64)) -> f64 + Send + Sync>,
    pub tau: f64,
    pub k: f64,
    pub coriolis: f64,
    pub gravity: f64,
}

impl Default for LinearShallowWaterSpde {
    fn default() -> Self {
        Self {
            depth_fn: Arc::new(|_| 1.0),
            tau: 1.0,
            k: 0.0,
            coriolis: 0.0,
            gravity: 9.81,
        }
    }
}

impl LinearShallowWaterSpde {
    pub fn new(
        depth_fn: impl Fn((f64, f64)) -> f64 + Send + Sync + 'static,
        tau: f64,
        k: f64,
        coriolis: f64,
        gravity: f64,
    ) -> Self {
        Self {
            depth_fn: Arc::new(depth_fn),
            tau,
            k,
            coriolis,
            gravity,
        }
    }
}

/// Discretize the linear shallow-water SPDE on a uniform square mesh using an implicit Euler scheme.
pub fn discretize_shallow_water(
    spde: &LinearShallowWaterSpde,
    disc: &TriangulatedSquare,
    ts: &[f64],
    kappa_matern: f64,
    mean_offset: f64,
) -> Result<ConstrainedGMRF, SpdeError> {
    if ts.len() < 2 {
        return Err(SpdeError::InsufficientTimes);
    }
    if ts.windows(2).any(|w| w[1] <= w[0]) {
        return Err(SpdeError::NonMonotoneTimes);
    }

    let (mass, stiffness, coupling) = assemble_system(spde, disc);

    let state_dim = mass.nrows();
    let constrained_dofs = expand_constraints(&disc.constraints, state_dim / 3);

    let mut mass_with_bc = mass.clone();
    apply_dense_constraints(
        &mut mass_with_bc,
        &constrained_dofs,
        disc.boundary_noise,
        ConstraintTarget::Mass,
    );

    let mut stiffness_with_bc = stiffness.clone();
    apply_dense_constraints(
        &mut stiffness_with_bc,
        &constrained_dofs,
        1.0,
        ConstraintTarget::General,
    );
    let mut coupling_with_bc = coupling.clone();
    apply_dense_constraints(
        &mut coupling_with_bc,
        &constrained_dofs,
        1.0,
        ConstraintTarget::General,
    );

    let m_inv_diag: Vec<f64> = mass_with_bc
        .diagonal()
        .iter()
        .map(|v| if *v > 0.0 { 1.0 / *v } else { 0.0 })
        .collect();
    let m_inv = DMatrix::from_diagonal(&DVector::from_vec(m_inv_diag.clone()));
    let m_inv_sqrt = DMatrix::from_diagonal(&DVector::from_vec(
        m_inv_diag
            .iter()
            .map(|v| if *v > 0.0 { v.sqrt() } else { 0.0 })
            .collect(),
    ));

    let k_matern = stiffness_with_bc.clone() + kappa_matern * kappa_matern * mass_with_bc.clone();
    let ratio = 1.0 / (8.0 * std::f64::consts::PI * kappa_matern.powi(4));
    let q_matern = ratio * k_matern.transpose() * &m_inv * &k_matern;
    let q_matern_sqrt = (ratio.sqrt()) * k_matern.transpose() * &m_inv_sqrt;

    let prior_precision = dense_to_csmat(&q_matern);
    let prior_sqrt = dense_to_csmat(&q_matern_sqrt);
    let _prior = GMRF::new(Array1::zeros(state_dim), prior_precision, Some(prior_sqrt));

    let noise_diag = build_noise_diag(state_dim, spde.tau, disc.boundary_noise, &constrained_dofs);
    let mean_vec = build_mean_offset(state_dim, ts.len(), mean_offset, &constrained_dofs);

    let joint_precision = implicit_euler_joint_precision(
        &mass_with_bc,
        &coupling_with_bc,
        &noise_diag,
        ts,
        &q_matern,
    )?;
    let joint_precision_sparse = dense_to_csmat(&joint_precision);
    let joint_sqrt = joint_precision
        .clone()
        .cholesky()
        .map(|chol| dense_to_csmat(&chol.l()));

    let joint = GMRF::new(mean_vec, joint_precision_sparse, joint_sqrt);
    Ok(ConstrainedGMRF::new(joint, constrained_dofs))
}

fn assemble_system(
    spde: &LinearShallowWaterSpde,
    disc: &TriangulatedSquare,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let n_nodes = disc.nodes.len();
    let state_dim = 3 * n_nodes;
    let mut mass = DMatrix::zeros(state_dim, state_dim);
    let mut stiffness = DMatrix::zeros(state_dim, state_dim);
    let mut coupling = DMatrix::zeros(state_dim, state_dim);

    let (x_vals, y_vals) = unique_coordinates(&disc.nodes);
    let hx = if x_vals.len() > 1 {
        x_vals[1] - x_vals[0]
    } else {
        1.0
    };
    let hy = if y_vals.len() > 1 {
        y_vals[1] - y_vals[0]
    } else {
        1.0
    };
    let area = hx * hy;
    let row_stride = x_vals.len();

    for (idx, node) in disc.nodes.iter().enumerate() {
        let h_idx = idx;
        let u_idx = idx + n_nodes;
        let v_idx = idx + 2 * n_nodes;
        let depth = (spde.depth_fn)((node.x, node.y));

        mass[(h_idx, h_idx)] += area;
        mass[(u_idx, u_idx)] += area;
        mass[(v_idx, v_idx)] += area;

        let left = if idx % row_stride > 0 {
            Some(idx - 1)
        } else {
            None
        };
        let right = if (idx + 1) % row_stride != 0 {
            Some(idx + 1)
        } else {
            None
        };
        let down = if idx >= row_stride {
            Some(idx - row_stride)
        } else {
            None
        };
        let up = if idx + row_stride < n_nodes {
            Some(idx + row_stride)
        } else {
            None
        };

        // Laplacian-style stiffness for each field.
        let mut diag_acc = 0.0;
        for neighbor in [left, right] {
            if let Some(n) = neighbor {
                let weight = 1.0 / hx.powi(2);
                add_sym_entry(&mut stiffness, h_idx, n, -weight);
                add_sym_entry(&mut stiffness, u_idx, n + n_nodes, -weight);
                add_sym_entry(&mut stiffness, v_idx, n + 2 * n_nodes, -weight);
                diag_acc += weight;
            }
        }
        for neighbor in [down, up] {
            if let Some(n) = neighbor {
                let weight = 1.0 / hy.powi(2);
                add_sym_entry(&mut stiffness, h_idx, n, -weight);
                add_sym_entry(&mut stiffness, u_idx, n + n_nodes, -weight);
                add_sym_entry(&mut stiffness, v_idx, n + 2 * n_nodes, -weight);
                diag_acc += weight;
            }
        }
        stiffness[(h_idx, h_idx)] += diag_acc;
        stiffness[(u_idx, u_idx)] += diag_acc;
        stiffness[(v_idx, v_idx)] += diag_acc;

        // h-u coupling (divergence of velocity)
        if let Some(n) = left {
            coupling[(h_idx, n + n_nodes)] += depth / (2.0 * hx);
        }
        if let Some(n) = right {
            coupling[(h_idx, n + n_nodes)] += -depth / (2.0 * hx);
        }
        if let Some(n) = down {
            coupling[(h_idx, n + 2 * n_nodes)] += depth / (2.0 * hy);
        }
        if let Some(n) = up {
            coupling[(h_idx, n + 2 * n_nodes)] += -depth / (2.0 * hy);
        }

        // momentum equations: gradient of height
        if let Some(n) = left {
            coupling[(u_idx, n)] += spde.gravity / (2.0 * hx);
        }
        if let Some(n) = right {
            coupling[(u_idx, n)] += -spde.gravity / (2.0 * hx);
        }
        if let Some(n) = down {
            coupling[(v_idx, n)] += spde.gravity / (2.0 * hy);
        }
        if let Some(n) = up {
            coupling[(v_idx, n)] += -spde.gravity / (2.0 * hy);
        }

        // friction and Coriolis effects on velocity
        coupling[(u_idx, u_idx)] += spde.k * area;
        coupling[(v_idx, v_idx)] += spde.k * area;
        coupling[(u_idx, v_idx)] += -spde.coriolis * area;
        coupling[(v_idx, u_idx)] += spde.coriolis * area;
    }

    (mass, stiffness, coupling)
}

fn implicit_euler_joint_precision(
    mass: &DMatrix<f64>,
    coupling: &DMatrix<f64>,
    noise_diag: &[f64],
    ts: &[f64],
    prior_precision: &DMatrix<f64>,
) -> Result<DMatrix<f64>, SpdeError> {
    let state_dim = mass.nrows();
    let steps = ts.len();
    let mut precision = DMatrix::zeros(state_dim * steps, state_dim * steps);

    // Prior on the initial state.
    add_block(&mut precision, 0, 0, prior_precision);

    for (i, window) in ts.windows(2).enumerate() {
        let dt = window[1] - window[0];
        let lhs = mass + coupling * dt;
        let solver = lhs.clone().lu();
        let a = solver.solve(mass).ok_or(SpdeError::SingularStep)?;

        let q_noise_inv = DMatrix::from_diagonal(&DVector::from_iterator(
            state_dim,
            noise_diag.iter().map(|&v| 1.0 / (dt * v * v)),
        ));

        let a_t_q = a.transpose() * &q_noise_inv;
        let upper_left = &a_t_q * &a;
        let upper_right = -a_t_q.clone();
        let lower_left = -&q_noise_inv * &a;
        let lower_right = q_noise_inv.clone();

        let block_start = i * state_dim;
        add_block(&mut precision, block_start, block_start, &upper_left);
        add_block(
            &mut precision,
            block_start,
            block_start + state_dim,
            &upper_right,
        );
        add_block(
            &mut precision,
            block_start + state_dim,
            block_start,
            &lower_left,
        );
        add_block(
            &mut precision,
            block_start + state_dim,
            block_start + state_dim,
            &lower_right,
        );
    }

    Ok(precision)
}

#[derive(Clone, Copy)]
enum ConstraintTarget {
    Mass,
    General,
}

fn apply_dense_constraints(
    matrix: &mut DMatrix<f64>,
    constrained_dofs: &[usize],
    boundary_noise: f64,
    target: ConstraintTarget,
) {
    let penalty = match target {
        ConstraintTarget::Mass => boundary_noise,
        ConstraintTarget::General => 1.0,
    };
    for &dof in constrained_dofs {
        for j in 0..matrix.ncols() {
            matrix[(dof, j)] = 0.0;
            matrix[(j, dof)] = 0.0;
        }
        matrix[(dof, dof)] = penalty;
    }
}

fn build_noise_diag(
    state_dim: usize,
    tau: f64,
    boundary_noise: f64,
    constrained_dofs: &[usize],
) -> Vec<f64> {
    let base = if tau > 0.0 { tau } else { f64::EPSILON };
    let boundary = if boundary_noise > 0.0 {
        boundary_noise
    } else {
        f64::EPSILON
    };
    let mut diag = vec![base; state_dim];
    for &dof in constrained_dofs {
        diag[dof] = boundary;
    }
    diag
}

fn build_mean_offset(
    state_dim: usize,
    time_steps: usize,
    mean_offset: f64,
    constrained_dofs: &[usize],
) -> Array1<f64> {
    let total = state_dim * time_steps;
    let mut mean = Array1::from_elem(total, mean_offset);
    for &dof in constrained_dofs {
        for step in 0..time_steps {
            mean[dof + step * state_dim] = 0.0;
        }
    }
    mean
}

fn expand_constraints(constraints: &[Constraint], n_nodes: usize) -> Vec<usize> {
    let mut dofs = Vec::new();
    for constraint in constraints {
        if let Constraint::Dirichlet { node, .. } = *constraint {
            for offset in 0..3 {
                dofs.push(node + offset * n_nodes);
            }
        }
    }
    dofs
}

fn unique_coordinates(nodes: &[diffeq_gmrfs_core::mesh::Node2D]) -> (Vec<f64>, Vec<f64>) {
    let mut xs: Vec<f64> = nodes.iter().map(|n| n.x).collect();
    let mut ys: Vec<f64> = nodes.iter().map(|n| n.y).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    ys.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    (xs, ys)
}

fn add_sym_entry(matrix: &mut DMatrix<f64>, i: usize, j: usize, value: f64) {
    matrix[(i, j)] += value;
    matrix[(j, i)] += value;
}

fn add_block(target: &mut DMatrix<f64>, row: usize, col: usize, block: &DMatrix<f64>) {
    let rows = block.nrows();
    let cols = block.ncols();
    for i in 0..rows {
        for j in 0..cols {
            target[(row + i, col + j)] += block[(i, j)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffeq_gmrfs_core::mesh::uniform_unit_square_discretization;

    #[test]
    fn assembles_coupled_system() {
        let disc = uniform_unit_square_discretization(2, 0.0, false, 1, 1e-2).unwrap();
        let spde = LinearShallowWaterSpde::default();
        let (mass, stiffness, coupling) = assemble_system(&spde, &disc);

        let state_dim = 3 * disc.nodes.len();
        assert_eq!(mass.shape(), (state_dim, state_dim));
        assert_eq!(stiffness.shape(), (state_dim, state_dim));
        assert_eq!(coupling.shape(), (state_dim, state_dim));

        // Diagonal mass entries should be positive and coupling should contain momentum-height links.
        assert!(mass.diagonal().iter().all(|v| *v > 0.0));
        assert!(coupling.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn discretization_includes_constraints_and_prior() {
        let disc = uniform_unit_square_discretization(1, 0.0, true, 1, 1e-2).unwrap();
        let spde = LinearShallowWaterSpde::new(|_| 1.0, 0.5, 0.1, 0.0, 9.81);
        let ts = vec![0.0, 0.1, 0.2];
        let gmrf = discretize_shallow_water(&spde, &disc, &ts, 1.0, 0.1).unwrap();

        let state_dim = 3 * disc.nodes.len();
        assert_eq!(gmrf.len(), state_dim * ts.len());
        assert!(!gmrf.constrained_dofs.is_empty());

        for dof in &gmrf.constrained_dofs {
            let idx = *dof + state_dim; // second time step
            assert_eq!(gmrf.base.mean[idx], 0.0);
        }

        // Prior precision should carry through to the joint precision diagonal.
        let mut diag = Vec::new();
        for i in 0..gmrf.base.precision.rows() {
            diag.push(gmrf.base.precision.get(i, i).copied().unwrap_or(0.0));
        }
        assert!(diag.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn solver_wrapper_solves_precision_system() {
        let disc = uniform_unit_square_discretization(1, 0.0, false, 1, 1e-2).unwrap();
        let spde = LinearShallowWaterSpde::default();
        let ts = vec![0.0, 0.2];
        let gmrf = discretize_shallow_water(&spde, &disc, &ts, 1.0, 0.0).unwrap();

        use crate::gmrf::PrecisionSolver;

        let solver = crate::gmrf::CholeskyPrecisionSolver::from_precision(&gmrf.base.precision)
            .expect("factorization should succeed");
        let rhs = Array1::ones(gmrf.len());
        let solution = solver.solve(&rhs);
        assert_eq!(solution.len(), gmrf.len());
    }
}
