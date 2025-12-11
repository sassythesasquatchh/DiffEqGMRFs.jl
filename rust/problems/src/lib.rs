//! Problem-specific assemblers for the Rust port.
//!
//! These routines mirror the Julia helpers in `src/problems` and operate on the
//! lightweight discretizations defined in `diffeq-gmrfs-core`.

use diffeq_gmrfs_core::mesh::{Constraint, Discretization1D, Node2D, TriangulatedSquare};
use ndarray::{Array1, Array2};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AssemblyError {
    #[error("input vector has incorrect length (expected {expected}, got {actual})")]
    MismatchedVectorLength { expected: usize, actual: usize },
    #[error("coefficient grid shape does not match discretization")]
    CoefficientShape,
    #[error("discretization must contain at least two nodes")]
    InvalidMesh,
}

/// Assemble the nonlinear advection operator and volumetric term for Burgers.
///
/// This uses a simple first-order finite-volume style stencil consistent with
/// the 1D periodic discretization helper in the core crate.
pub fn assemble_burgers_advection_matrix(
    disc: &Discretization1D,
    cur_weights: &Array1<f64>,
) -> Result<(Array2<f64>, Array1<f64>), AssemblyError> {
    if cur_weights.len() != disc.nodes.len() {
        return Err(AssemblyError::MismatchedVectorLength {
            expected: disc.nodes.len(),
            actual: cur_weights.len(),
        });
    }

    let n = disc.nodes.len();
    if n < 2 {
        return Err(AssemblyError::InvalidMesh);
    }

    let spacings = node_spacings_1d(&disc.nodes);
    let mut g = Array2::<f64>::zeros((n, n));
    let mut v = Array1::<f64>::zeros(n);

    for i in 0..n - 1 {
        let h = spacings[i];
        let u_left = cur_weights[i];
        let u_right = cur_weights[i + 1];
        let u_avg = 0.5 * (u_left + u_right);
        let grad = (u_right - u_left) / h;

        let conv = u_avg / h;
        g[[i, i]] += conv;
        g[[i, i + 1]] -= conv;
        g[[i + 1, i]] += conv;
        g[[i + 1, i + 1]] -= conv;

        let mass_coeff = h / 2.0 * u_avg * grad;
        v[i] += mass_coeff;
        v[i + 1] += mass_coeff;
    }

    apply_constraints(&mut g, Some(&mut v), &disc.constraints);
    Ok((g, v))
}

/// Assemble the mass and diffusion matrices for Burgers.
pub fn assemble_burgers_mass_diffusion_matrices(
    disc: &Discretization1D,
    lumping: bool,
) -> Result<(Array2<f64>, Array2<f64>), AssemblyError> {
    let n = disc.nodes.len();
    if n < 2 {
        return Err(AssemblyError::InvalidMesh);
    }

    let spacings = node_spacings_1d(&disc.nodes);
    let mut m = Array2::<f64>::zeros((n, n));
    let mut g = Array2::<f64>::zeros((n, n));

    for i in 0..n - 1 {
        let h = spacings[i];
        let mass_local = h / 6.0;
        m[[i, i]] += 2.0 * mass_local;
        m[[i, i + 1]] += mass_local;
        m[[i + 1, i]] += mass_local;
        m[[i + 1, i + 1]] += 2.0 * mass_local;

        let diff_local = 1.0 / h;
        g[[i, i]] += diff_local;
        g[[i, i + 1]] -= diff_local;
        g[[i + 1, i]] -= diff_local;
        g[[i + 1, i + 1]] += diff_local;
    }

    if lumping {
        let diagonal: Vec<f64> = (0..n).map(|i| m.row(i).sum()).collect();
        m.fill(0.0);
        for (i, value) in diagonal.iter().enumerate() {
            m[[i, i]] = *value;
        }
    }

    apply_constraints(&mut m, None, &disc.constraints);
    apply_constraints(&mut g, None, &disc.constraints);
    Ok((m, g))
}

/// Assemble a diffusion matrix and load vector for the Darcy problem.
pub fn assemble_darcy_diff_matrix(
    disc: &TriangulatedSquare,
    coeff_mat: &Array2<f64>,
    beta: f64,
    inflated_boundary: bool,
) -> Result<(Array2<f64>, Array1<f64>, Option<Vec<usize>>), AssemblyError> {
    let n = disc.nodes.len();
    if n < 4 {
        return Err(AssemblyError::InvalidMesh);
    }

    let (x_vals, y_vals) = unique_coordinates(&disc.nodes);
    let nx = x_vals.len();
    let ny = y_vals.len();
    if coeff_mat.shape() != [nx, ny] {
        return Err(AssemblyError::CoefficientShape);
    }

    let hx = if nx > 1 { x_vals[1] - x_vals[0] } else { 1.0 };
    let hy = if ny > 1 { y_vals[1] - y_vals[0] } else { 1.0 };
    let row_stride = nx;

    let mut g = Array2::<f64>::zeros((n, n));
    let mut f = Array1::<f64>::zeros(n);
    let mut keep = if inflated_boundary { Some(Vec::new()) } else { None };

    for (idx, node) in disc.nodes.iter().enumerate() {
        let in_unit = node.x >= 0.0 && node.x <= 1.0 && node.y >= 0.0 && node.y <= 1.0;
        if inflated_boundary && !in_unit {
            if let Some(ref mut kept) = keep {
                kept.push(idx);
            }
        }
        if !in_unit {
            continue;
        }

        let x_idx = ((node.x - x_vals[0]) / hx).round() as usize;
        let y_idx = ((node.y - y_vals[0]) / hy).round() as usize;
        let coeff_val = coeff_mat[[x_idx, y_idx]];

        let center = idx;
        let left = if x_idx > 0 { Some(center - 1) } else { None };
        let right = if x_idx + 1 < nx { Some(center + 1) } else { None };
        let down = if y_idx > 0 {
            Some(center - row_stride)
        } else {
            None
        };
        let up = if y_idx + 1 < ny {
            Some(center + row_stride)
        } else {
            None
        };

        let fx = coeff_val / (hx * hx);
        let fy = coeff_val / (hy * hy);
        let mut diagonal = 0.0;
        if let Some(l) = left {
            g[[center, l]] -= fx;
            g[[l, center]] -= fx;
            diagonal += fx;
        }
        if let Some(r) = right {
            g[[center, r]] -= fx;
            g[[r, center]] -= fx;
            diagonal += fx;
        }
        if let Some(d) = down {
            g[[center, d]] -= fy;
            g[[d, center]] -= fy;
            diagonal += fy;
        }
        if let Some(u) = up {
            g[[center, u]] -= fy;
            g[[u, center]] -= fy;
            diagonal += fy;
        }
        g[[center, center]] += diagonal;
        f[center] += beta * hx * hy;
    }

    apply_constraints(&mut g, Some(&mut f), &disc.constraints);
    Ok((g, f, keep))
}

fn node_spacings_1d(nodes: &[f64]) -> Vec<f64> {
    nodes
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect()
}

fn apply_constraints(matrix: &mut Array2<f64>, mut rhs: Option<&mut Array1<f64>>, constraints: &[Constraint]) {
    for constraint in constraints {
        match *constraint {
            Constraint::Dirichlet { node, value } => {
                let n = matrix.ncols();
                for j in 0..n {
                    matrix[[node, j]] = 0.0;
                    matrix[[j, node]] = 0.0;
                }
                matrix[[node, node]] = 1.0;
                if let Some(r) = rhs.as_mut() {
                    r[node] = value;
                }
            }
            Constraint::Periodic {
                master,
                slave,
                weight,
            } => {
                let n = matrix.ncols();
                for j in 0..n {
                    matrix[[master, j]] += weight * matrix[[slave, j]];
                    matrix[[slave, j]] = 0.0;
                }
                for i in 0..n {
                    matrix[[i, master]] += weight * matrix[[i, slave]];
                    matrix[[i, slave]] = 0.0;
                }
                matrix[[slave, slave]] = 1.0;
                if let Some(r) = rhs.as_mut() {
                    r[master] += weight * r[slave];
                    r[slave] = 0.0;
                }
            }
        }
    }
}

fn unique_coordinates(nodes: &[Node2D]) -> (Vec<f64>, Vec<f64>) {
    let mut xs: Vec<f64> = nodes.iter().map(|n| n.x).collect();
    let mut ys: Vec<f64> = nodes.iter().map(|n| n.y).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    ys.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffeq_gmrfs_core::mesh::{periodic_unit_interval_discretization, uniform_unit_square_discretization};
    use ndarray::array;

    #[test]
    fn burgers_mass_and_diffusion_match_linear_elements() {
        let disc = Discretization1D {
            nodes: vec![0.0, 0.5, 1.0],
            element_order: 1,
            quadrature_order: 2,
            constraints: vec![],
            boundary_noise: 0.0,
        };
        let (m, g) = assemble_burgers_mass_diffusion_matrices(&disc, false).unwrap();
        let h = 0.5;
        let m_local = h / 6.0;
        let expected_mass = array![
            [2.0 * m_local, m_local, 0.0],
            [m_local, 4.0 * m_local, m_local],
            [0.0, m_local, 2.0 * m_local],
        ];
        assert!(m
            .iter()
            .zip(expected_mass.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));

        let k = 1.0 / h;
        let expected_diff = array![
            [k, -k, 0.0],
            [-k, 2.0 * k, -k],
            [0.0, -k, k],
        ];
        assert!(g
            .iter()
            .zip(expected_diff.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn burgers_advection_respects_periodic_constraint() {
        let disc = periodic_unit_interval_discretization(2, 1, 0.0).unwrap();
        let weights = array![1.0, 2.0, 3.0];
        let (g, v) = assemble_burgers_advection_matrix(&disc, &weights).unwrap();
        // periodic constraint should collapse first/last rows
        assert_eq!(g[[2, 2]], 1.0);
        assert_eq!(v.len(), 3);
        assert!(g.iter().any(|val| val.abs() > 0.0));
    }

    #[test]
    fn darcy_diffusion_assembles_interior() {
        let mesh = uniform_unit_square_discretization(2, 0.0, true, 1, 0.0).unwrap();
        let coeff = array![
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let (g, f, keep) = assemble_darcy_diff_matrix(&mesh, &coeff, 1.0, false).unwrap();
        assert_eq!(keep, None);
        // Interior node is at index 4 for a 3x3 grid
        assert!(g[(4, 4)] > 0.0);
        assert!(f[4] > 0.0);
    }
}
