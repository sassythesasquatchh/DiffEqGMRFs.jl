use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct Node2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    Dirichlet {
        node: usize,
        value: f64,
    },
    Periodic {
        master: usize,
        slave: usize,
        weight: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Discretization1D {
    pub nodes: Vec<f64>,
    pub element_order: usize,
    pub quadrature_order: usize,
    pub constraints: Vec<Constraint>,
    pub boundary_noise: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TriangulatedSquare {
    pub nodes: Vec<Node2D>,
    /// Indices into `nodes`, two triangles per cell.
    pub elements: Vec<[usize; 3]>,
    pub element_order: usize,
    pub quadrature_order: usize,
    pub constraints: Vec<Constraint>,
    pub boundary_noise: f64,
}

#[derive(Debug, Error)]
pub enum MeshError {
    #[error("element order must be at least 1")] // align with common FEM expectations
    InvalidElementOrder,
    #[error("N must be positive")]
    InvalidResolution,
}

/// Periodic constraint tying the first and last node of the unit interval.
pub fn periodic_unit_interval_discretization(
    n: usize,
    element_order: usize,
    boundary_noise: f64,
) -> Result<Discretization1D, MeshError> {
    if n == 0 {
        return Err(MeshError::InvalidResolution);
    }
    if element_order == 0 {
        return Err(MeshError::InvalidElementOrder);
    }
    let step = 1.0 / n as f64;
    let nodes: Vec<f64> = (0..=n).map(|i| i as f64 * step).collect();

    let constraints = vec![Constraint::Periodic {
        master: 0,
        slave: nodes.len() - 1,
        weight: 1.0,
    }];

    Ok(Discretization1D {
        nodes,
        element_order,
        quadrature_order: element_order + 1,
        constraints,
        boundary_noise,
    })
}

/// Uniform triangular discretization of a square with optional boundary padding and Dirichlet data.
pub fn uniform_unit_square_discretization(
    n_xy: usize,
    boundary_width: f64,
    use_dirichlet_bc: bool,
    element_order: usize,
    boundary_noise: f64,
) -> Result<TriangulatedSquare, MeshError> {
    if n_xy == 0 {
        return Err(MeshError::InvalidResolution);
    }
    if element_order == 0 {
        return Err(MeshError::InvalidElementOrder);
    }

    let spacing = 1.0 / n_xy as f64;
    let x_range = evenly_spaced(-boundary_width, 1.0 + boundary_width, spacing);
    let y_range = evenly_spaced(-boundary_width, 1.0 + boundary_width, spacing);

    let mut nodes = Vec::with_capacity(x_range.len() * y_range.len());
    for &y in &y_range {
        for &x in &x_range {
            nodes.push(Node2D { x, y });
        }
    }

    let row_stride = x_range.len();
    let mut elements = Vec::new();
    for j in 0..y_range.len() - 1 {
        for i in 0..x_range.len() - 1 {
            let lower_left = j * row_stride + i;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + row_stride;
            let upper_right = upper_left + 1;

            // two triangles per quad cell
            elements.push([lower_left, lower_right, upper_right]);
            elements.push([lower_left, upper_right, upper_left]);
        }
    }

    let mut constraints = Vec::new();
    if use_dirichlet_bc {
        constraints.extend(boundary_dirichlet_nodes(
            &nodes,
            x_range.first().copied().unwrap(),
            x_range.last().copied().unwrap(),
            y_range.first().copied().unwrap(),
            y_range.last().copied().unwrap(),
        ));
    }

    Ok(TriangulatedSquare {
        nodes,
        elements,
        element_order,
        quadrature_order: element_order + 1,
        constraints,
        boundary_noise,
    })
}

fn boundary_dirichlet_nodes(
    nodes: &[Node2D],
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
) -> Vec<Constraint> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| {
            if (node.x - min_x).abs() < f64::EPSILON
                || (node.x - max_x).abs() < f64::EPSILON
                || (node.y - min_y).abs() < f64::EPSILON
                || (node.y - max_y).abs() < f64::EPSILON
            {
                Some(Constraint::Dirichlet {
                    node: idx,
                    value: 0.0,
                })
            } else {
                None
            }
        })
        .collect()
}

fn evenly_spaced(start: f64, end: f64, step: f64) -> Vec<f64> {
    let count = ((end - start) / step).round() as usize;
    (0..=count).map(|i| start + i as f64 * step).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_periodic_interval() {
        let discretization = periodic_unit_interval_discretization(4, 2, 1e-2).unwrap();
        assert_eq!(discretization.nodes, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(discretization.element_order, 2);
        assert_eq!(discretization.quadrature_order, 3);
        assert_eq!(discretization.constraints.len(), 1);
        match &discretization.constraints[0] {
            Constraint::Periodic {
                master,
                slave,
                weight,
            } => {
                assert_eq!((*master, *slave), (0, 4));
                assert!((*weight - 1.0).abs() < 1e-12);
            }
            _ => panic!("expected periodic constraint"),
        }
    }

    #[test]
    fn builds_uniform_square_with_dirichlet() {
        let mesh = uniform_unit_square_discretization(2, 0.0, true, 2, 1e-2).unwrap();
        assert_eq!(mesh.nodes.len(), 9); // (n+1)^2 nodes
        assert_eq!(mesh.elements.len(), 8); // 2 triangles per square cell, 2x2 grid -> 4 cells
        assert!(!mesh.constraints.is_empty());
        assert!(mesh.constraints.iter().all(
            |c| matches!(c, Constraint::Dirichlet { value, .. } if (*value - 0.0).abs() < 1e-12)
        ));
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(matches!(
            periodic_unit_interval_discretization(0, 1, 0.0),
            Err(MeshError::InvalidResolution)
        ));
        assert!(matches!(
            uniform_unit_square_discretization(0, 0.0, false, 1, 0.0),
            Err(MeshError::InvalidResolution)
        ));
        assert!(matches!(
            uniform_unit_square_discretization(1, 0.0, false, 0, 0.0),
            Err(MeshError::InvalidElementOrder)
        ));
    }
}
