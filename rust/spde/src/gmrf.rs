use ndarray::{Array1, ArrayView1};
use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, TriMat};

/// Gaussian Markov random field with optional square-root precision for sampling or solves.
#[derive(Debug, Clone)]
pub struct GMRF {
    pub mean: Array1<f64>,
    pub precision: CsMat<f64>,
    pub sqrt_precision: Option<CsMat<f64>>, 
}

impl GMRF {
    pub fn new(mean: Array1<f64>, precision: CsMat<f64>, sqrt_precision: Option<CsMat<f64>>) -> Self {
        Self {
            mean,
            precision,
            sqrt_precision,
        }
    }

    pub fn len(&self) -> usize {
        self.mean.len()
    }
}

/// Wrapper that records constrained degrees of freedom from the spatial discretization.
#[derive(Debug, Clone)]
pub struct ConstrainedGMRF {
    pub base: GMRF,
    pub constrained_dofs: Vec<usize>,
}

impl ConstrainedGMRF {
    pub fn new(base: GMRF, constrained_dofs: Vec<usize>) -> Self {
        Self {
            base,
            constrained_dofs,
        }
    }

    pub fn len(&self) -> usize {
        self.base.len()
    }
}

/// Abstraction for precision solvers so iterative/direct methods can share an API.
pub trait PrecisionSolver {
    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64>;
}

/// Direct Cholesky-based precision solver built on dense matrices for small test systems.
pub struct CholeskyPrecisionSolver {
    factor: nalgebra::Cholesky<f64, nalgebra::Dyn>,
}

impl CholeskyPrecisionSolver {
    pub fn from_precision(precision: &CsMat<f64>) -> Option<Self> {
        let dense = csmat_to_dense(precision);
        dense.cholesky().map(|factor| Self { factor })
    }
}

impl PrecisionSolver for CholeskyPrecisionSolver {
    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let rhs_vec = DVector::from_column_slice(rhs.as_slice().unwrap());
        let solution = self.factor.solve(&rhs_vec);
        Array1::from(solution.as_slice().to_vec())
    }
}

fn csmat_to_dense(mat: &CsMat<f64>) -> DMatrix<f64> {
    let (rows, cols) = mat.shape();
    let mut dense = DMatrix::zeros(rows, cols);
    for (val, (row, col)) in mat.iter() {
        dense[(row, col)] = *val;
    }
    dense
}

pub fn dense_to_csmat(dense: &DMatrix<f64>) -> CsMat<f64> {
    let (rows, cols) = dense.shape();
    let mut triplet = TriMat::with_capacity((rows, cols), rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let value = dense[(i, j)];
            if value != 0.0 {
                triplet.add_triplet(i, j, value);
            }
        }
    }
    triplet.to_csr()
}

/// Helper to convert a 1D view into a diagonal sparse matrix.
pub fn diag_to_csmat(diag: ArrayView1<'_, f64>) -> CsMat<f64> {
    let n = diag.len();
    let mut triplet = TriMat::with_capacity((n, n), n);
    for (idx, value) in diag.iter().enumerate() {
        if *value != 0.0 {
            triplet.add_triplet(idx, idx, *value);
        }
    }
    triplet.to_csr()
}
