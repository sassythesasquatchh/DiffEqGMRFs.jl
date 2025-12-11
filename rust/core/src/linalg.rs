use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
use sprs::CsMat;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LinalgError {
    #[error("matrix dimension {found} is not divisible into {blocks} blocks")]
    NonDivisibleBlocks { found: usize, blocks: usize },
    #[error("matrix is not square")]
    NonSquareMatrix,
    #[error("cholesky factorization failed: matrix not positive definite")]
    NotPositiveDefinite,
    #[error("rhs length {found} does not match matrix dimension {expected}")]
    DimensionMismatch { found: usize, expected: usize },
}

/// Block-tridiagonal Cholesky factorization specialized for symmetric positive-definite systems.
///
/// Mirrors the Julia helper in `src/tridiagonal_cholesky.jl` but stores dense diagonal blocks and
/// block couplings to support efficient forward/backward solves.
#[derive(Debug, Clone)]
pub struct BlockTridiagonalCholesky {
    dim: usize,
    block_size: usize,
    factors: Vec<Cholesky<f64, Dyn>>, // lower-triangular Cholesky factors
    cs: Vec<DMatrix<f64>>,            // coupling blocks between neighbors
}

impl BlockTridiagonalCholesky {
    /// Factor a symmetric positive-definite block-tridiagonal matrix.
    pub fn factor(matrix: &CsMat<f64>, n_blocks: usize) -> Result<Self, LinalgError> {
        if matrix.rows() != matrix.cols() {
            return Err(LinalgError::NonSquareMatrix);
        }
        let dim = matrix.rows();
        if dim % n_blocks != 0 {
            return Err(LinalgError::NonDivisibleBlocks {
                found: dim,
                blocks: n_blocks,
            });
        }
        let block_size = dim / n_blocks;

        let mut factors = Vec::with_capacity(n_blocks);
        let mut cs = Vec::with_capacity(n_blocks.saturating_sub(1));

        let first_block = dense_block(matrix, 0, block_size, 0, block_size);
        let first_chol = first_block
            .cholesky()
            .ok_or(LinalgError::NotPositiveDefinite)?;
        factors.push(first_chol);

        for block_idx in 1..n_blocks {
            let row_start = block_idx * block_size;
            let row_end = row_start + block_size;
            let col_start_prev = (block_idx - 1) * block_size;
            let col_end_prev = col_start_prev + block_size;

            let b_block = dense_block(matrix, row_start, row_end, col_start_prev, col_end_prev);
            let c = solve_lower(factors.last().unwrap(), b_block.transpose())?.transpose();
            cs.push(c.clone());

            let diag_block = dense_block(matrix, row_start, row_end, row_start, row_end);
            let schur = diag_block - &c * c.transpose();
            let cho = schur.cholesky().ok_or(LinalgError::NotPositiveDefinite)?;
            factors.push(cho);
        }

        Ok(Self {
            dim,
            block_size,
            factors,
            cs,
        })
    }

    /// Solve the lower-triangular system L y = b using the stored block factors.
    pub fn forward_solve(&self, b: &DVector<f64>) -> Result<Vec<DVector<f64>>, LinalgError> {
        if b.len() != self.dim {
            return Err(LinalgError::DimensionMismatch {
                found: b.len(),
                expected: self.dim,
            });
        }
        let mut x = Vec::with_capacity(self.factors.len());
        for (idx, block_rhs) in b.as_slice().chunks(self.block_size).enumerate() {
            let rhs_vec = DVector::from_column_slice(block_rhs);
            if idx == 0 {
                let solved = solve_lower(
                    self.factors.first().unwrap(),
                    vector_to_matrix(rhs_vec.clone()),
                )?;
                x.push(solved.column(0).into());
            } else {
                let adjustment = &self.cs[idx - 1] * &x[idx - 1];
                let solved =
                    solve_lower(&self.factors[idx], vector_to_matrix(rhs_vec - adjustment))?;
                x.push(solved.column(0).into());
            }
        }
        Ok(x)
    }

    /// Solve the upper-triangular system L^T x = y given the intermediate forward solve results.
    pub fn backward_solve(
        &self,
        y_blocks: &[DVector<f64>],
    ) -> Result<Vec<DVector<f64>>, LinalgError> {
        if y_blocks.len() != self.factors.len() {
            return Err(LinalgError::DimensionMismatch {
                found: y_blocks.len(),
                expected: self.factors.len(),
            });
        }
        let mut x = y_blocks.to_vec();
        let last_idx = self.factors.len() - 1;
        x[last_idx] = solve_upper(self.factors.last().unwrap(), &x[last_idx])?;
        for i in (0..last_idx).rev() {
            let correction = self.cs[i].transpose() * &x[i + 1];
            x[i] = solve_upper(&self.factors[i], &(x[i].clone() - correction))?;
        }
        Ok(x)
    }

    /// Solve the full system (L L^T) x = b.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, LinalgError> {
        let y_blocks = self.forward_solve(b)?;
        let x_blocks = self.backward_solve(&y_blocks)?;
        let mut result = DVector::zeros(self.dim);
        for (idx, block) in x_blocks.iter().enumerate() {
            let start = idx * self.block_size;
            result.rows_mut(start, self.block_size).copy_from(block);
        }
        Ok(result)
    }
}

fn solve_lower(chol: &Cholesky<f64, Dyn>, rhs: DMatrix<f64>) -> Result<DMatrix<f64>, LinalgError> {
    let mut rhs = rhs;
    if !chol.l().solve_lower_triangular_mut(&mut rhs) {
        return Err(LinalgError::NotPositiveDefinite);
    }
    Ok(rhs)
}

fn solve_upper(chol: &Cholesky<f64, Dyn>, rhs: &DVector<f64>) -> Result<DVector<f64>, LinalgError> {
    let mut rhs = vector_to_matrix(rhs.clone());
    if !chol.l().transpose().solve_upper_triangular_mut(&mut rhs) {
        return Err(LinalgError::NotPositiveDefinite);
    }
    Ok(rhs.column(0).into())
}

fn vector_to_matrix(vec: DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(vec.len(), 1, vec.as_slice())
}

fn dense_block(
    matrix: &CsMat<f64>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> DMatrix<f64> {
    let nrows = row_end - row_start;
    let ncols = col_end - col_start;
    let mut block = DMatrix::zeros(nrows, ncols);

    for (row_idx, row_vec) in matrix.outer_iterator().enumerate() {
        if row_idx < row_start || row_idx >= row_end {
            continue;
        }
        for (col_idx, value) in row_vec.iter() {
            if col_idx < col_start || col_idx >= col_end {
                continue;
            }
            block[(row_idx - row_start, col_idx - col_start)] = *value;
        }
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block_tridiagonal(n_blocks: usize, block_size: usize) -> CsMat<f64> {
        let dim = n_blocks * block_size;
        let mut tri = sprs::TriMat::new((dim, dim));

        for block in 0..n_blocks {
            for i in 0..block_size {
                let row = block * block_size + i;
                tri.add_triplet(row, row, 4.0 + (block as f64));
                if i + 1 < block_size {
                    tri.add_triplet(row, row + 1, 1.0);
                    tri.add_triplet(row + 1, row, 1.0);
                }
            }
            if block + 1 < n_blocks {
                let row_start = block * block_size;
                let col_start = (block + 1) * block_size;
                for i in 0..block_size {
                    tri.add_triplet(row_start + i, col_start + i, -0.2);
                    tri.add_triplet(col_start + i, row_start + i, -0.2);
                }
            }
        }

        tri.to_csr()
    }

    #[test]
    fn factors_and_solves_block_tridiagonal() {
        let n_blocks = 3;
        let block_size = 2;
        let matrix = build_block_tridiagonal(n_blocks, block_size);
        let factor = BlockTridiagonalCholesky::factor(&matrix, n_blocks).unwrap();

        let rhs = DVector::from_iterator(
            n_blocks * block_size,
            (0..n_blocks * block_size).map(|i| i as f64 + 1.0),
        );
        let solution = factor.solve(&rhs).unwrap();

        // Validate against dense solution
        let dense = dense_block(&matrix, 0, n_blocks * block_size, 0, n_blocks * block_size);
        let expected = dense.cholesky().unwrap().solve(&rhs);

        assert!((solution - expected).norm() < 1e-8);
    }

    #[test]
    fn errors_on_mismatched_rhs() {
        let n_blocks = 2;
        let block_size = 2;
        let matrix = build_block_tridiagonal(n_blocks, block_size);
        let factor = BlockTridiagonalCholesky::factor(&matrix, n_blocks).unwrap();

        let rhs = DVector::from_element(3, 1.0);
        let err = factor.forward_solve(&rhs).unwrap_err();
        match err {
            LinalgError::DimensionMismatch { found, expected } => {
                assert_eq!(found, 3);
                assert_eq!(expected, 4);
            }
            _ => panic!("unexpected error"),
        }
    }
}
