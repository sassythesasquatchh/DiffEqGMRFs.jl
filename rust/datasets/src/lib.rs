//! Dataset loaders mirroring the Julia `src/datasets` module.
//!
//! The loaders rely on the `matfile` crate to parse MATLAB v5 files and expose
//! the same helper accessors used throughout the DiffEqGMRFs examples.

use matfile::{Array, MatFile, NumericData};
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use num_traits::ToPrimitive;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("failed to open dataset: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse mat file: {0}")]
    Parse(#[from] matfile::Error),
    #[error("dataset is missing variable '{0}'")]
    MissingVariable(String),
    #[error("dataset contains complex values which are unsupported in the Rust port")]
    ComplexUnsupported,
    #[error("unable to reshape dataset array: {0}")]
    Shape(String),
}

/// Burgers equation snapshots and metadata.
#[derive(Debug, Clone)]
pub struct BurgersDataset {
    input: Array2<f64>,
    output: Array3<f64>,
    pub x_coords: Array1<f64>,
    pub ts: Array1<f64>,
    pub nu: f64,
}

impl BurgersDataset {
    pub fn from_mat_file<P: AsRef<Path>>(path: P) -> Result<Self, DatasetError> {
        let mut file = std::fs::File::open(path)?;
        let mat = MatFile::parse(&mut file)?;

        let input = read_named_array(&mat, "input")?
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|err| DatasetError::Shape(err.to_string()))?;
        let output = read_named_array(&mat, "output")?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|err| DatasetError::Shape(err.to_string()))?;
        let visc = read_named_array(&mat, "visc")?
            .into_dimensionality::<ndarray::IxDyn>()
            .map_err(|err| DatasetError::Shape(err.to_string()))?;
        let nu = *visc.first().ok_or_else(|| DatasetError::MissingVariable("visc".into()))?;

        Ok(Self::from_parts(input, output, nu))
    }

    pub fn from_parts(input: Array2<f64>, output: Array3<f64>, nu: f64) -> Self {
        let x_coords = Array1::linspace(0.0, 1.0, input.shape()[1]);
        let ts = Array1::linspace(0.0, 1.0, output.shape()[1]);
        Self {
            input,
            output,
            x_coords,
            ts,
            nu,
        }
    }

    pub fn len(&self) -> usize {
        self.output.shape()[0]
    }

    pub fn get_initial_condition(&self, idx: usize) -> Array1<f64> {
        self.input
            .slice(ndarray::s![idx, ..])
            .to_owned()
    }

    pub fn get_solution(&self, idx: usize) -> Array2<f64> {
        self.output
            .slice(ndarray::s![idx, .., ..])
            .to_owned()
    }
}

/// Darcy flow solutions and coefficients.
#[derive(Debug, Clone)]
pub struct DarcyDataset {
    sol: Array3<f64>,
    coeff: Array3<f64>,
    pub x_coords: Array1<f64>,
    pub y_coords: Array1<f64>,
}

impl DarcyDataset {
    pub fn from_mat_file<P: AsRef<Path>>(path: P) -> Result<Self, DatasetError> {
        let mut file = std::fs::File::open(path)?;
        let mat = MatFile::parse(&mut file)?;

        let sol = read_named_array(&mat, "sol")?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|err| DatasetError::Shape(err.to_string()))?;
        let coeff = read_named_array(&mat, "coeff")?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|err| DatasetError::Shape(err.to_string()))?;

        Ok(Self::from_parts(sol, coeff))
    }

    pub fn from_parts(sol: Array3<f64>, coeff: Array3<f64>) -> Self {
        let x_coords = Array1::linspace(0.0, 1.0, sol.shape()[1]);
        let y_coords = Array1::linspace(0.0, 1.0, sol.shape()[2]);
        Self {
            sol,
            coeff,
            x_coords,
            y_coords,
        }
    }

    pub fn len(&self) -> usize {
        self.sol.shape()[0]
    }

    pub fn get_problem(&self, idx: usize) -> (Array2<f64>, Array2<f64>) {
        (
            self.sol.slice(ndarray::s![idx, .., ..]).to_owned(),
            self.coeff.slice(ndarray::s![idx, .., ..]).to_owned(),
        )
    }
}

pub fn get_xy_idcs(point: (f64, f64), x_coords: &Array1<f64>, y_coords: &Array1<f64>) -> (usize, usize) {
    let x_idx = nearest_index(point.0, x_coords);
    let y_idx = nearest_index(point.1, y_coords);
    (x_idx, y_idx)
}

fn nearest_index(value: f64, coords: &Array1<f64>) -> usize {
    coords
        .iter()
        .enumerate()
        .min_by(|a, b| {
            let da = (a.1 - value).abs();
            let db = (b.1 - value).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn read_named_array(mat: &MatFile, name: &str) -> Result<ArrayD<f64>, DatasetError> {
    let array: &Array = mat
        .find_by_name(name)
        .ok_or_else(|| DatasetError::MissingVariable(name.to_string()))?;
    let values = numeric_to_f64(array.data())?;
    let shape = array.size().clone();
    let row_major_values = column_major_to_row_major(&values, &shape);
    ArrayD::from_shape_vec(IxDyn(&shape), row_major_values)
        .map_err(|err| DatasetError::Shape(err.to_string()))
}

fn numeric_to_f64(data: &NumericData) -> Result<Vec<f64>, DatasetError> {
    match data {
        NumericData::Double { real, imag } => ensure_real(real, imag),
        NumericData::Single { real, imag } => ensure_real(real, imag),
        NumericData::Int8 { real, imag } => ensure_real(real, imag),
        NumericData::UInt8 { real, imag } => ensure_real(real, imag),
        NumericData::Int16 { real, imag } => ensure_real(real, imag),
        NumericData::UInt16 { real, imag } => ensure_real(real, imag),
        NumericData::Int32 { real, imag } => ensure_real(real, imag),
        NumericData::UInt32 { real, imag } => ensure_real(real, imag),
        NumericData::Int64 { real, imag } => ensure_real(real, imag),
        NumericData::UInt64 { real, imag } => ensure_real(real, imag),
    }
}

fn ensure_real<T>(real: &[T], imag: &Option<Vec<T>>) -> Result<Vec<f64>, DatasetError>
where
    T: Copy + ToPrimitive,
{
    if imag.is_some() {
        return Err(DatasetError::ComplexUnsupported);
    }
    real
        .iter()
        .map(|&v| v.to_f64().ok_or(DatasetError::ComplexUnsupported))
        .collect()
}

fn column_major_to_row_major(values: &[f64], shape: &[usize]) -> Vec<f64> {
    let total = values.len();
    let mut out = vec![0.0; total];
    for (col_major_idx, &value) in values.iter().enumerate() {
        let coords = unravel_column_major(col_major_idx, shape);
        let row_major_idx = ravel_row_major(&coords, shape);
        out[row_major_idx] = value;
    }
    out
}

fn unravel_column_major(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &dim in shape {
        let coordinate = idx % dim;
        coords.push(coordinate);
        idx /= dim;
    }
    coords
}

fn ravel_row_major(coords: &[usize], shape: &[usize]) -> usize {
    let mut multiplier = 1;
    let mut index = 0;
    for (&coord, &dim) in coords.iter().rev().zip(shape.iter().rev()) {
        index += coord * multiplier;
        multiplier *= dim;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn burgers_from_parts_matches_julia_shapes() {
        let input = array![[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]];
        let output = array![
            [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
            [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]],
        ];
        let ds = BurgersDataset::from_parts(input, output, 0.01);
        assert_eq!(ds.len(), 2);
        assert!((ds.nu - 0.01).abs() < 1e-12);
        assert_eq!(ds.x_coords.len(), 3);
        assert_eq!(ds.ts.len(), 2);
        assert_eq!(ds.get_initial_condition(1), array![0.5, 0.5, 0.5]);
        assert_eq!(
            ds.get_solution(0),
            array![[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
        );
    }

    #[test]
    fn darcy_accessors_match_expected() {
        let sol = array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.1, 0.2], [0.3, 0.4]],
        ];
        let coeff = array![
            [[10.0, 20.0], [30.0, 40.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ];
        let ds = DarcyDataset::from_parts(sol, coeff);
        assert_eq!(ds.len(), 2);
        let (sol0, coeff0) = ds.get_problem(0);
        assert_eq!(sol0, array![[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(coeff0, array![[10.0, 20.0], [30.0, 40.0]]);
    }

    #[test]
    fn nearest_index_matches_midpoints() {
        let coords = Array1::linspace(0.0, 1.0, 5);
        assert_eq!(nearest_index(0.49, &coords), 2);
        assert_eq!(nearest_index(0.0, &coords), 0);
    }
}
