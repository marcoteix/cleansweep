use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError};
use anyhow::Result;

#[pyfunction]
pub fn check_positive(x: i32, strict: bool) -> PyResult<()> {

    let mut threshold: i32 = 0;
    let mut description = String::from("non-negative");

    if strict {
        threshold = 1;
        description = String::from("positive");

    }

    if x < threshold {
        return Err(PyValueError::new_err(
            format!("Expected a {} integer; got {}", description, x)))
    } else {
        Ok(())
    }
} 

#[pyclass(name = "Alleles", subclass)]
pub struct Alleles {
    pub vector: Vec<u8>
}

#[pymethods]
impl Alleles {

    #[new]
    pub fn new(vector: Vec<u8>) -> PyResult<Self> {
        Ok(Alleles { vector: vector })
    }

    pub fn len(&self) -> PyResult<usize> {
        Ok(self.vector.len())
    }

    pub fn copy_from_slice(&mut self, slice: &[u8], start: usize, end: usize) -> PyResult<()> {

        if end > self.len()? as usize {
            return Err(PyValueError::new_err(format!("End ({}) is greater than the vector size ({}).", end, self.len()?)))
        }

        self.vector[start..end].copy_from_slice(slice);

        Ok(())
    }

    pub fn clone(&self) -> PyResult<Alleles> {
        Ok(Alleles { vector: self.vector.clone() })
    }

}

#[derive(Clone)]
#[pyclass(name = "AltBC", subclass)]
pub struct AltBC {
    pub vector: Vec<i32>
}

#[pymethods]
impl AltBC {

    #[new]
    pub fn new(vector: Vec<i32>) -> PyResult<Self> {
        Ok(AltBC { vector: vector })
    }

    pub fn len(&self) -> PyResult<usize> {
        Ok(self.vector.len())
    }
}

impl AltBC {
    pub fn clone(&self) -> Result<AltBC> {
        Ok(AltBC{vector: self.vector.clone()})
    }
}

#[pyclass(name = "SamplingResult", subclass)]
pub struct SamplingResult {
    #[pyo3(get)]
    pub results: Vec<Py<ChainSamplingResult>>,
}

#[pymethods]
impl SamplingResult {

    #[new]
    pub fn new(chain_sampling_results: Vec<Py<ChainSamplingResult>>) -> PyResult<Self> {
        
        Ok(SamplingResult { results: chain_sampling_results })
    }
}

#[pyclass(name = "ChainSamplingResult", subclass)]
pub struct ChainSamplingResult {
    #[pyo3(get)]
    pub dispersion: Vec<f64>,
    #[pyo3(get)]
    pub alt_allele_proportion: Vec<f64>,
    #[pyo3(get)]
    pub alleles: Vec<i32>,
    #[pyo3(get)]
    pub acceptance_rate: f64,
}

#[pymethods]
impl ChainSamplingResult {

    #[new]
    pub fn new(dispersion: Vec<f64>, alt_allele_proportion: Vec<f64>, alleles: Vec<u8>, 
        acceptance_rate: f64) -> PyResult<Self> {

        let alleles_i32: Vec<i32> = alleles.iter().map(|&x| x as i32).collect();

        
        Ok(ChainSamplingResult { dispersion: dispersion, alt_allele_proportion: alt_allele_proportion, 
            alleles: alleles_i32, acceptance_rate: acceptance_rate})
    }
}

#[pyclass(name = "ModelParameters", subclass)]
pub struct ModelParameters {
    pub dispersion: f64,
    pub alt_allele_proportion: f64,
    pub alleles: Vec<u8>,
    pub n_accept: u32,
    pub n_proposals: u32,
    pub logp: f64,
}

#[pymethods]
impl ModelParameters {

    #[new]
    pub fn new(
        dispersion: f64,
        alt_allele_proportion: f64,
        alleles: Vec<u8>,
        n_accept: u32,
        n_proposals: u32,
        logp: f64,
    ) -> PyResult<Self> {

        Ok( ModelParameters { 
            dispersion: dispersion, 
            alt_allele_proportion: alt_allele_proportion, 
            alleles: alleles, 
            n_accept: n_accept, 
            n_proposals: n_proposals,
            logp: logp,
        } )
    }
}