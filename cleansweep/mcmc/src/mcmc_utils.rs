use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError};

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
    fn new(vector: Vec<u8>) -> PyResult<Self> {
        Ok(Alleles { vector: vector })
    }

}

#[pyclass(name = "AltBC", subclass)]
pub struct AltBC {
    pub vector: Vec<i32>
}

#[pymethods]
impl AltBC {

    #[new]
    fn new(vector: Vec<i32>) -> PyResult<Self> {
        Ok(AltBC { vector: vector })
    }

}