//! Batches, and their export to Python without a copy.

use std::sync::Arc;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{Array as _, ArrayRef, FixedSizeListArray, Int32Array};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

use crate::dlpack;

/// A C-order int32 array that Python reads through DLPack or Arrow. Tokens
/// are `u32` in Rust; vocabularies are far below 2³¹, and torch has little
/// `uint32` support.
#[pyclass(module = "plitzer._plitzer", frozen)]
pub struct Array {
    data: Buffer,
    shape: Vec<usize>,
    name: &'static str,
}

impl Array {
    pub fn new(data: Vec<u32>, shape: Vec<usize>, name: &'static str) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Array {
            data: Buffer::from_vec(data),
            shape,
            name,
        }
    }

    /// A 1-D array is an `Int32Array`. A 2-D array is a fixed-size list per
    /// row, typed `arrow.fixed_shape_tensor`, so pyarrow sees a tensor.
    fn arrow(&self) -> PyResult<(Field, ArrayRef)> {
        let values = Int32Array::new(
            ScalarBuffer::new(self.data.clone(), 0, self.data.len() / 4),
            None,
        );
        match self.shape[..] {
            [_] => Ok((
                Field::new(self.name, DataType::Int32, false),
                Arc::new(values),
            )),
            [_, cols] => {
                let item = Arc::new(Field::new("item", DataType::Int32, false));
                let size = i32::try_from(cols)
                    .map_err(|_| PyValueError::new_err("row too long for Arrow"))?;
                let list = FixedSizeListArray::new(item, size, Arc::new(values), None);
                let metadata = [
                    (
                        "ARROW:extension:name".to_string(),
                        "arrow.fixed_shape_tensor".to_string(),
                    ),
                    (
                        "ARROW:extension:metadata".to_string(),
                        format!("{{\"shape\":[{cols}]}}"),
                    ),
                ];
                let field = Field::new(self.name, list.data_type().clone(), false).with_metadata(
                    metadata
                        .into_iter()
                        .collect::<std::collections::HashMap<_, _>>(),
                );
                Ok((field, Arc::new(list)))
            }
            _ => unreachable!("arrays are 1-D or 2-D"),
        }
    }
}

#[pymethods]
impl Array {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    /// The DLPack capsule. `max_version` from 1.0 on gives the versioned ABI;
    /// none gives the legacy ABI for older consumers.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        // A CPU array has no stream to wait on.
        let _ = stream;
        if dl_device.is_some_and(|d| d != dlpack::CPU) {
            return Err(PyBufferError::new_err("plitzer arrays are on the CPU only"));
        }
        let copied = copy == Some(true);
        let data = if copied {
            Buffer::from(self.data.as_slice())
        } else {
            self.data.clone()
        };
        let versioned = max_version.is_some_and(|(major, _)| major >= 1);
        dlpack::capsule(py, data, &self.shape, versioned, copied)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        dlpack::CPU
    }

    /// The Arrow PyCapsule interface. `requested_schema` is ignored, as the
    /// interface allows.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let _ = requested_schema;
        let (field, array) = self.arrow()?;
        let schema =
            FFI_ArrowSchema::try_from(&field).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let array = FFI_ArrowArray::new(&array.to_data());
        Ok((
            PyCapsule::new_with_value(py, schema, c"arrow_schema")?,
            PyCapsule::new_with_value(py, array, c"arrow_array")?,
        ))
    }

    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, dtype=int32)", self.shape)
    }
}

/// One step of one GPU: a sequence of L tokens per lane, and the count of
/// real (not pad) tokens in each.
#[pyclass(module = "plitzer._plitzer", frozen)]
pub struct Batch {
    #[pyo3(get)]
    tokens: Py<Array>,
    #[pyo3(get)]
    lengths: Py<Array>,
}

impl Batch {
    pub fn new(
        py: Python<'_>,
        tokens: Vec<u32>,
        lengths: Vec<u32>,
        seq_len: usize,
    ) -> PyResult<Self> {
        let lanes = lengths.len();
        Ok(Batch {
            tokens: Py::new(py, Array::new(tokens, vec![lanes, seq_len], "tokens"))?,
            lengths: Py::new(py, Array::new(lengths, vec![lanes], "lengths"))?,
        })
    }
}
