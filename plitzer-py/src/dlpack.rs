//! DLPack export of int32 data on the CPU, with the versioned (1.0) and the
//! legacy ABI. The structs follow `dlpack.h`.
//!
//! A consumer takes a capsule by renaming it (`dltensor` to `used_dltensor`)
//! and later calls the tensor's deleter. A capsule that nobody took still has
//! its first name when Python destroys it, and only then does its destructor
//! call the deleter. PyO3's capsule destructors cannot see the name, so this
//! module gives its own.

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;

use arrow_buffer::Buffer;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

const KDL_CPU: i32 = 1;
const KDL_INT: u8 = 0;
const FLAG_IS_COPIED: u64 = 1 << 1;
pub const CPU: (i32, i32) = (KDL_CPU, 0);

/// What a consumer's tensor keeps alive: the data and its shape and strides.
struct Context {
    data: Buffer,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

impl Context {
    fn new(data: Buffer, shape: &[usize]) -> Box<Self> {
        let shape: Vec<i64> = shape.iter().map(|&n| n as i64).collect();
        // C order, in elements. DLPack allows null strides for C order, but
        // some consumers want them set.
        let mut strides = vec![1i64; shape.len()];
        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }
        Box::new(Context {
            data,
            shape,
            strides,
        })
    }

    /// The tensor over this context. Its pointers stay valid while the
    /// context lives, as a `Box` does not move its contents.
    fn tensor(&mut self) -> DLTensor {
        DLTensor {
            data: self.data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: CPU.0,
                device_id: CPU.1,
            },
            ndim: self.shape.len() as i32,
            dtype: DLDataType {
                code: KDL_INT,
                bits: 32,
                lanes: 1,
            },
            shape: self.shape.as_mut_ptr(),
            strides: self.strides.as_mut_ptr(),
            byte_offset: 0,
        }
    }
}

/// The two managed tensor structs, as the deleter and the capsule
/// destructor see them.
trait Managed: Sized {
    const NAME: &'static CStr;
    fn deleter(&self) -> Option<unsafe extern "C" fn(*mut Self)>;
    fn context(&self) -> *mut c_void;
}

impl Managed for DLManagedTensor {
    const NAME: &'static CStr = c"dltensor";
    fn deleter(&self) -> Option<unsafe extern "C" fn(*mut Self)> {
        self.deleter
    }
    fn context(&self) -> *mut c_void {
        self.manager_ctx
    }
}

impl Managed for DLManagedTensorVersioned {
    const NAME: &'static CStr = c"dltensor_versioned";
    fn deleter(&self) -> Option<unsafe extern "C" fn(*mut Self)> {
        self.deleter
    }
    fn context(&self) -> *mut c_void {
        self.manager_ctx
    }
}

/// The deleter of both structs: frees the struct and its context.
unsafe extern "C" fn free<T: Managed>(tensor: *mut T) {
    if tensor.is_null() {
        return;
    }
    // SAFETY: `tensor` and its context come from `Box::into_raw` in `capsule`,
    // and DLPack calls the deleter once.
    let tensor = unsafe { Box::from_raw(tensor) };
    drop(unsafe { Box::from_raw(tensor.context().cast::<Context>()) });
}

/// Calls the deleter of a live tensor from `capsule`.
unsafe fn delete<T: Managed>(tensor: *mut T) {
    // SAFETY: the caller gives a live tensor.
    if let Some(deleter) = unsafe { (*tensor).deleter() } {
        unsafe { deleter(tensor) }
    }
}

/// Deletes the tensor of a capsule that no consumer renamed.
unsafe extern "C" fn destroy<T: Managed>(capsule: *mut ffi::PyObject) {
    // SAFETY: Python gives the capsule being destroyed. `PyCapsule_IsValid`
    // sets no error, and the pointer is ours while the name is unchanged.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, T::NAME.as_ptr()) == 1 {
            let tensor = ffi::PyCapsule_GetPointer(capsule, T::NAME.as_ptr());
            delete(tensor.cast::<T>());
        }
    }
}

/// A capsule that shares `data`, a C-order int32 array of `shape`, without a
/// copy. `versioned` picks the DLPack 1.0 ABI; `copied` sets its flag.
pub fn capsule<'py>(
    py: Python<'py>,
    data: Buffer,
    shape: &[usize],
    versioned: bool,
    copied: bool,
) -> PyResult<Bound<'py, PyCapsule>> {
    let mut context = Context::new(data, shape);
    let tensor = context.tensor();
    let manager_ctx = Box::into_raw(context).cast::<c_void>();
    if versioned {
        let managed = DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx,
            deleter: Some(free::<DLManagedTensorVersioned>),
            flags: if copied { FLAG_IS_COPIED } else { 0 },
            dl_tensor: tensor,
        };
        wrap(py, managed)
    } else {
        let managed = DLManagedTensor {
            dl_tensor: tensor,
            manager_ctx,
            deleter: Some(free::<DLManagedTensor>),
        };
        wrap(py, managed)
    }
}

fn wrap<T: Managed>(py: Python<'_>, managed: T) -> PyResult<Bound<'_, PyCapsule>> {
    let tensor = Box::into_raw(Box::new(managed));
    let pointer = NonNull::new(tensor.cast::<c_void>()).expect("a Box is not null");
    // SAFETY: the pointer is valid until `destroy` or a consumer deletes it.
    let capsule = unsafe {
        PyCapsule::new_with_pointer_and_destructor(py, pointer, T::NAME, Some(destroy::<T>))
    };
    if capsule.is_err() {
        // SAFETY: no capsule holds the tensor, so it is still ours alone.
        unsafe { delete(tensor) };
    }
    capsule
}
