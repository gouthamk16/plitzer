//! Decodes one token chunk with parquet's column reader. The values and the
//! levels go straight into buffers sized from the index record, and the row
//! offsets come from the repetition levels: no Arrow list is built, which
//! halves the decode time (D-94).

use std::sync::Arc;

use arrow_buffer::{Buffer, ScalarBuffer};
use bytes::Bytes;
use parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use parquet::column::reader::ColumnReaderImpl;
use parquet::data_type::{DataType, Int32Type, Int64Type};
use parquet::errors::ParquetError;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::serialized_reader::SerializedPageReader;
use parquet::schema::types::ColumnDescPtr;

use super::tokens::RowGroupTokens;
use crate::index::RowGroup;

/// Why a chunk does not decode to tokens.
pub(super) enum Fail {
    Parquet(ParquetError),
    /// A null token, or one that is negative or too large for `u32`.
    BadToken,
    /// The levels or the rows do not have the slots of the index record.
    Slots,
}

impl Fail {
    pub(super) fn at(self, path: String, row_group: u32) -> super::Error {
        match self {
            Fail::Parquet(source) => super::Error::Decode {
                path,
                row_group,
                source,
            },
            Fail::BadToken => super::Error::BadToken { path, row_group },
            Fail::Slots => super::Error::Slots { path, row_group },
        }
    }
}

impl From<ParquetError> for Fail {
    fn from(e: ParquetError) -> Self {
        Fail::Parquet(e)
    }
}

/// The tokens of `chunk`, the whole token column chunk of `rg`.
pub(super) fn decode(
    column: ColumnDescPtr,
    rg: &RowGroup,
    chunk: Bytes,
) -> Result<RowGroupTokens, Fail> {
    // The chunk starts at offset 0 of `chunk`, not at its place in the file.
    let meta = ColumnChunkMetaData::builder(column.clone())
        .set_compression(rg.codec.to_parquet())
        .set_data_page_offset(0)
        .set_total_compressed_size(chunk.len() as i64)
        .build()?;
    let pages = SerializedPageReader::new(Arc::new(chunk), &meta, rg.rows as usize, None)?;
    let unsigned = unsigned(&column);
    let tokens = match column.physical_type() {
        PhysicalType::INT32 => int32(read::<Int32Type>(column, pages, rg)?, unsigned)?,
        PhysicalType::INT64 => int64(read::<Int64Type>(column, pages, rg)?, unsigned)?,
        other => return Err(ParquetError::General(format!("a token column of {other}")).into()),
    };
    // A level that is neither a token nor an empty row, such as a null
    // struct around the integer, has a slot in the index but no token.
    if tokens.slots() != u64::from(rg.values) {
        return Err(Fail::Slots);
    }
    Ok(tokens)
}

/// INT32 values with no copy. UINT_8, UINT_16 and UINT_32 are stored with
/// the bits of a u32; a signed type must not be negative.
fn int32(
    (values, offsets): (Vec<i32>, Vec<usize>),
    unsigned: bool,
) -> Result<RowGroupTokens, Fail> {
    if !unsigned && values.iter().any(|&t| t < 0) {
        return Err(Fail::BadToken);
    }
    let len = values.len();
    let tokens = ScalarBuffer::new(Buffer::from_vec(values), 0, len);
    Ok(RowGroupTokens::from_parts(offsets, tokens))
}

/// INT64 values copied into u32, each in its range.
fn int64(
    (values, offsets): (Vec<i64>, Vec<usize>),
    unsigned: bool,
) -> Result<RowGroupTokens, Fail> {
    let token = |&t: &i64| match unsigned {
        true => u32::try_from(t as u64),
        false => u32::try_from(t),
    };
    let tokens: Result<Vec<u32>, _> = values.iter().map(token).collect();
    let tokens = tokens.map_err(|_| Fail::BadToken)?;
    Ok(RowGroupTokens::from_parts(offsets, tokens.into()))
}

/// The values of the chunk and the row offsets into them.
fn read<T: DataType>(
    column: ColumnDescPtr,
    pages: SerializedPageReader<Bytes>,
    rg: &RowGroup,
) -> Result<(Vec<T::T>, Vec<usize>), Fail> {
    let (rows, slots) = (rg.rows as usize, rg.values as usize);
    let max_def = column.max_def_level();
    let nullable = column.self_type().is_optional();
    let mut reader = ColumnReaderImpl::<T>::new(column, Box::new(pages));
    let (mut values, mut def, mut rep) = (buffer(slots)?, buffer(slots)?, buffer(slots)?);
    let (read, _, levels) =
        reader.read_records(rows, Some(&mut def), Some(&mut rep), &mut values)?;
    let more = reader
        .read_records(1, Some(&mut def), Some(&mut rep), &mut values)?
        .0;
    if read != rows || more != 0 {
        let why = format!("the chunk does not have the {rows} rows of its footer");
        return Err(ParquetError::General(why).into());
    }
    if levels != slots {
        return Err(Fail::Slots);
    }
    Ok((values, offsets(&def, &rep, max_def, nullable, rows)?))
}

/// An empty buffer with room for `slots` items. The count comes from the
/// footer, so a corrupt one gives an error, not an abort.
fn buffer<T>(slots: usize) -> Result<Vec<T>, Fail> {
    let mut v = Vec::new();
    v.try_reserve_exact(slots).map_err(|_| {
        ParquetError::General(format!("no memory for the {slots} slots of the footer"))
    })?;
    Ok(v)
}

/// The row offsets from the levels: a row starts at each repetition level 0,
/// and a token is a definition level at its maximum. One level below it is
/// a null token if the element is optional; lower levels are an empty or a
/// null list, which give an empty row.
fn offsets(
    def: &[i16],
    rep: &[i16],
    max_def: i16,
    nullable: bool,
    rows: usize,
) -> Result<Vec<usize>, Fail> {
    let mut offsets = Vec::with_capacity(rows + 1);
    let mut n = 0;
    for (&d, &r) in def.iter().zip(rep) {
        if r == 0 {
            offsets.push(n);
        }
        if d == max_def {
            n += 1;
        } else if nullable && d == max_def - 1 {
            return Err(Fail::BadToken);
        }
    }
    offsets.push(n);
    Ok(offsets)
}

fn unsigned(column: &ColumnDescPtr) -> bool {
    match column.logical_type_ref() {
        Some(LogicalType::Integer(int)) => !int.is_signed,
        _ => matches!(
            column.converted_type(),
            ConvertedType::UINT_8
                | ConvertedType::UINT_16
                | ConvertedType::UINT_32
                | ConvertedType::UINT_64
        ),
    }
}
