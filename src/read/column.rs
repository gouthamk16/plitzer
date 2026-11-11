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
    /// The levels do not have the slots of the index record.
    Slots,
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
    match column.physical_type() {
        PhysicalType::INT32 => {
            let (values, offsets) = read::<Int32Type>(column, pages, rg)?;
            // UINT_8, UINT_16 and UINT_32 are stored with the bits of a u32.
            if !unsigned && values.iter().any(|&t| t < 0) {
                return Err(Fail::BadToken);
            }
            let len = values.len();
            let tokens = ScalarBuffer::new(Buffer::from_vec(values), 0, len);
            Ok(RowGroupTokens::from_parts(offsets, tokens))
        }
        PhysicalType::INT64 => {
            let (values, offsets) = read::<Int64Type>(column, pages, rg)?;
            let fits = |&t: &i64| match unsigned {
                true => (t as u64) <= u64::from(u32::MAX),
                false => (0..=i64::from(u32::MAX)).contains(&t),
            };
            if !values.iter().all(fits) {
                return Err(Fail::BadToken);
            }
            let tokens: Vec<u32> = values.iter().map(|&t| t as u32).collect();
            Ok(RowGroupTokens::from_parts(offsets, tokens.into()))
        }
        other => Err(ParquetError::General(format!("a token column of {other}")).into()),
    }
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
    let (mut values, mut def, mut rep) = (
        Vec::with_capacity(slots),
        Vec::with_capacity(slots),
        Vec::with_capacity(slots),
    );
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
