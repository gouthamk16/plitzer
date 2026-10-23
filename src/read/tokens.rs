use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrowPrimitiveType, GenericListArray, OffsetSizeTrait};
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;

/// The tokens of every row of one row group.
#[derive(Clone, Debug, Default)]
pub struct RowGroupTokens {
    /// Row `r` is `values[offsets[r]..offsets[r + 1]]`.
    offsets: Vec<usize>,
    values: ScalarBuffer<u32>,
}

impl RowGroupTokens {
    /// The tokens of a list column, or `None` if a token is null, negative
    /// or larger than `u32`. Int32 and UInt32 values are not copied.
    pub(super) fn new(column: &dyn Array) -> Option<Self> {
        match column.data_type() {
            DataType::List(_) => Self::from_list(column.as_list::<i32>()),
            DataType::LargeList(_) => Self::from_list(column.as_list::<i64>()),
            _ => None,
        }
    }

    pub fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn row(&self, row: usize) -> &[u32] {
        &self.values[self.offsets[row]..self.offsets[row + 1]]
    }

    /// The memory the tokens hold, for the prefetch byte budget.
    pub fn bytes(&self) -> usize {
        self.values.len() * 4 + self.offsets.len() * size_of::<usize>()
    }

    fn from_list<O: OffsetSizeTrait>(list: &GenericListArray<O>) -> Option<Self> {
        let offsets: Vec<usize> = list.value_offsets().iter().map(|o| o.as_usize()).collect();
        // A null row is empty, which the parquet reader gives as equal offsets.
        let null_has_tokens =
            (0..list.len()).any(|r| list.is_null(r) && offsets[r] != offsets[r + 1]);
        let values = list.values();
        if null_has_tokens || values.null_count() > 0 {
            return None;
        }
        let values = match values.data_type() {
            DataType::Int32 => {
                let v = values.as_primitive::<Int32Type>().values();
                v.iter()
                    .all(|&t| t >= 0)
                    .then(|| ScalarBuffer::new(v.inner().clone(), 0, v.len()))?
            }
            DataType::UInt32 => values.as_primitive::<UInt32Type>().values().clone(),
            DataType::Int64 => converted::<Int64Type>(values)?,
            DataType::UInt64 => converted::<UInt64Type>(values)?,
            DataType::Int16 => converted::<Int16Type>(values)?,
            DataType::UInt16 => converted::<UInt16Type>(values)?,
            DataType::Int8 => converted::<Int8Type>(values)?,
            DataType::UInt8 => converted::<UInt8Type>(values)?,
            _ => return None,
        };
        Some(RowGroupTokens { offsets, values })
    }
}

/// Values of another integer type, copied into `u32`.
fn converted<T: ArrowPrimitiveType>(values: &dyn Array) -> Option<ScalarBuffer<u32>>
where
    T::Native: TryInto<u32>,
{
    let v = values.as_primitive::<T>().values();
    let tokens: Option<Vec<u32>> = v.iter().map(|&t| t.try_into().ok()).collect();
    tokens.map(ScalarBuffer::from)
}
