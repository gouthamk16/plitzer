use arrow_buffer::ScalarBuffer;

/// The tokens of every row of one row group.
#[derive(Clone, Debug, Default)]
pub struct RowGroupTokens {
    /// Row `r` is `values[offsets[r]..offsets[r + 1]]`.
    offsets: Vec<usize>,
    values: ScalarBuffer<u32>,
}

impl RowGroupTokens {
    /// Row `r` is `values[offsets[r]..offsets[r + 1]]`.
    pub(super) fn from_parts(offsets: Vec<usize>, values: ScalarBuffer<u32>) -> Self {
        RowGroupTokens { offsets, values }
    }

    pub fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// One slot for each token, and one for each row with none: the leaf
    /// value count of the column chunk.
    pub fn slots(&self) -> u64 {
        let rows = self.offsets.windows(2);
        rows.map(|r| (r[1] - r[0]).max(1) as u64).sum()
    }

    pub fn view(&self) -> crate::pack::Rows<'_> {
        crate::pack::Rows {
            values: &self.values,
            offsets: &self.offsets,
        }
    }

    pub fn row(&self, row: usize) -> &[u32] {
        &self.values[self.offsets[row]..self.offsets[row + 1]]
    }

    /// The memory the tokens hold, for the prefetch byte budget.
    pub fn bytes(&self) -> usize {
        self.values.len() * 4 + self.offsets.len() * size_of::<usize>()
    }
}
