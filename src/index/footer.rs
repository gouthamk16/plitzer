use std::ops::Range;

use bytes::Bytes;
use object_store::{ObjectMeta, ObjectStore};
use parquet::DecodeResult;
use parquet::errors::ParquetError;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataPushDecoder};

/// Bytes read from the end of each file in the first request. FineWeb-Edu
/// footers are about 3.1 MiB, so this fits one in a single request; a larger
/// footer needs a second.
const TAIL: u64 = 4 << 20;

/// The file's footer, without its page indexes, which the index never needs.
pub(super) async fn read(
    store: &dyn ObjectStore,
    meta: &ObjectMeta,
) -> Result<ParquetMetaData, ParquetError> {
    let mut decoder = ParquetMetaDataPushDecoder::try_new(meta.size)?
        .with_page_index_policy(PageIndexPolicy::Skip);
    let (range, tail) = tail(store, meta).await?;
    decoder.push_range(range, tail)?;
    loop {
        match decoder.try_decode()? {
            DecodeResult::Data(metadata) => return Ok(metadata),
            DecodeResult::NeedsData(ranges) => {
                let data = store
                    .get_ranges(&meta.location, &ranges)
                    .await
                    .map_err(external)?;
                decoder.push_ranges(ranges, data)?;
            }
            DecodeResult::Finished => {
                return Err(ParquetError::General("no footer in file".into()));
            }
        }
    }
}

/// The last `TAIL` bytes of the file, which usually hold the whole footer,
/// so most files need one request. Also checks the footer length against
/// the file size: parquet 60 panics on a length larger than the file.
async fn tail(
    store: &dyn ObjectStore,
    meta: &ObjectMeta,
) -> Result<(Range<u64>, Bytes), ParquetError> {
    let range = meta.size.saturating_sub(TAIL)..meta.size;
    let mut data = store
        .get_ranges(&meta.location, std::slice::from_ref(&range))
        .await
        .map_err(external)?;
    let tail = data.pop().unwrap_or_default();
    let len = tail
        .len()
        .checked_sub(8)
        .map(|end| u32::from_le_bytes(tail[end..end + 4].try_into().unwrap()));
    match len {
        Some(len) if u64::from(len) + 8 <= meta.size => Ok((range, tail)),
        _ => Err(ParquetError::General(
            "footer length is larger than the file".into(),
        )),
    }
}

fn external(e: object_store::Error) -> ParquetError {
    ParquetError::External(Box::new(e))
}
