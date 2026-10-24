//! A resumable checkpoint, as bytes. All integers are little-endian:
//!
//! ```text
//! magic "PLTZCKPT" | version u32 | column (len u32, UTF-8) | seed u64 | lanes u32 |
//! window u32 | seq_len u32 | docs u32 | cap u32 | pad u32 | fingerprint u64 |
//! epoch u64 | first_lane u32 | count u32 | each lane: step u64 | doc u64 | offset u32 |
//! checksum u64 (FNV-1a of all bytes before it)
//! ```

use std::fmt;
use std::ops::Range;

use crate::codec::{Reader, fnv1a, len, put_str};
use crate::pack::{LaneState, Packing};

const MAGIC: &[u8; 8] = b"PLTZCKPT";
const VERSION: u32 = 1;

/// The settings that fix the order and the sequences. A resume with any
/// other value would give other batches, so a checkpoint refuses it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub column: String,
    pub seed: u64,
    /// B: the lanes of the global batch.
    pub lanes: u32,
    /// W: row groups per window.
    pub window: u32,
    pub packing: Packing,
}

/// The state of lanes `first_lane..first_lane + lanes.len()` before a step
/// of epoch `epoch`. One rank's part holds its own lanes; a global
/// checkpoint holds all B lanes and resumes on any number of GPUs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub config: Config,
    /// `Index::fingerprint` of the dataset.
    pub fingerprint: u64,
    pub epoch: u64,
    pub first_lane: u32,
    pub lanes: Vec<LaneState>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The bytes are not a complete checkpoint.
    Bytes(&'static str),
    /// A checkpoint of another format version.
    Version(u32),
    /// The bytes were changed after they were written.
    Checksum,
    /// The checkpoint was made with another setting or dataset.
    Mismatch {
        field: &'static str,
        saved: String,
        now: String,
    },
    /// Parts that do not combine, or a part for other lanes.
    Parts(String),
    /// A lane state or the epoch outside its range, which no run can make.
    Range(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bytes(what) => write!(f, "not a plitzer checkpoint: {what}"),
            Error::Version(v) => write!(
                f,
                "checkpoint version {v}, but this plitzer reads {VERSION}"
            ),
            Error::Checksum => write!(f, "the checkpoint bytes are damaged"),
            Error::Mismatch { field, saved, now } => {
                write!(
                    f,
                    "the checkpoint has {field} = {saved}, but this run has {now}"
                )
            }
            Error::Parts(why) | Error::Range(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

impl Checkpoint {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend(VERSION.to_le_bytes());
        let c = &self.config;
        put_str(&mut out, &c.column);
        out.extend(c.seed.to_le_bytes());
        let p = c.packing;
        for v in [c.lanes, c.window, p.seq_len, p.docs, p.cap, p.pad] {
            out.extend(v.to_le_bytes());
        }
        out.extend(self.fingerprint.to_le_bytes());
        out.extend(self.epoch.to_le_bytes());
        out.extend(self.first_lane.to_le_bytes());
        out.extend(len(self.lanes.len()).to_le_bytes());
        for s in &self.lanes {
            out.extend(s.step.to_le_bytes());
            out.extend(s.doc.to_le_bytes());
            out.extend(s.offset.to_le_bytes());
        }
        out.extend(fnv1a(&out).to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !bytes.starts_with(MAGIC) {
            return Err(Error::Bytes("no magic"));
        }
        // The checksum comes first, so damage to the version field is not
        // reported as a checkpoint from another version.
        let (body, sum) = bytes
            .split_last_chunk::<8>()
            .ok_or(Error::Bytes("truncated"))?;
        if fnv1a(body) != u64::from_le_bytes(*sum) {
            return Err(Error::Checksum);
        }
        let mut r = Reader(body.get(MAGIC.len()..).ok_or(Error::Bytes("truncated"))?);
        let version = r.u32().ok_or(Error::Bytes("truncated"))?;
        if version != VERSION {
            return Err(Error::Version(version));
        }
        let checkpoint = fields(&mut r).ok_or(Error::Bytes("truncated"))?;
        if !r.0.is_empty() {
            return Err(Error::Bytes("trailing bytes"));
        }
        Ok(checkpoint)
    }

    /// True if `other` comes from the same run, epoch and step.
    fn same_position(&self, other: &Checkpoint) -> bool {
        let step = self.lanes.first().map(|s| s.step);
        other.config == self.config
            && other.fingerprint == self.fingerprint
            && other.epoch == self.epoch
            && other.lanes.iter().all(|s| Some(s.step) == step)
    }

    /// True if this checkpoint holds all B lanes.
    pub fn is_global(&self) -> bool {
        self.first_lane == 0 && self.lanes.len() == self.config.lanes as usize
    }

    /// One global checkpoint from the parts of all ranks, in any order.
    pub fn merge(mut parts: Vec<Checkpoint>) -> Result<Checkpoint, Error> {
        let Some(first) = parts.first() else {
            return Err(Error::Parts("no checkpoint parts to merge".into()));
        };
        let (config, fingerprint, epoch) = (first.config.clone(), first.fingerprint, first.epoch);
        if !parts.iter().all(|p| first.same_position(p)) {
            return Err(Error::Parts(
                "the parts come from different runs or steps".into(),
            ));
        }
        parts.sort_by_key(|p| p.first_lane);
        let mut lanes = Vec::with_capacity(config.lanes as usize);
        for p in parts {
            if p.first_lane as usize != lanes.len() {
                let why = format!("lane {} is missing or held twice", lanes.len());
                return Err(Error::Parts(why));
            }
            lanes.extend(p.lanes);
        }
        let merged = Checkpoint {
            config,
            fingerprint,
            epoch,
            first_lane: 0,
            lanes,
        };
        match merged.is_global() {
            true => Ok(merged),
            false => Err(Error::Parts("the parts do not hold all lanes".into())),
        }
    }

    /// The states of lanes `lanes`, which the checkpoint must hold. They
    /// must be at one step, since a loader steps all its lanes together.
    pub fn states(&self, lanes: Range<u32>) -> Result<Vec<LaneState>, Error> {
        let held = u64::from(self.first_lane)..u64::from(self.first_lane) + self.lanes.len() as u64;
        let (start, end) = (u64::from(lanes.start), u64::from(lanes.end));
        if start > end || start < held.start || end > held.end {
            let why = format!("the checkpoint holds lanes {held:?}, but this rank needs {lanes:?}");
            return Err(Error::Parts(why));
        }
        let from = (u64::from(lanes.start) - held.start) as usize;
        let states = &self.lanes[from..from + lanes.len()];
        if states.iter().any(|s| s.step != states[0].step) {
            return Err(Error::Parts("the lanes are at different steps".into()));
        }
        Ok(states.to_vec())
    }

    /// Refuses a checkpoint from another setting or dataset, and names the
    /// first field that differs.
    pub fn check(&self, config: &Config, fingerprint: u64) -> Result<(), Error> {
        let (a, b) = (&self.config, config);
        let (pa, pb) = (a.packing, b.packing);
        let fields: [(&'static str, String, String); 9] = [
            ("column", a.column.clone(), b.column.clone()),
            ("seed", a.seed.to_string(), b.seed.to_string()),
            ("lanes", a.lanes.to_string(), b.lanes.to_string()),
            ("window", a.window.to_string(), b.window.to_string()),
            ("seq_len", pa.seq_len.to_string(), pb.seq_len.to_string()),
            ("docs", pa.docs.to_string(), pb.docs.to_string()),
            ("cap", pa.cap.to_string(), pb.cap.to_string()),
            ("pad", pa.pad.to_string(), pb.pad.to_string()),
            (
                "fingerprint",
                self.fingerprint.to_string(),
                fingerprint.to_string(),
            ),
        ];
        match fields.into_iter().find(|(_, saved, now)| saved != now) {
            Some((field, saved, now)) => Err(Error::Mismatch { field, saved, now }),
            None => Ok(()),
        }
    }
}

/// The fields after the version, before the checksum.
fn fields(r: &mut Reader) -> Option<Checkpoint> {
    let column = r.str()?;
    let seed = r.u64()?;
    let [lanes, window, seq_len, docs, cap, pad] = [(); 6].map(|_| r.u32());
    let packing = Packing {
        seq_len: seq_len?,
        docs: docs?,
        cap: cap?,
        pad: pad?,
    };
    let config = Config {
        column,
        seed,
        lanes: lanes?,
        window: window?,
        packing,
    };
    let (fingerprint, epoch, first_lane) = (r.u64()?, r.u64()?, r.u32()?);
    // The count comes from the bytes, so nothing is preallocated from it.
    let mut states = Vec::new();
    for _ in 0..r.u32()? {
        let (step, doc, offset) = (r.u64()?, r.u64()?, r.u32()?);
        states.push(LaneState { step, doc, offset });
    }
    Some(Checkpoint {
        config,
        fingerprint,
        epoch,
        first_lane,
        lanes: states,
    })
}

#[cfg(test)]
mod tests;
