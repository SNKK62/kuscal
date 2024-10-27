use std::fmt::Display;
use std::io::{Read, Write};

pub fn serialize_size(sz: usize, writer: &mut impl Write) -> std::io::Result<()> {
    writer.write_all(&(sz as u32).to_le_bytes())
}

pub fn deserialize_size(reader: &mut impl Read) -> std::io::Result<usize> {
    let mut buf = [0u8; std::mem::size_of::<u32>()];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf) as usize)
}

pub fn serialize_str(s: &str, writer: &mut impl Write) -> std::io::Result<()> {
    serialize_size(s.len(), writer)?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

pub fn deserialize_str(reader: &mut impl Read) -> std::io::Result<String> {
    let mut buf = vec![0u8; deserialize_size(reader)?];
    reader.read_exact(&mut buf)?;
    let s = String::from_utf8(buf).unwrap();
    Ok(s)
}

#[repr(u8)]
pub enum ValueKind {
    F64,
    I64,
    Str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    F64(f64),
    I64(i64),
    Str(String),
}

impl Default for Value {
    fn default() -> Self {
        Self::F64(0.)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64(value) => write!(f, "{value}"),
            Self::I64(value) => write!(f, "{value}"),
            Self::Str(value) => write!(f, "{value}"),
        }
    }
}

impl Value {
    pub fn kind(&self) -> ValueKind {
        match self {
            Self::F64(_) => ValueKind::F64,
            Self::I64(_) => ValueKind::I64,
            Self::Str(_) => ValueKind::Str,
        }
    }

    pub fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let kind = self.kind() as u8;
        writer.write_all(&[kind])?;
        match self {
            Self::F64(value) => {
                writer.write_all(&value.to_le_bytes())?;
            }
            Self::I64(value) => {
                writer.write_all(&value.to_le_bytes())?;
            }
            Self::Str(value) => serialize_str(value, writer)?,
        };
        Ok(())
    }

    #[allow(non_upper_case_globals)]
    pub fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
        const F64: u8 = ValueKind::F64 as u8;
        const I64: u8 = ValueKind::I64 as u8;
        const Str: u8 = ValueKind::Str as u8;

        let mut kind_buf = [0u8; 1];
        reader.read_exact(&mut kind_buf)?;
        match kind_buf[0] {
            F64 => {
                let mut buf = [0u8; std::mem::size_of::<f64>()];
                reader.read_exact(&mut buf)?;
                Ok(Value::F64(f64::from_le_bytes(buf)))
            }
            I64 => {
                let mut buf = [0u8; std::mem::size_of::<i64>()];
                reader.read_exact(&mut buf)?;
                Ok(Value::I64(i64::from_le_bytes(buf)))
            }
            Str => Ok(Value::Str(deserialize_str(reader)?)),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "ValueKind {} does not match to any known kinds",
                    kind_buf[0]
                ),
            )),
        }
    }

    pub fn coerce_f64(&self) -> f64 {
        match self {
            Self::F64(value) => *value,
            Self::I64(value) => *value as f64,
            _ => panic!("Coercion failed: {:?} cannot be coerced to f64", self),
        }
    }

    pub fn coerce_i64(&self) -> i64 {
        match self {
            Self::F64(value) => *value as i64,
            Self::I64(value) => *value,
            _ => panic!("Coercion failed: {:?} cannot be coerced to i64", self),
        }
    }

    pub fn coerce_str(&self) -> String {
        match self {
            Self::F64(value) => format!("{value}"),
            Self::I64(value) => format!("{value}"),
            Self::Str(value) => value.clone(),
        }
    }
}
