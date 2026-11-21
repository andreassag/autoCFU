//! Error types for the AutoCFU Rust inference crate.

use std::path::PathBuf;

use thiserror::Error;

/// Unified error type returned by all fallible operations in `autocfu`.
#[derive(Error, Debug)]
pub enum AutoCFUError {
    /// Invalid slicing, filtering, or merging configuration value.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Input image dimensions or geometric parameters are invalid.
    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    /// Referenced file path could not be found on disk.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// Standard filesystem I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// YAML configuration parsing failure.
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// JSON serialization failure.
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV export failure.
    #[error("CSV serialization error: {0}")]
    Csv(#[from] csv::Error),

    /// Image decoding or encoding failure.
    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    /// 16-bit TIFF instance mask encoding failure.
    #[error("TIFF encoding error: {0}")]
    Tiff(#[from] tiff::TiffError),

    /// ONNX Runtime initialization, tensor conversion, or execution error.
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),

    /// ONNX graph signature does not match CenterNet, StarDist, or Countability contracts.
    #[error("Unsupported or unrecognized ONNX model outputs: {0}")]
    UnsupportedModel(String),
}

impl From<ort::Error> for AutoCFUError {
    fn from(err: ort::Error) -> Self {
        Self::OnnxRuntime(err.to_string())
    }
}

/// Convenience result alias for `autocfu`.
pub type Result<T> = std::result::Result<T, AutoCFUError>;
