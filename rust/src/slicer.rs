//! Overlapping 2D grid slicing with mathematical parity to Python `autocfu.slicer`.

use crate::config::SlicingConfig;
use crate::error::{AutoCFUError, Result};

/// Axis-aligned 2D tile coordinates within a Petri dish image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceWindow {
    /// Inclusive left x-coordinate in pixels.
    pub xmin: usize,
    /// Inclusive top y-coordinate in pixels.
    pub ymin: usize,
    /// Exclusive right x-coordinate in pixels.
    pub xmax: usize,
    /// Exclusive bottom y-coordinate in pixels.
    pub ymax: usize,
}

impl SliceWindow {
    /// Width of the slice window in pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.xmax.saturating_sub(self.xmin)
    }

    /// Height of the slice window in pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.ymax.saturating_sub(self.ymin)
    }
}

/// 2D overlapping tile grid generator matching Python's [`PlateSlicer`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlateSlicer {
    /// Width of each tile in pixels.
    pub slice_width: usize,
    /// Height of each tile in pixels.
    pub slice_height: usize,
    /// Horizontal overlap ratio in `[0.0, 1.0)`.
    pub overlap_width_ratio: f64,
    /// Vertical overlap ratio in `[0.0, 1.0)`.
    pub overlap_height_ratio: f64,
    /// Horizontal stride in pixels (`floor(slice_width * (1 - overlap))`).
    pub step_x: usize,
    /// Vertical stride in pixels (`floor(slice_height * (1 - overlap))`).
    pub step_y: usize,
}

impl PlateSlicer {
    /// Construct a new slicer and compute integer strides.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidGeometry`] if slice dimensions are zero or
    /// overlap ratios fall outside `[0.0, 1.0)`.
    pub fn new(
        slice_width: usize,
        slice_height: usize,
        overlap_width_ratio: f64,
        overlap_height_ratio: f64,
    ) -> Result<Self> {
        if slice_width == 0 || slice_height == 0 {
            return Err(AutoCFUError::InvalidGeometry(format!(
                "Slice dimensions must be positive, got {slice_width}x{slice_height}"
            )));
        }
        if !(0.0..1.0).contains(&overlap_width_ratio) || !(0.0..1.0).contains(&overlap_height_ratio)
        {
            return Err(AutoCFUError::InvalidGeometry(
                "Overlap ratios must be in the range [0.0, 1.0)".to_string(),
            ));
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let step_x = ((slice_width as f64) * (1.0 - overlap_width_ratio))
            .floor()
            .max(1.0) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let step_y = ((slice_height as f64) * (1.0 - overlap_height_ratio))
            .floor()
            .max(1.0) as usize;

        Ok(Self {
            slice_width,
            slice_height,
            overlap_width_ratio,
            overlap_height_ratio,
            step_x,
            step_y,
        })
    }

    /// Build a [`PlateSlicer`] from a validated [`SlicingConfig`].
    ///
    /// # Errors
    /// Propagates validation errors from [`PlateSlicer::new`].
    pub fn from_config(config: &SlicingConfig) -> Result<Self> {
        Self::new(
            config.slice_width,
            config.slice_height,
            config.overlap_width_ratio,
            config.overlap_height_ratio,
        )
    }

    fn generate_1d_intervals(
        full_length: usize,
        slice_len: usize,
        step: usize,
    ) -> Vec<(usize, usize)> {
        if full_length <= slice_len {
            return vec![(0, full_length)];
        }

        let mut intervals = Vec::new();
        let mut start = 0usize;
        while start + slice_len <= full_length {
            intervals.push((start, start + slice_len));
            start += step;
        }

        let last_start = full_length - slice_len;
        if let Some((prev_start, _)) = intervals.last().copied() {
            if prev_start < last_start {
                intervals.push((last_start, full_length));
            }
        }
        intervals
    }

    /// Generate row-major 2D [`SliceWindow`] coordinates covering `image_width x image_height`.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidGeometry`] if image dimensions are zero.
    pub fn get_slice_windows(
        &self,
        image_width: usize,
        image_height: usize,
    ) -> Result<Vec<SliceWindow>> {
        if image_width == 0 || image_height == 0 {
            return Err(AutoCFUError::InvalidGeometry(format!(
                "Image dimensions must be positive, got {image_width}x{image_height}"
            )));
        }

        let x_intervals = Self::generate_1d_intervals(image_width, self.slice_width, self.step_x);
        let y_intervals = Self::generate_1d_intervals(image_height, self.slice_height, self.step_y);

        let mut windows = Vec::with_capacity(x_intervals.len() * y_intervals.len());
        for &(ymin, ymax) in &y_intervals {
            for &(xmin, xmax) in &x_intervals {
                windows.push(SliceWindow {
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                });
            }
        }
        Ok(windows)
    }
}
