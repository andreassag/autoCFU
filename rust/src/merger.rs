//! Detection geometry, radial/polygon reconstruction, and GreedyNMM / NMS deduplication.

use std::cmp::Ordering;

use serde::Serialize;

use crate::config::{MatchMetric, MergerType};
use crate::error::{AutoCFUError, Result};

/// Reconstruct 2D star-convex polygon vertices from centroid `(cx, cy)` and equiangular rays.
///
/// Angles follow `theta_k = 2 * pi * k / N` with `x_k = cx + r_k * cos(theta_k)` and
/// `y_k = cy + r_k * sin(theta_k)`.
///
/// # Errors
/// Returns [`AutoCFUError::InvalidGeometry`] when `rays` is empty.
pub fn reconstruct_star_polygon(cx: f64, cy: f64, rays: &[f64]) -> Result<Vec<[f64; 2]>> {
    if rays.is_empty() {
        return Err(AutoCFUError::InvalidGeometry(
            "Cannot reconstruct star polygon from an empty ray slice".to_string(),
        ));
    }
    let two_pi_over_n = (2.0 * std::f64::consts::PI) / (rays.len() as f64);
    let vertices = rays
        .iter()
        .enumerate()
        .map(|(idx, &r)| {
            let theta = (idx as f64) * two_pi_over_n;
            [cx + r * theta.cos(), cy + r * theta.sin()]
        })
        .collect();
    Ok(vertices)
}

fn polygon_bounds(polygon: &[[f64; 2]]) -> Result<[f64; 4]> {
    let first = polygon
        .first()
        .ok_or_else(|| AutoCFUError::InvalidGeometry("Empty polygon has no bounds".to_string()))?;
    let mut xmin = first[0];
    let mut ymin = first[1];
    let mut xmax = first[0];
    let mut ymax = first[1];
    for pt in polygon.iter().skip(1) {
        xmin = xmin.min(pt[0]);
        ymin = ymin.min(pt[1]);
        xmax = xmax.max(pt[0]);
        ymax = ymax.max(pt[1]);
    }
    Ok([xmin, ymin, xmax, ymax])
}

/// Unified detected or segmented microbial colony in pixel coordinates.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Detection {
    /// Bounding box `[xmin, ymin, xmax, ymax]` in pixels.
    pub bbox: [f64; 4],
    /// Optical or geometric center `[cx, cy]` in pixels.
    pub center_xy: [f64; 2],
    /// Estimated colony radius in pixels (mean ray length for StarDist).
    pub radius_px: f64,
    /// Model confidence score in `[0.0, 1.0]`.
    pub score: f64,
    /// Predicted foreground class index.
    pub class_id: i64,
    /// Human-readable class label.
    pub label: String,
    /// Radial distances along `N` equiangular rays when emitted by StarDist.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rays: Option<Vec<f64>>,
    /// Star-convex polygon vertices `[x, y]` in global pixel coordinates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<[f64; 2]>>,
}

impl Detection {
    /// Construct a detection from a bounding box `[xmin, ymin, xmax, ymax]`.
    #[must_use]
    pub fn from_bbox(bbox: [f64; 4], score: f64, class_id: i64, label: impl Into<String>) -> Self {
        let cx = 0.5 * (bbox[0] + bbox[2]);
        let cy = 0.5 * (bbox[1] + bbox[3]);
        let w = (bbox[2] - bbox[0]).max(0.0);
        let h = (bbox[3] - bbox[1]).max(0.0);
        Self {
            bbox,
            center_xy: [cx, cy],
            radius_px: 0.25 * (w + h),
            score,
            class_id,
            label: label.into(),
            rays: None,
            polygon: None,
        }
    }

    /// Construct a radial CenterNet detection from center `(cx, cy)` and radius `r`.
    #[must_use]
    pub fn from_center_radius(
        center_xy: [f64; 2],
        radius_px: f64,
        score: f64,
        class_id: i64,
        label: impl Into<String>,
    ) -> Self {
        let r = radius_px.max(0.0);
        Self {
            bbox: [
                center_xy[0] - r,
                center_xy[1] - r,
                center_xy[0] + r,
                center_xy[1] + r,
            ],
            center_xy,
            radius_px: r,
            score,
            class_id,
            label: label.into(),
            rays: None,
            polygon: None,
        }
    }

    /// Construct a 2D StarDist detection from centroid `(cx, cy)` and radial vector `rays`.
    ///
    /// # Errors
    /// Returns [`AutoCFUError::InvalidGeometry`] if `rays` is empty.
    pub fn from_stardist(
        center_xy: [f64; 2],
        rays: Vec<f64>,
        polygon_override: Option<Vec<[f64; 2]>>,
        score: f64,
        class_id: i64,
        label: impl Into<String>,
    ) -> Result<Self> {
        let polygon = match polygon_override {
            Some(poly) if !poly.is_empty() => poly,
            _ => reconstruct_star_polygon(center_xy[0], center_xy[1], &rays)?,
        };
        let bbox = polygon_bounds(&polygon)?;
        let radius_px = if rays.is_empty() {
            0.25 * ((bbox[2] - bbox[0]).max(0.0) + (bbox[3] - bbox[1]).max(0.0))
        } else {
            rays.iter().sum::<f64>() / (rays.len() as f64)
        };
        Ok(Self {
            bbox,
            center_xy,
            radius_px,
            score,
            class_id,
            label: label.into(),
            rays: Some(rays),
            polygon: Some(polygon),
        })
    }

    /// Bounding-box width in pixels.
    #[must_use]
    pub fn width(&self) -> f64 {
        (self.bbox[2] - self.bbox[0]).max(0.0)
    }

    /// Bounding-box height in pixels.
    #[must_use]
    pub fn height(&self) -> f64 {
        (self.bbox[3] - self.bbox[1]).max(0.0)
    }

    /// Translate bounding box, center, and polygon coordinates by `(dx, dy)` pixels.
    #[must_use]
    pub fn translated(&self, dx: f64, dy: f64) -> Self {
        let polygon = self
            .polygon
            .as_ref()
            .map(|pts| pts.iter().map(|&[x, y]| [x + dx, y + dy]).collect());
        Self {
            bbox: [
                self.bbox[0] + dx,
                self.bbox[1] + dy,
                self.bbox[2] + dx,
                self.bbox[3] + dy,
            ],
            center_xy: [self.center_xy[0] + dx, self.center_xy[1] + dy],
            radius_px: self.radius_px,
            score: self.score,
            class_id: self.class_id,
            label: self.label.clone(),
            rays: self.rays.clone(),
            polygon,
        }
    }
}

fn box_intersection_area(box_a: [f64; 4], box_b: [f64; 4]) -> f64 {
    let ix1 = box_a[0].max(box_b[0]);
    let iy1 = box_a[1].max(box_b[1]);
    let ix2 = box_a[2].min(box_b[2]);
    let iy2 = box_a[3].min(box_b[3]);
    (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0)
}

fn box_area(b: [f64; 4]) -> f64 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// Compute Intersection over Union (IoU) between two `[xmin, ymin, xmax, ymax]` boxes.
#[must_use]
pub fn calculate_iou(box_a: [f64; 4], box_b: [f64; 4]) -> f64 {
    let inter = box_intersection_area(box_a, box_b);
    if inter <= 0.0 {
        return 0.0;
    }
    let union = box_area(box_a) + box_area(box_b) - inter;
    if union <= 0.0 {
        return 0.0;
    }
    inter / union
}

/// Compute Intersection over Smaller Area (IoS) between two boxes.
#[must_use]
pub fn calculate_ios(box_a: [f64; 4], box_b: [f64; 4]) -> f64 {
    let inter = box_intersection_area(box_a, box_b);
    if inter <= 0.0 {
        return 0.0;
    }
    let smaller = box_area(box_a).min(box_area(box_b));
    if smaller <= 0.0 {
        return 0.0;
    }
    inter / smaller
}

/// Compute Euclidean center distance divided by `min(det_a.radius_px, det_b.radius_px)`.
#[must_use]
pub fn calculate_center_distance_ratio(det_a: &Detection, det_b: &Detection) -> f64 {
    let dx = det_a.center_xy[0] - det_b.center_xy[0];
    let dy = det_a.center_xy[1] - det_b.center_xy[1];
    let dist = dx.hypot(dy);
    let min_r = det_a.radius_px.min(det_b.radius_px);
    if min_r <= 0.0 {
        if dist == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        dist / min_r
    }
}

/// Evaluate whether two detections match under the given metric and threshold.
#[must_use]
pub fn detections_match(
    det_a: &Detection,
    det_b: &Detection,
    match_threshold: f64,
    match_metric: MatchMetric,
) -> bool {
    match match_metric {
        MatchMetric::CenterDist => calculate_center_distance_ratio(det_a, det_b) <= match_threshold,
        MatchMetric::Ios => calculate_ios(det_a.bbox, det_b.bbox) >= match_threshold,
        MatchMetric::Iou => calculate_iou(det_a.bbox, det_b.bbox) >= match_threshold,
    }
}

fn sort_detections_desc(detections: &[Detection]) -> Vec<Detection> {
    let mut indexed: Vec<(usize, &Detection)> = detections.iter().enumerate().collect();
    indexed.sort_by(|(idx_a, det_a), (idx_b, det_b)| {
        det_b
            .score
            .total_cmp(&det_a.score)
            .then_with(|| idx_a.cmp(idx_b))
    });
    indexed.into_iter().map(|(_, det)| det.clone()).collect()
}

fn merge_cluster(cluster: &[Detection]) -> Result<Detection> {
    let anchor = cluster.first().ok_or_else(|| {
        AutoCFUError::InvalidGeometry("Cannot merge empty detection cluster".to_string())
    })?;
    if cluster.len() == 1 {
        return Ok(anchor.clone());
    }

    let raw_weights: Vec<f64> = cluster.iter().map(|d| d.score.max(0.0)).collect();
    let raw_sum: f64 = raw_weights.iter().sum();
    let (weights, weight_sum) = if raw_sum <= 0.0 {
        (vec![1.0; cluster.len()], cluster.len() as f64)
    } else {
        (raw_weights, raw_sum)
    };

    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut radius = 0.0;
    let mut max_score = f64::NEG_INFINITY;
    for (&w, det) in weights.iter().zip(cluster.iter()) {
        cx += w * det.center_xy[0];
        cy += w * det.center_xy[1];
        radius += w * det.radius_px;
        if det.score > max_score {
            max_score = det.score;
        }
    }
    cx /= weight_sum;
    cy /= weight_sum;
    radius /= weight_sum;

    if let Some(first_rays) = &anchor.rays {
        let num_rays = first_rays.len();
        if num_rays > 0
            && cluster
                .iter()
                .all(|d| d.rays.as_ref().is_some_and(|r| r.len() == num_rays))
        {
            let mut merged_rays = vec![0.0; num_rays];
            for (&w, det) in weights.iter().zip(cluster.iter()) {
                if let Some(rays) = &det.rays {
                    for (acc, &val) in merged_rays.iter_mut().zip(rays.iter()) {
                        *acc += w * val;
                    }
                }
            }
            for acc in &mut merged_rays {
                *acc /= weight_sum;
            }
            let polygon = reconstruct_star_polygon(cx, cy, &merged_rays)?;
            let bbox = polygon_bounds(&polygon)?;
            let mean_r = merged_rays.iter().sum::<f64>() / (num_rays as f64);
            return Ok(Detection {
                bbox,
                center_xy: [cx, cy],
                radius_px: mean_r,
                score: max_score,
                class_id: anchor.class_id,
                label: anchor.label.clone(),
                rays: Some(merged_rays),
                polygon: Some(polygon),
            });
        }
    }

    let mut merged_bbox = [0.0; 4];
    for (&w, det) in weights.iter().zip(cluster.iter()) {
        for (acc, &coord) in merged_bbox.iter_mut().zip(det.bbox.iter()) {
            *acc += w * coord;
        }
    }
    for coord in &mut merged_bbox {
        *coord /= weight_sum;
    }

    Ok(Detection {
        bbox: merged_bbox,
        center_xy: [cx, cy],
        radius_px: radius,
        score: max_score,
        class_id: anchor.class_id,
        label: anchor.label.clone(),
        rays: None,
        polygon: None,
    })
}

/// Greedy Non-Maximum Merging across overlapping tile seams.
#[derive(Debug, Clone, Copy)]
pub struct GreedyNMM {
    /// Overlap or center-distance ratio threshold for clustering.
    pub match_threshold: f64,
    /// Metric used to match detections.
    pub match_metric: MatchMetric,
}

impl GreedyNMM {
    /// Create a new [`GreedyNMM`] instance.
    #[must_use]
    pub const fn new(match_threshold: f64, match_metric: MatchMetric) -> Self {
        Self {
            match_threshold,
            match_metric,
        }
    }

    /// Merge overlapping detections via confidence-weighted centroid/radius/ray averaging.
    ///
    /// # Errors
    /// Propagates polygon reconstruction errors if malformed StarDist rays are provided.
    pub fn merge(&self, detections: &[Detection]) -> Result<Vec<Detection>> {
        if detections.is_empty() {
            return Ok(Vec::new());
        }

        let ordered = sort_detections_desc(detections);
        let mut consumed = vec![false; ordered.len()];
        let mut merged = Vec::new();

        for i in 0..ordered.len() {
            if consumed[i] {
                continue;
            }
            consumed[i] = true;
            let anchor = &ordered[i];
            let mut cluster = vec![anchor.clone()];
            for j in (i + 1)..ordered.len() {
                if consumed[j] {
                    continue;
                }
                let cand = &ordered[j];
                if cand.class_id != anchor.class_id {
                    continue;
                }
                if detections_match(anchor, cand, self.match_threshold, self.match_metric) {
                    consumed[j] = true;
                    cluster.push(cand.clone());
                }
            }
            merged.push(merge_cluster(&cluster)?);
        }

        Ok(merged)
    }
}

/// Standard Non-Maximum Suppression supporting `center_dist`, `ios`, and `iou`.
#[derive(Debug, Clone, Copy)]
pub struct NMS {
    /// Overlap or center-distance threshold for suppressing lower-scoring candidates.
    pub match_threshold: f64,
    /// Metric used to decide overlap.
    pub match_metric: MatchMetric,
}

impl NMS {
    /// Create a new [`NMS`] instance.
    #[must_use]
    pub const fn new(match_threshold: f64, match_metric: MatchMetric) -> Self {
        Self {
            match_threshold,
            match_metric,
        }
    }

    /// Suppress lower-scoring overlapping detections.
    #[must_use]
    pub fn merge(&self, detections: &[Detection]) -> Vec<Detection> {
        if detections.is_empty() {
            return Vec::new();
        }
        let ordered = sort_detections_desc(detections);
        let mut suppressed = vec![false; ordered.len()];
        let mut kept = Vec::new();

        for i in 0..ordered.len() {
            if suppressed[i] {
                continue;
            }
            let anchor = &ordered[i];
            kept.push(anchor.clone());
            for j in (i + 1)..ordered.len() {
                if suppressed[j] {
                    continue;
                }
                let cand = &ordered[j];
                if cand.class_id != anchor.class_id {
                    continue;
                }
                if detections_match(anchor, cand, self.match_threshold, self.match_metric) {
                    suppressed[j] = true;
                }
            }
        }
        kept
    }
}

/// Apply the configured merging strategy to a slice of detections.
///
/// # Errors
/// Propagates geometry errors from [`GreedyNMM::merge`].
pub fn apply_merger(
    merger_type: MergerType,
    match_threshold: f64,
    match_metric: MatchMetric,
    detections: &[Detection],
) -> Result<Vec<Detection>> {
    match merger_type {
        MergerType::GreedyNmm => GreedyNMM::new(match_threshold, match_metric).merge(detections),
        MergerType::Nms => Ok(NMS::new(match_threshold, match_metric).merge(detections)),
        MergerType::None => {
            let mut copy = detections.to_vec();
            copy.sort_by(|a, b| b.score.total_cmp(&a.score).then(Ordering::Equal));
            Ok(copy)
        }
    }
}
