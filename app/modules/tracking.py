"""
Multi-object tracking module – SORT (Simple Online and Realtime Tracking).

Assigns **persistent track IDs** to person detections across consecutive
frames so that the downstream LSTM module can build per-identity temporal
sequences of bounding-box features (position, velocity, aspect ratio)
and classify sustained behavioural patterns (e.g. following, encircling,
aggressive approach) rather than analysing each frame in isolation.

    ┌────────────┐      ┌─────────────┐      ┌────────────────┐
    │  YOLOv9    │─────▶│  SORT        │─────▶│  LSTM          │
    │  Detector  │      │  Tracker     │      │  Analyser      │
    │            │      │              │      │                │
    │ [x1,y1,    │      │ [x1,y1,      │      │ Per-track_id   │
    │  x2,y2,    │      │  x2,y2,      │      │ feature seqs ▶ │
    │  conf]     │      │  track_id]   │      │ harassment     │
    └────────────┘      └─────────────┘      │ score          │
                                              └────────────────┘

The ``track_id`` produced here is the key that groups detections of
the *same person* over time, enabling the LSTM to reason about
trajectories rather than instantaneous snapshots.

Implementation
--------------
This module ships a self-contained SORT implementation based on the
original paper by Bewley et al. (2016).  It uses a **linear Kalman
filter** (via ``filterpy``) for state prediction and the **Hungarian
algorithm** (via ``scipy``) for detection-to-track association.
No external ``sort`` pip package is required.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from app.schemas.schemas import BoundingBox, DetectionClass, DetectionResult

logger = logging.getLogger(__name__)


# =====================================================================
# Utility functions
# =====================================================================

def _iou_batch(bb_test: np.ndarray, bb_truth: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bb_test:  (N, 4) array  – ``[x1, y1, x2, y2]``
        bb_truth: (M, 4) array  – ``[x1, y1, x2, y2]``

    Returns:
        (N, M) IoU matrix.
    """
    bb_test = np.expand_dims(bb_test, 1)    # (N, 1, 4)
    bb_truth = np.expand_dims(bb_truth, 0)  # (1, M, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_truth[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_truth[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_truth[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_truth[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (
        (bb_test[..., 2] - bb_test[..., 0])
        * (bb_test[..., 3] - bb_test[..., 1])
    )
    area_truth = (
        (bb_truth[..., 2] - bb_truth[..., 0])
        * (bb_truth[..., 3] - bb_truth[..., 1])
    )

    union = area_test + area_truth - intersection
    return intersection / np.maximum(union, 1e-6)


def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` to ``[cx, cy, s, r]``."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h          # scale (area)
    r = w / max(h, 1e-6)  # aspect ratio
    return np.array([cx, cy, s, r]).reshape((4, 1))


def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, s, r]`` back to ``[x1, y1, x2, y2]``."""
    cx, cy, s, r = x[:4].flatten()
    s = max(s, 0)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ])


# =====================================================================
# Kalman-filter-backed single-object tracker
# =====================================================================

class _KalmanBoxTracker:
    """
    Internal tracker for a single object using a linear Kalman filter
    with a constant-velocity motion model over ``[cx, cy, s, r]``.
    """

    _id_counter: int = 0

    def __init__(self, bbox: np.ndarray) -> None:
        """
        Args:
            bbox: Initial bounding box ``[x1, y1, x2, y2]``.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition (constant velocity)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Covariance matrices
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty on velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialise state
        self.kf.x[:4] = _convert_bbox_to_z(bbox)

        self.time_since_update: int = 0
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0

        _KalmanBoxTracker._id_counter += 1
        self.id: int = _KalmanBoxTracker._id_counter

    def update(self, bbox: np.ndarray) -> None:
        """Update state with an observed bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(_convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """Advance state and return the predicted bounding box."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _convert_x_to_bbox(self.kf.x)

    def get_state(self) -> np.ndarray:
        """Return the current bounding box estimate."""
        return _convert_x_to_bbox(self.kf.x)


# =====================================================================
# Hungarian matching
# =====================================================================

def _associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match detections to existing trackers using the Hungarian algorithm.

    Returns:
        (matched_indices, unmatched_detections, unmatched_trackers)
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty(0, dtype=int),
        )

    iou_matrix = _iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)
        matched_indices = np.column_stack((row_idx, col_idx))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = [
        d for d in range(len(detections)) if d not in matched_indices[:, 0]
    ]
    unmatched_trackers = [
        t for t in range(len(trackers)) if t not in matched_indices[:, 1]
    ]

    # Filter out low-IoU matches
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m)

    matched = np.array(matches).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int)
    return matched, np.array(unmatched_detections), np.array(unmatched_trackers)


# =====================================================================
# Main SORT tracker
# =====================================================================

class SORTTracker:
    """
    SORT multi-object tracker.

    Maintains a set of Kalman-filter-backed track hypotheses and
    associates incoming detections to them frame-by-frame.

    Args:
        max_age: Number of consecutive misses before a track is deleted.
        min_hits: Minimum number of hits before a track is reported.
        iou_threshold: Minimum IoU for a detection–track match.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: list[_KalmanBoxTracker] = []
        self._frame_count: int = 0

        logger.info(
            "SORTTracker initialised (max_age=%d, min_hits=%d, iou=%.2f)",
            max_age, min_hits, iou_threshold,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """
        Reset internal state.  Called during the FastAPI lifespan hook
        so the tracker is ready before the first request.
        """
        self._trackers = []
        self._frame_count = 0
        _KalmanBoxTracker._id_counter = 0
        logger.info("SORT tracker state reset – ready for tracking.")

    # ------------------------------------------------------------------
    # Core update (low-level)
    # ------------------------------------------------------------------

    def update_raw(
        self,
        detections: np.ndarray,
    ) -> np.ndarray:
        """
        Process one frame of detections and return tracked objects.

        Args:
            detections: ``(N, 5)`` array with rows ``[x1, y1, x2, y2, confidence]``.
                        Pass an **empty** ``(0, 5)`` array when no detections exist.

        Returns:
            ``(M, 5)`` array with rows ``[x1, y1, x2, y2, track_id]``.

        Note:
            ``track_id`` is a persistent integer identifier.  The LSTM
            module uses it to accumulate per-person feature sequences
            over time and classify sustained behaviours (harassment,
            stalking, aggression) rather than relying on single-frame
            snapshots.
        """
        self._frame_count += 1

        # ---- Predict new locations for all existing trackers ----
        predicted_boxes = np.zeros((len(self._trackers), 4))
        to_delete: list[int] = []

        for i, trk in enumerate(self._trackers):
            pos = trk.predict()
            predicted_boxes[i] = pos
            if np.any(np.isnan(pos)):
                to_delete.append(i)

        for i in reversed(to_delete):
            self._trackers.pop(i)
        predicted_boxes = np.delete(predicted_boxes, to_delete, axis=0) if to_delete else predicted_boxes

        # ---- Associate detections to trackers ----
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5))

        matched, unmatched_dets, unmatched_trks = _associate_detections_to_trackers(
            detections[:, :4],
            predicted_boxes,
            self.iou_threshold,
        )

        # ---- Update matched trackers with new detections ----
        for det_idx, trk_idx in matched:
            self._trackers[trk_idx].update(detections[det_idx, :4])

        # ---- Create new trackers for unmatched detections ----
        for det_idx in unmatched_dets:
            self._trackers.append(_KalmanBoxTracker(detections[det_idx, :4]))

        # ---- Build output & prune dead tracks ----
        results: list[np.ndarray] = []
        active_trackers: list[_KalmanBoxTracker] = []

        for trk in self._trackers:
            bbox = trk.get_state()

            is_confirmed = (
                trk.time_since_update < 1
                and (trk.hit_streak >= self.min_hits or self._frame_count <= self.min_hits)
            )
            if is_confirmed:
                results.append(np.concatenate([bbox, [trk.id]]))

            if trk.time_since_update <= self.max_age:
                active_trackers.append(trk)

        self._trackers = active_trackers

        if results:
            output = np.stack(results)
            logger.debug(
                "SORT returned %d active track(s) from %d detection(s).",
                len(output), len(detections),
            )
            return output

        logger.debug("SORT returned 0 active tracks.")
        return np.empty((0, 5))

    # ------------------------------------------------------------------
    # Schema-level wrapper (used by the API layer)
    # ------------------------------------------------------------------

    async def update(
        self,
        detections: list[DetectionResult],
    ) -> list[DetectionResult]:
        """
        High-level async wrapper consumed by the ``/analyze`` endpoint.

        Converts ``DetectionResult`` objects to the ``(N, 5)`` array
        expected by :meth:`update_raw`, runs the tracker, and maps
        the resulting ``track_id`` values back onto the schema objects.

        When detections cannot be matched (e.g. they were lost by the
        tracker), they are dropped from the returned list.
        """
        if not detections:
            logger.debug("No detections to track – returning empty list.")
            return []

        # Build (N, 5) array: [x1, y1, x2, y2, confidence]
        det_array = np.array(
            [
                [
                    d.bounding_box.x_min,
                    d.bounding_box.y_min,
                    d.bounding_box.x_max,
                    d.bounding_box.y_max,
                    d.confidence,
                ]
                for d in detections
            ],
            dtype=np.float64,
        )

        tracked = self.update_raw(det_array)  # (M, 5): [x1,y1,x2,y2,track_id]

        if len(tracked) == 0:
            return []

        # Map tracked boxes back to DetectionResult objects by IoU overlap
        tracked_results: list[DetectionResult] = []
        used_det_indices: set[int] = set()

        for trk_row in tracked:
            tx1, ty1, tx2, ty2, tid = trk_row
            best_iou = 0.0
            best_idx = -1

            for idx, d in enumerate(detections):
                if idx in used_det_indices:
                    continue
                diou = float(
                    _iou_batch(
                        np.array([[tx1, ty1, tx2, ty2]]),
                        np.array([[
                            d.bounding_box.x_min,
                            d.bounding_box.y_min,
                            d.bounding_box.x_max,
                            d.bounding_box.y_max,
                        ]]),
                    )[0, 0]
                )
                if diou > best_iou:
                    best_iou = diou
                    best_idx = idx

            if best_idx >= 0:
                used_det_indices.add(best_idx)
                det = detections[best_idx]
                # Update the bounding box to the Kalman-smoothed position
                det.bounding_box = BoundingBox(
                    x_min=round(float(tx1), 2),
                    y_min=round(float(ty1), 2),
                    x_max=round(float(tx2), 2),
                    y_max=round(float(ty2), 2),
                )
                det.track_id = int(tid)
                tracked_results.append(det)

        logger.debug(
            "SORT schema wrapper returning %d tracked result(s).",
            len(tracked_results),
        )
        return tracked_results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
tracker = SORTTracker()

