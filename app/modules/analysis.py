"""
Temporal analysis module – Pair-based LSTM harassment classification.

=======================================================================
  KEY DIFFERENTIATOR: PAIR-BASED ANALYSIS
=======================================================================
Most surveillance systems classify **individuals** (e.g. "person is
running", "person is loitering").  Our system is fundamentally different:
we analyse every **(Person A, Person B) pair** and ask "is A harassing B?"

This pair-based approach captures **interpersonal dynamics** that single-
person classifiers miss:

  • Interpersonal distance shrinking over time  (approaching / cornering)
  • Hand intrusion into another person's torso region  (pushing / grabbing)
  • Arm movement synchronisation  (one person swinging while the other
    recoils — asymmetric motion is a strong harassment signal)

The LSTM receives a temporal *sequence* of these pair features (buffered
over a sliding window of N frames) and outputs a harassment probability.
This temporal view is critical: a single frame of two people close
together is ambiguous, but 30 frames of one person repeatedly invading
another's space while the other retreats is unambiguous.

Pipeline position
-----------------
    YOLOv9 (detect persons)
        → SORT (assign persistent track_ids)
            → MediaPipe (extract skeleton keypoints per track)
                → **LSTMAnalyser** (pair-based feature engineering
                   + temporal LSTM classification)
                    → AlertPayload if harassment_probability > 85 %
=======================================================================
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime
from itertools import combinations
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from app.core.config import settings
from app.schemas.schemas import (
    AlertPayload,
    AlertSeverity,
    DetectionClass,
    DetectionResult,
)

logger = logging.getLogger(__name__)

# =====================================================================
# MediaPipe Pose landmark indices (subset used for feature engineering)
# Full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# =====================================================================

# Torso / hip region
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

# Hand / wrist keypoints
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14

# For centroid computation (upper body + hips)
_CENTROID_INDICES = [_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]

# For arm velocity (elbows + wrists)
_ARM_INDICES = [_LEFT_ELBOW, _RIGHT_ELBOW, _LEFT_WRIST, _RIGHT_WRIST]

# Number of engineered features per pair per frame
NUM_PAIR_FEATURES = 3  # interpersonal_distance, hand_intrusion_score, arm_sync_score

# Default sliding window length (frames)
DEFAULT_SEQUENCE_LENGTH = 30


# =====================================================================
# PyTorch LSTM model definition
# =====================================================================

class HarassmentLSTM(nn.Module):
    """
    Lightweight LSTM classifier for pair-based harassment detection.

    Architecture
    ------------
    Input  → LSTM (hidden_size, num_layers)
           → Dropout
           → Linear (hidden_size → 1)
           → Sigmoid

    The model receives a sequence of shape ``(seq_len, num_features)``
    for a single (Person A, Person B) pair and outputs a scalar
    harassment probability in ``[0, 1]``.
    """

    def __init__(
        self,
        input_size: int = NUM_PAIR_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, seq_len, input_size)`` tensor.

        Returns:
            ``(batch, 1)`` tensor of harassment probabilities.
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time-step's hidden state
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return self.sigmoid(out)


# =====================================================================
# Feature engineering (pair-based)
# =====================================================================

def compute_harassment_features(
    skeleton_a: np.ndarray,
    skeleton_b: np.ndarray,
    prev_skeleton_a: Optional[np.ndarray] = None,
    prev_skeleton_b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Engineer harassment-indicative features from two skeletons.
    The logic is **pair-symmetric**, meaning (A, B) yields identical features
    as (B, A).

    Args:
        skeleton_a: ``(33, 3)`` or ``(33, 2)`` array of MediaPipe Pose
                    landmarks for Person A.
        skeleton_b: Same format for Person B.
        prev_skeleton_a: Previous frame's skeleton for Person A (optional).
        prev_skeleton_b: Previous frame's skeleton for Person B (optional).

    Returns:
        ``(3,)`` feature vector:
            [0] interpersonal_distance  – normalised Euclidean distance.
            [1] hand_intrusion_score    – max intrusion of A's hands into B's torso or B's hands into A's.
            [2] arm_sync_score          – cosine similarity of arm movement vectors.
            
    NOTE: All features are scaled to a [0, 1] range before inference to maintain model stability.
    """
    # Use only (x, y) for 2-D spatial reasoning
    kps_a = skeleton_a[:, :2].astype(np.float64)
    kps_b = skeleton_b[:, :2].astype(np.float64)

    # ------------------------------------------------------------------
    # 1. Normalized Interpersonal Distance (Symmetric)
    # ------------------------------------------------------------------
    # Euclidean distance between centroids, normalized by the bounding boxes' heights.
    centroid_a = kps_a[_CENTROID_INDICES].mean(axis=0)
    centroid_b = kps_b[_CENTROID_INDICES].mean(axis=0)
    raw_distance = np.linalg.norm(centroid_a - centroid_b)

    height_a = kps_a[:, 1].max() - kps_a[:, 1].min()
    height_b = kps_b[:, 1].max() - kps_b[:, 1].min()
    avg_height = max((height_a + height_b) / 2.0, 1e-6)

    # Normalize and clip to [0, 1] range
    interpersonal_distance = min(1.0, raw_distance / avg_height)

    # ------------------------------------------------------------------
    # 2. Hand Intrusion Score (Symmetric: A into B, or B into A)
    # ------------------------------------------------------------------
    # Center-point-in-box check between hand keypoints and torso bounding region.
    def _compute_intrusion(actor_kps, target_kps) -> float:
        torso = target_kps[[_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]]
        t_min, t_max = torso.min(axis=0), torso.max(axis=0)
        wrists = actor_kps[[_LEFT_WRIST, _RIGHT_WRIST]]
        
        score = 0.0
        for w in wrists:
            # Simple bounding box overlap check (center-point-in-box)
            if t_min[0] <= w[0] <= t_max[0] and t_min[1] <= w[1] <= t_max[1]:
                score = 1.0
        return score

    intrusion_a_into_b = _compute_intrusion(kps_a, kps_b)
    intrusion_b_into_a = _compute_intrusion(kps_b, kps_a)
    hand_intrusion_score = max(intrusion_a_into_b, intrusion_b_into_a)

    # ------------------------------------------------------------------
    # 3. Arm Sync Score (Cosine similarity of movement vectors)
    # ------------------------------------------------------------------
    arm_sync_score = 0.5  # Neutral default (0 orthogonal) scaled to [0, 1]
    if prev_skeleton_a is not None and prev_skeleton_b is not None:
        prev_kps_a = prev_skeleton_a[:, :2].astype(np.float64)
        prev_kps_b = prev_skeleton_b[:, :2].astype(np.float64)
        
        # Combined movement vectors for the arms
        move_a = (kps_a[_LEFT_WRIST] - prev_kps_a[_LEFT_WRIST]) + (kps_a[_RIGHT_WRIST] - prev_kps_a[_RIGHT_WRIST])
        move_b = (kps_b[_LEFT_WRIST] - prev_kps_b[_LEFT_WRIST]) + (kps_b[_RIGHT_WRIST] - prev_kps_b[_RIGHT_WRIST])
        
        norm_a = np.linalg.norm(move_a)
        norm_b = np.linalg.norm(move_b)
        
        if norm_a > 1e-6 and norm_b > 1e-6:
            # Cosine similarity yields [-1.0, 1.0]
            cos_sim = np.dot(move_a, move_b) / (norm_a * norm_b)
            # Scale to [0, 1]
            arm_sync_score = (cos_sim + 1.0) / 2.0

    return np.array(
        [interpersonal_distance, hand_intrusion_score, arm_sync_score],
        dtype=np.float32,
    )


# =====================================================================
# Main analyser class
# =====================================================================

class LSTMAnalyser:
    """
    Pair-based temporal harassment analyser.

    For every unique ``(track_id_A, track_id_B)`` pair observed by the
    SORT tracker, this class:

    1. Accepts skeleton keypoints and engineers pair features via
       :func:`compute_harassment_features`.
    2. Buffers those features in a per-pair sliding window.
    3. When the buffer is full, runs the :class:`HarassmentLSTM` to
       produce a harassment probability.
    4. If the probability exceeds the configured threshold, generates
       an :class:`AlertPayload`.
    """

    def __init__(
        self,
        onnx_path: str = settings.ONNX_WEIGHTS_PATH,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        self.onnx_path = onnx_path
        self.sequence_length = sequence_length
        self._session: Optional[ort.InferenceSession] = None

        # Sequence Buffer: collections.deque with maxlen=30 to hold the sequences per pair
        self._pair_buffers: dict[tuple[int, int], deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.sequence_length)
        )

        # Cache of the most recent skeleton keypoints per track_id,
        # used to compute arm velocity between consecutive frames.
        self._prev_skeletons: dict[int, np.ndarray] = {}

        logger.info(
            "LSTMAnalyser initialised (onnx_path=%s, seq_len=%d)",
            self.onnx_path,
            self.sequence_length,
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    async def load_model(self) -> None:
        """
        Load the trained LSTM ONNX weights from disk.
        """
        try:
            # Load the session using CPUExecutionProvider for lightweight CPU inference
            self._session = ort.InferenceSession(
                self.onnx_path,
                providers=["CPUExecutionProvider"]
            )
            logger.info(
                "LSTM ONNX model loaded successfully from %s.",
                self.onnx_path,
            )
        except Exception:
            logger.warning(
                "LSTM ONNX session failed to load from %s – checking if export is needed.",
                self.onnx_path,
            )

    # ------------------------------------------------------------------
    # Sequence buffering
    # ------------------------------------------------------------------

    def update_track_sequence(
        self,
        track_id_pair: tuple[int, int],
        features: np.ndarray,
    ) -> int:
        """
        Append a feature vector to the sliding window buffer for the
        given track-ID pair.
        """
        # Canonicalise pair order for Symmetry
        pair_key = (min(track_id_pair), max(track_id_pair))
        self._pair_buffers[pair_key].append(features.copy())

        buf_len = len(self._pair_buffers[pair_key])
        return buf_len

    def get_pair_sequence(
        self,
        track_id_pair: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Retrieve the full feature sequence for a pair if the buffer
        has reached ``sequence_length``.
        """
        pair_key = (min(track_id_pair), max(track_id_pair))
        buf = self._pair_buffers.get(pair_key)

        if buf is None or len(buf) < self.sequence_length:
            return None

        return np.array(buf, dtype=np.float32)

    # ------------------------------------------------------------------
    # LSTM inference
    # ------------------------------------------------------------------

    async def analyze(
        self,
        track_id_pair: tuple[int, int],
    ) -> float:
        """
        Run LSTM inference on the buffered feature sequence for a pair.
        """
        if self._session is None:
            await self.load_model()
            if self._session is None:
                logger.error("ONNX model session not loaded.")
                return 0.0

        sequence = self.get_pair_sequence(track_id_pair)
        if sequence is None:
            return 0.0

        try:
            # Properly reshape the 30-frame window into (1, 30, 3) before passing to the ONNX session
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            
            outputs = self._session.run([output_name], {input_name: input_data})
            probability = float(outputs[0][0][0])

            probability = float(np.clip(probability, 0.0, 1.0))
            return round(probability, 4)

        except Exception:
            logger.exception("ONNX inference failed for pair %s.", track_id_pair)
            return 0.0

    # ------------------------------------------------------------------
    # High-level API (called by the /analyze endpoint)
    # ------------------------------------------------------------------

    async def compute_harassment_score(
        self,
        detections: list[DetectionResult],
        skeletons: Optional[dict[int, np.ndarray]] = None,
    ) -> float:
        """
        Compute the aggregate harassment score across all tracked pairs
        in the current frame.
        """
        if self._session is None:
            await self.load_model()

        if not detections:
            return 0.0

        tracked = [d for d in detections if d.track_id is not None]
        track_ids = [d.track_id for d in tracked]

        if len(track_ids) < 2:
            return 0.0

        max_score = 0.0

        for id_a, id_b in combinations(sorted(set(track_ids)), 2):
            pair_key = (id_a, id_b)

            if skeletons and id_a in skeletons and id_b in skeletons:
                prev_a = self._prev_skeletons.get(id_a)
                prev_b = self._prev_skeletons.get(id_b)
                
                features = compute_harassment_features(
                    skeletons[id_a], skeletons[id_b],
                    prev_a, prev_b
                )
            else:
                features = self._bbox_fallback_features(id_a, id_b, tracked)

            self.update_track_sequence(pair_key, features)

            score = await self.analyze(pair_key)
            if score > max_score:
                max_score = score

        # Update cached skeletons for next frame's movement vector calculations
        if skeletons:
            self._prev_skeletons = {k: v.copy() for k, v in skeletons.items()}

        return max_score

    async def maybe_generate_alert(
        self,
        frame_id: str,
        camera_id: Optional[str],
        detections: list[DetectionResult],
        harassment_score: float,
    ) -> Optional[AlertPayload]:
        threshold = settings.HARASSMENT_ALERT_THRESHOLD

        if harassment_score < threshold:
            return None

        severity = self._classify_severity(harassment_score)

        alert = AlertPayload(
            alert_id=f"alert_{uuid.uuid4().hex[:12]}",
            frame_id=frame_id,
            camera_id=camera_id,
            severity=severity,
            harassment_confidence=harassment_score,
            detections=detections,
            message=(
                f"⚠️ Harassment detected with {harassment_score:.0%} confidence "
                f"on camera {camera_id or 'unknown'}."
            ),
        )

        logger.warning(
            "ALERT triggered: %s (score=%.4f)", alert.alert_id, harassment_score,
        )
        return alert

    @staticmethod
    def _classify_severity(score: float) -> AlertSeverity:
        if score >= 0.95:
            return AlertSeverity.CRITICAL
        if score >= 0.90:
            return AlertSeverity.HIGH
        if score >= 0.85:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

    @staticmethod
    def _bbox_fallback_features(
        id_a: int,
        id_b: int,
        tracked: list[DetectionResult],
    ) -> np.ndarray:
        det_a = next((d for d in tracked if d.track_id == id_a), None)
        det_b = next((d for d in tracked if d.track_id == id_b), None)

        if det_a is None or det_b is None:
            return np.zeros(NUM_PAIR_FEATURES, dtype=np.float32)

        cx_a = (det_a.bounding_box.x_min + det_a.bounding_box.x_max) / 2
        cy_a = (det_a.bounding_box.y_min + det_a.bounding_box.y_max) / 2
        cx_b = (det_b.bounding_box.x_min + det_b.bounding_box.x_max) / 2
        cy_b = (det_b.bounding_box.y_min + det_b.bounding_box.y_max) / 2

        height_a = det_a.bounding_box.y_max - det_a.bounding_box.y_min
        height_b = det_b.bounding_box.y_max - det_b.bounding_box.y_min
        avg_height = max((height_a + height_b) / 2.0, 1e-6)
        
        raw_dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
        interpersonal_distance = min(1.0, raw_dist / avg_height)

        overlap_x = max(
            0.0,
            min(det_a.bounding_box.x_max, det_b.bounding_box.x_max)
            - max(det_a.bounding_box.x_min, det_b.bounding_box.x_min),
        )
        overlap_y = max(
            0.0,
            min(det_a.bounding_box.y_max, det_b.bounding_box.y_max)
            - max(det_a.bounding_box.y_min, det_b.bounding_box.y_min),
        )
        overlap_area = overlap_x * overlap_y
        area_b = (
            (det_b.bounding_box.x_max - det_b.bounding_box.x_min)
            * (det_b.bounding_box.y_max - det_b.bounding_box.y_min)
        )
        hand_intrusion_score = min(1.0, overlap_area / max(area_b, 1e-6))

        arm_sync_score = 0.5

        return np.array(
            [interpersonal_distance, hand_intrusion_score, arm_sync_score],
            dtype=np.float32,
        )

    def cleanup_stale_tracks(self, active_track_ids: set[int]) -> int:
        stale_keys = [
            k for k in self._pair_buffers
            if k[0] not in active_track_ids or k[1] not in active_track_ids
        ]
        for k in stale_keys:
            del self._pair_buffers[k]

        return len(stale_keys)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
analyser = LSTMAnalyser()
