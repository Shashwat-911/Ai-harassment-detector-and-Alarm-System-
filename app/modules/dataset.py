import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

class HarassmentDataset(Dataset):
    """
    PyTorch Dataset that loads synthetic coordinates (.npy files) of shape (30, 2, 2)
    and maps them to engineered feature sequences of shape (30, 3) to match the
    inputs expected by HarassmentLSTM.
    
    Features computed:
      1. interpersonal_distance: Euclidean distance normalized by standard height.
      2. hand_intrusion_score: Simulates proximity-based spatial invasion.
      3. arm_sync_score: Cosine similarity of frame-to-frame movement vectors.
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to the synthetic data folder (e.g. app/data/synthetic)
        """
        self.data_dir = data_dir
        self.samples = []
        
        # Support structures where data_dir is the 'synthetic' directory, or the parent 'data' directory
        blocking_dir = os.path.join(data_dir, "blocking")
        if not os.path.exists(blocking_dir):
            blocking_dir = os.path.join(data_dir, "synthetic", "blocking")
            
        pursuing_dir = os.path.join(data_dir, "pursuing")
        if not os.path.exists(pursuing_dir):
            pursuing_dir = os.path.join(data_dir, "synthetic", "pursuing")

        # Load blocking samples (label 1)
        if os.path.exists(blocking_dir):
            blocking_files = glob.glob(os.path.join(blocking_dir, "**", "*.npy"), recursive=True)
            for f in blocking_files:
                self.samples.append((f, 1.0))
                
        # Load pursuing samples (label 1)
        if os.path.exists(pursuing_dir):
            pursuing_files = glob.glob(os.path.join(pursuing_dir, "**", "*.npy"), recursive=True)
            for f in pursuing_files:
                self.samples.append((f, 1.0))
                
        if not self.samples:
            raise FileNotFoundError(
                f"No synthetic .npy datasets found in '{blocking_dir}' or '{pursuing_dir}'"
            )

    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path, label = self.samples[idx]
        
        # Load coordinate sequence: shape (30, 2, 2)
        raw_coords = np.load(file_path)
        
        # Extract features: shape (30, 3)
        features = self._extract_features(raw_coords)
        
        # Convert to torch.float32 tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)  # shape (1,) for BCELoss compatibility
        
        return features_tensor, label_tensor
        
    def _extract_features(self, raw_coords: np.ndarray) -> np.ndarray:
        """
        Maps (30, 2, 2) coordinate sequences to (30, 3) feature sequences.
        """
        seq_len = raw_coords.shape[0]
        features = np.zeros((seq_len, 3), dtype=np.float32)
        
        for t in range(seq_len):
            pos_a = raw_coords[t, 0]  # shape (2,)
            pos_b = raw_coords[t, 1]  # shape (2,)
            
            # Map normalized coordinates back to standard 640x480 pixel space
            pos_a_px = pos_a * np.array([640.0, 480.0])
            pos_b_px = pos_b * np.array([640.0, 480.0])
            
            # 1. Interpersonal Distance (Symmetric)
            raw_dist = np.linalg.norm(pos_a_px - pos_b_px)
            # Normalize by standard person height (approx. 150px)
            interpersonal_distance = min(1.0, raw_dist / 150.0)
            
            # 2. Hand Intrusion Score (Symmetric approximation)
            # Simulated based on spatial proximity (distance < 100px)
            hand_intrusion_score = max(0.0, 1.0 - (raw_dist / 100.0))
            
            # 3. Arm Sync Score (Cosine similarity of movement)
            arm_sync_score = 0.5  # default neutral
            
            if t > 0:
                prev_a_px = raw_coords[t - 1, 0] * np.array([640.0, 480.0])
                prev_b_px = raw_coords[t - 1, 1] * np.array([640.0, 480.0])
                
                move_a = pos_a_px - prev_a_px
                move_b = pos_b_px - prev_b_px
                
                norm_a = np.linalg.norm(move_a)
                norm_b = np.linalg.norm(move_b)
                
                if norm_a > 1e-6 and norm_b > 1e-6:
                    cos_sim = np.dot(move_a, move_b) / (norm_a * norm_b)
                    # Scale from [-1.0, 1.0] to [0.0, 1.0]
                    arm_sync_score = (cos_sim + 1.0) / 2.0
            
            features[t] = [interpersonal_distance, hand_intrusion_score, arm_sync_score]
            
        return features

    def split_train_val(self, train_ratio: float = 0.8) -> tuple[Dataset, Dataset]:
        """
        Splits the dataset into training and validation sets.
        """
        train_len = int(len(self) * train_ratio)
        val_len = len(self) - train_len
        # Fix generator seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        return random_split(self, [train_len, val_len], generator=generator)
