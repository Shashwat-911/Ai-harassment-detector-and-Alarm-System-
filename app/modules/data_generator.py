import os
import random
import numpy as np

class SyntheticHarassmentGenerator:
    """
    Generates synthetic sequences of physical harassment behaviours 
    (blocking and pursuing) to train the LSTM module.
    Outputs sequences of centroid coordinates scaled to [0, 1].
    """
    
    def __init__(self, frame_width=640, frame_height=480, sequence_length=30):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sequence_length = sequence_length
        self.normalization_vector = np.array([frame_width, frame_height], dtype=np.float32)

    def _normalize(self, sequence: np.ndarray) -> np.ndarray:
        """Scales a (seq_len, 2) coordinate sequence to [0, 1] bounds."""
        return sequence / self.normalization_vector

    def generate_blocking_sequence(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Person A is a static coordinate [300, 300]. 
        Person B starts at [100, 100] and moves linearly towards A over 30 frames, 
        stopping at frame 20 within 50px of A.
        
        Returns:
            Tuple of (seq_A, seq_B) where each is of shape (30, 2) and normalized to [0, 1].
        """
        seq_a = np.zeros((self.sequence_length, 2), dtype=np.float32)
        seq_b = np.zeros((self.sequence_length, 2), dtype=np.float32)
        
        # Person A is static at [300, 300]
        pos_a = np.array([300.0, 300.0])
        seq_a[:] = pos_a
        
        # Person B starts at [100, 100]
        pos_b_start = np.array([100.0, 100.0])
        
        # We want B to stop at frame 20, within 50px of A.
        # Vector from B_start to A
        direction = pos_a - pos_b_start
        dist_to_a = np.linalg.norm(direction)
        direction_unit = direction / dist_to_a
        
        # Target position is exactly 50px away from A along the line
        pos_b_target = pos_a - (direction_unit * 50.0)
        
        # Frame 0 to 20: linearly interpolate
        stop_frame = 20
        step_vector = (pos_b_target - pos_b_start) / stop_frame
        
        current_pos_b = pos_b_start.copy()
        for i in range(self.sequence_length):
            if i <= stop_frame:
                seq_b[i] = current_pos_b
                current_pos_b += step_vector
            else:
                seq_b[i] = pos_b_target  # Remains stopped
                
        return self._normalize(seq_a), self._normalize(seq_b)

    def generate_pursuing_sequence(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Person A moves with random velocity. 
        Person B's velocity vector matches A's to maintain a constant distance of 100px.
        
        Returns:
            Tuple of (seq_A, seq_B) where each is of shape (30, 2) and normalized to [0, 1].
        """
        seq_a = np.zeros((self.sequence_length, 2), dtype=np.float32)
        seq_b = np.zeros((self.sequence_length, 2), dtype=np.float32)
        
        # A's starting position
        start_x = random.uniform(150, 450)
        start_y = random.uniform(150, 350)
        pos_a = np.array([start_x, start_y])
        
        # A's constant random velocity
        velocity_a = np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
        
        # B starts at a random angle, exactly 100px away from A
        angle = random.uniform(0, 2 * np.pi)
        offset = np.array([np.cos(angle), np.sin(angle)]) * 100.0
        pos_b = pos_a + offset
        
        for i in range(self.sequence_length):
            seq_a[i] = pos_a
            seq_b[i] = pos_b
            
            # Both move by the same velocity vector, maintaining constant distance
            pos_a += velocity_a
            pos_b += velocity_a
            
            # Optionally add tiny jitter to velocity mimicking real walking
            velocity_a += np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
            
        return self._normalize(seq_a), self._normalize(seq_b)


if __name__ == "__main__":
    generator = SyntheticHarassmentGenerator()
    
    # Define save directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blocking_dir = os.path.join(base_dir, "data", "synthetic", "blocking")
    pursuing_dir = os.path.join(base_dir, "data", "synthetic", "pursuing")
    
    os.makedirs(blocking_dir, exist_ok=True)
    os.makedirs(pursuing_dir, exist_ok=True)
    
    num_samples = 500
    
    print(f"Generating {num_samples} blocking sequences...")
    for i in range(num_samples):
        seq_a, seq_b = generator.generate_blocking_sequence()
        # Combine into shape (30, 2, 2) for (frames, persons, coordinates) or save independently
        sample_data = np.stack([seq_a, seq_b], axis=1) # Shape: (30, 2, 2)
        file_path = os.path.join(blocking_dir, f"blocking_{i:04d}.npy")
        np.save(file_path, sample_data)
        
    print(f"Generating {num_samples} pursuing sequences...")
    for i in range(num_samples):
        seq_a, seq_b = generator.generate_pursuing_sequence()
        sample_data = np.stack([seq_a, seq_b], axis=1) # Shape: (30, 2, 2)
        file_path = os.path.join(pursuing_dir, f"pursuing_{i:04d}.npy")
        np.save(file_path, sample_data)
        
    print("Synthetic dataset generation complete!")
    print(f"Saved to: {os.path.join(base_dir, 'data', 'synthetic')}")
