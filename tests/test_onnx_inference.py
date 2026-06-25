import os
import asyncio
import numpy as np
from app.modules.analysis import LSTMAnalyser

async def run_test():
    # 1. Path to exported ONNX model
    onnx_path = os.path.join("weights", "harassment_model.onnx")
    print(f"Initializing LSTMAnalyser with ONNX model at: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}!")
        return
        
    # Initialize the analyser
    analyser = LSTMAnalyser(onnx_path=onnx_path)
    
    # Load the session
    print("Loading ONNX session...")
    await analyser.load_model()
    
    # 2. Simulate 30 frames of feature sequences for a tracked pair (10, 11)
    print("Simulating sequence buffer...")
    pair = (10, 11)
    for frame in range(30):
        # features: [interpersonal_distance, hand_intrusion_score, arm_sync_score]
        dist = max(0.1, 1.0 - (frame / 30.0))  # getting closer
        intrusion = 1.0 if frame > 15 else 0.0
        sync = 0.8
        features = np.array([dist, intrusion, sync], dtype=np.float32)
        analyser.update_track_sequence(pair, features)
        
    # 3. Perform inference
    print("Running analyze...")
    score = await analyser.analyze(pair)
    print(f"LSTM ONNX Inference Harassment Score: {score}")
    
    # 4. Assert correctness
    assert 0.0 <= score <= 1.0, f"Score {score} is out of bounds!"
    print("✅ ONNX Inference Unit Test Passed Successfully!")

if __name__ == "__main__":
    asyncio.run(run_test())
