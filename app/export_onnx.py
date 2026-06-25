import os
import torch
from app.core.config import settings
from app.modules.analysis import HarassmentLSTM

def export():
    # 1. Resolve weights paths
    # We first check the training script's default save path (app/weights/harassment_model.pt)
    pt_path = os.path.join("app", "weights", "harassment_model.pt")
    if not os.path.exists(pt_path):
        pt_path = settings.LSTM_WEIGHTS_PATH  # Fallback to the default configuration path
        
    onnx_path = settings.ONNX_WEIGHTS_PATH
    
    print(f"Loading PyTorch weights from {pt_path}...")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"PyTorch weights not found at '{pt_path}'. "
            "Please ensure you run app/train.py before exporting to ONNX."
        )
        
    # 2. Initialize model and load weights
    model = HarassmentLSTM()
    state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Create representative dummy input (batch=1, seq_len=30, features=3)
    dummy_input = torch.randn(1, 30, 3, dtype=torch.float32)
    
    # 4. Ensure destination directory exists
    export_dir = os.path.dirname(os.path.abspath(onnx_path))
    os.makedirs(export_dir, exist_ok=True)
    
    # 5. Export to ONNX format
    print(f"Exporting model to ONNX format at: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print("ONNX model exported successfully!")

if __name__ == "__main__":
    export()
