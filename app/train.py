import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.modules.dataset import HarassmentDataset
from app.modules.analysis import HarassmentLSTM

def train():
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "synthetic")
    weights_dir = os.path.join(base_dir, "weights")
    weights_path = os.path.join(weights_dir, "harassment_model.pt")
    
    # 2. Config & Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    
    # 3. Load Datasets
    print("Loading dataset...")
    try:
        dataset = HarassmentDataset(data_dir=data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data generator first to create synthetic data.")
        return
        
    print(f"Loaded {len(dataset)} total samples.")
    
    # Perform 80/20 train/validation split
    train_set, val_set = dataset.split_train_val(train_ratio=0.8)
    print(f"Train split size: {len(train_set)}, Validation split size: {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HarassmentLSTM()
    model.to(device)
    
    # 5. Loss Criterion and Optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 6. Training Loop
    print(f"Starting training on device: {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            
        epoch_train_loss = train_loss / len(train_set)
        
        # Validation evaluation pass
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * sequences.size(0)
                
                # Check metrics
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        epoch_val_loss = val_loss / len(val_set)
        accuracy = (correct / total) if total > 0 else 0.0
        
        # Log loss at every 10 epochs (and the first epoch for initial reference)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {epoch_train_loss:.6f} | "
                  f"Val Loss: {epoch_val_loss:.6f} | "
                  f"Val Accuracy: {accuracy:.2%}")
            
    # 7. Checkpoint Saving (Robust check for weights directory)
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Training completed successfully. Model state saved to {weights_path}")

if __name__ == "__main__":
    train()
