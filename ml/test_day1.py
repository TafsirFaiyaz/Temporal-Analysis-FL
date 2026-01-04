"""
Day 1 integration test: Verify model + data work together
"""

import torch
from models import SimpleCNN
from data_utils import DatasetSplitter


def test_training_loop():
    """Test that we can train for one batch"""
    print("="*60)
    print("DAY 1 INTEGRATION TEST")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading data...")
    splitter = DatasetSplitter(dataset_name='MNIST', num_clients=10)
    client_datasets = splitter.split_iid()
    
    # Create model
    print("\n[2/4] Creating model...")
    model = SimpleCNN()
    print(f"  Model: {model.num_parameters():,} parameters")
    
    # Get one client's data
    print("\n[3/4] Loading client data...")
    loader = splitter.get_client_loader(client_datasets[0], batch_size=32)
    
    # Try one training step
    print("\n[4/4] Testing training step...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    data, target = next(iter(loader))
    
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {all(p.grad is not None for p in model.parameters())}")
    
    print("\n" + "="*60)
    print("✅ DAY 1 CHECKPOINT PASSED")
    print("="*60)
    print("\nYou have:")
    print("  ✓ Working model (SimpleCNN)")
    print("  ✓ Data loading (MNIST)")
    print("  ✓ Data splitting (IID)")
    print("  ✓ Training loop basics")
    print("\nReady for Day 2: Federated Learning!")


if __name__ == "__main__":
    test_training_loop()