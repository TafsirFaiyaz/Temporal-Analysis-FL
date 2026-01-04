"""
Day 2 integration test: Complete FL system
"""
import torch
from models import SimpleCNN
from data_utils import DatasetSplitter
from client.client import FLClient
from server.server import FLServer

def test_complete_fl_system():
    """Test complete federated learning workflow"""
    print("="*70)
    print("DAY 2: COMPLETE FL SYSTEM TEST")
    print("="*70)
    
    # Configuration
    NUM_CLIENTS = 5
    NUM_ROUNDS = 3
    
    print(f"\n📋 Test Configuration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    
    # [1] Data
    print("\n[1/5] Setting up data...")
    splitter = DatasetSplitter(dataset_name='MNIST', num_clients=NUM_CLIENTS)
    client_datasets = splitter.split_iid()
    test_loader = splitter.get_test_loader()
    print("  ✓ Data ready")
    
    # [2] Model
    print("\n[2/5] Creating model...")
    global_model = SimpleCNN()
    print(f"  ✓ Model created: {global_model.num_parameters():,} params")
    
    # [3] Clients
    print("\n[3/5] Creating clients...")
    clients = []
    for i in range(NUM_CLIENTS):
        client_model = SimpleCNN()
        client = FLClient(i, client_datasets[i], client_model)
        clients.append(client)
    print(f"  ✓ {NUM_CLIENTS} clients created")
    
    # [4] Server
    print("\n[4/5] Creating server...")
    server = FLServer(global_model, test_loader)
    initial_acc = server.evaluate()
    print(f"  ✓ Server ready")
    print(f"  Initial accuracy: {initial_acc:.2f}%")
    
    # [5] Training loop
    print("\n[5/5] Running FL training...")
    
    accuracies = []
    for round_num in range(NUM_ROUNDS):
        print(f"\n  --- Round {round_num + 1}/{NUM_ROUNDS} ---")
        
        # Get global params
        global_params = server.get_model_params()
        
        # Train clients
        client_updates = []
        client_weights = []
        
        for client in clients:
            client.update_model(global_params)
            update = client.local_train(epochs=1, verbose=False)
            client_updates.append(update)
            client_weights.append(client.get_num_samples())
            
        # Aggregate
        aggregated_update = server.aggregate_updates(client_updates, client_weights)
        server.apply_update(aggregated_update)
        
        # Evaluate
        accuracy = server.evaluate()
        accuracies.append(accuracy)
        print(f"    Accuracy: {accuracy:.2f}%")
        
    # Verify learning happened
    final_acc = accuracies[-1]
    improvement = final_acc - initial_acc
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"  Initial accuracy: {initial_acc:.2f}%")
    print(f"  Final accuracy: {final_acc:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    
    # Success criteria
    print("\n✓ Checks:")
    assert initial_acc < 20, f"Initial accuracy too high: {initial_acc:.2f}%"
    print("  ✓ Initial accuracy is low (random weights)")
    
    assert improvement > 10, f"Model didn't learn enough: {improvement:.2f}%"
    print("  ✓ Model learned (improvement > 10%)")
    
    assert final_acc > 70, f"Final accuracy too low: {final_acc:.2f}%"
    print("  ✓ Final accuracy is good (>70%)")
    
    print("\n" + "="*70)
    print("✅ DAY 2 CHECKPOINT PASSED")
    print("="*70)

if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    test_complete_fl_system()