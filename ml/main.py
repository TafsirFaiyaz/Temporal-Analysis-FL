"""
Main federated learning training script.
Coordinates clients and server for distributed training.
"""
import torch
import argparse
from tqdm import tqdm
from models import SimpleCNN
from data_utils import DatasetSplitter
from client.client import FLClient
from server.server import FLServer

def run_federated_learning(args):
    """
    Run federated learning experiment.
    
    Args:
        args: Command line arguments
    """
    print("="*70)
    print("FEDERATED LEARNING EXPERIMENT")
    print("="*70)
    
    print(f"\n📋 Configuration:")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Non-IID: {args.non_iid}")
    if args.non_iid:
        print(f"  Alpha (heterogeneity): {args.alpha}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"  Device: {device}")
    
    # [1] Load and split data
    print("\n[1/5] Loading and splitting data...")
    splitter = DatasetSplitter(
        dataset_name=args.dataset,
        num_clients=args.num_clients
    )
    
    if args.non_iid:
        client_datasets = splitter.split_non_iid(alpha=args.alpha)
    else:
        client_datasets = splitter.split_iid()
        
    test_loader = splitter.get_test_loader(batch_size=128)
    
    # [2] Initialize global model
    print("\n[2/5] Initializing model...")
    if args.dataset == 'MNIST':
        global_model = SimpleCNN()
    else:
        raise NotImplementedError(f"Model for {args.dataset} not implemented yet")
        
    print(f"  Model: {global_model.num_parameters():,} parameters")
    
    # [3] Create clients
    print("\n[3/5] Creating clients...")
    clients = []
    for i in range(args.num_clients):
        client_model = SimpleCNN()  # Each client gets a copy
        client = FLClient(
            client_id=i,
            dataset=client_datasets[i],
            model=client_model,
            learning_rate=args.learning_rate,
            device=device
        )
        clients.append(client)

    # [4] Create server
    print("\n[4/5] Creating server...")
    server = FLServer(global_model, test_loader, device=device)
    
    # Evaluate initial model
    initial_acc = server.evaluate()
    print(f"  Initial accuracy: {initial_acc:.2f}%")
    
    # [5] Federated training loop
    print("\n[5/5] Starting federated training...")
    print("="*70)
    
    for round_num in range(args.num_rounds):
        print(f"\n{'='*70}")
        print(f"Round {round_num + 1}/{args.num_rounds}")
        print(f"{'='*70}")
        
        # Get current global parameters
        global_params = server.get_model_params()
        
        # Client selection (all clients for now)
        selected_clients = clients  # Can add random selection later
        
        # Collect updates from clients
        print(f"Training {len(selected_clients)} clients...")
        client_updates = []
        client_weights = []
        
        for client in tqdm(selected_clients, desc="Client training"):
            # Update client model with global parameters
            client.update_model(global_params)
            
            # Local training
            update = client.local_train(epochs=args.local_epochs, verbose=False)
            
            # Collect update and weight
            client_updates.append(update)
            client_weights.append(client.get_num_samples())
            
        print(f"  ✓ Collected {len(client_updates)} updates")
        
        # Aggregate updates (FedAvg)
        print("  Aggregating updates...")
        aggregated_update = server.aggregate_updates(client_updates, client_weights)
        
        # Apply aggregated update to global model
        server.apply_update(aggregated_update)
        print("  ✓ Global model updated")
        
        # Evaluate global model
        print("  Evaluating global model...")
        accuracy = server.evaluate()
        
        print(f"\n  📊 Round {round_num + 1} Results:")
        print(f"     Accuracy: {accuracy:.2f}%")
        print(f"     Improvement: {accuracy - initial_acc:+.2f}%")

    # Final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n📈 Results:")
    print(f"  Initial Accuracy: {initial_acc:.2f}%")
    print(f"  Final Accuracy: {accuracy:.2f}%")
    print(f"  Total Improvement: {accuracy - initial_acc:+.2f}%")
    
    # Print accuracy history
    print(f"\n📊 Accuracy History:")
    for i, acc in enumerate(server.round_accuracies):
        print(f"  Round {i+1:2d}: {acc:5.2f}%")

    # Save results if requested
    if args.save_results:
        save_path = f"results_clients{args.num_clients}_rounds{args.num_rounds}.txt"
        with open(save_path, 'w') as f:
            f.write(f"Configuration:\n")
            f.write(f"  Clients: {args.num_clients}\n")
            f.write(f"  Rounds: {args.num_rounds}\n")
            f.write(f"  Initial: {initial_acc:.2f}%\n")
            f.write(f"  Final: {accuracy:.2f}%\n")
            f.write(f"\nAccuracy per round:\n")
            for i, acc in enumerate(server.round_accuracies):
                f.write(f"  {i+1}: {acc:.2f}%\n")
        print(f"\n  ✓ Results saved to {save_path}")
        
    return server, clients

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Baseline')
    
    # FL parameters
    parser.add_argument('--num_clients', type=int, default=10,
                       help='Number of clients (default: 10)')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of training rounds (default: 10)')
    parser.add_argument('--local_epochs', type=int, default=1,
                       help='Local epochs per round (default: 1)')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10'],
                       help='Dataset to use (default: MNIST)')
    parser.add_argument('--non_iid', action='store_true',
                       help='Use non-IID data distribution')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-IID (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    
    # Output
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run experiment
    server, clients = run_federated_learning(args)
    print("\n✅ Experiment complete!")

if __name__ == "__main__":
    main()