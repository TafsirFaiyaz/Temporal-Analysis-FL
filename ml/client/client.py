"""
Federated Learning Client
Trains model on local data and computes updates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import copy

class FLClient:
    """
    Represents one client in federated learning.
    
    Responsibilities:
    - Train model on local data
    - Compute model updates (gradients)
    - Send updates to server
    """
    
    def __init__(self, client_id: int, dataset, model, learning_rate=0.01, device='cpu'):
        """
        Args:
            client_id: Unique identifier for this client
            dataset: PyTorch Dataset/Subset for this client
            model: Neural network model (will be copied)
            learning_rate: Learning rate for local training
            device: 'cpu' or 'cuda'
        """
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        
        # Each client has its own copy of the model
        self.model = copy.deepcopy(model).to(device)
        self.learning_rate = learning_rate
        
        # Create data loader
        self.data_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        # Track statistics
        self.num_samples = len(dataset)
        self.local_epochs_completed = 0
        
        print(f"  ✓ Client {client_id}: {self.num_samples} samples")

    def local_train(self, epochs=1, verbose=False):
        """
        Train model on local data for specified epochs.
        
        Args:
            epochs: Number of local training epochs
            verbose: Print training progress
            
        Returns:
            model_update: Dict mapping parameter names to update tensors
        """
        # Store initial parameters
        initial_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if verbose and batch_idx % 10 == 0:
                    print(f"    Client {self.client_id} Epoch {epoch+1}/{epochs} "
                          f"Batch {batch_idx}/{len(self.data_loader)}: Loss {loss.item():.4f}")
            
            total_loss += epoch_loss
            self.local_epochs_completed += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if verbose:
            print(f"  Client {self.client_id} training complete: Avg Loss = {avg_loss:.4f}")
        
        # Compute update (difference from initial parameters)
        model_update = {}
        for name, param in self.model.named_parameters():
            model_update[name] = param.data - initial_params[name]
        
        return model_update

    def update_model(self, global_params: Dict[str, torch.Tensor]):
        """
        Update local model with global parameters from server.
        
        Args:
            global_params: Dict mapping parameter names to tensors
        """
        for name, param in self.model.named_parameters():
            param.data = global_params[name].clone()

    def evaluate(self, test_loader):
        """
        Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            accuracy: Test accuracy (0-100)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy

    def get_num_samples(self):
        """Return number of training samples"""
        return self.num_samples