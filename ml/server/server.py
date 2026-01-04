"""
Federated Learning Server
Aggregates client updates and maintains global model.
"""
import torch
import torch.nn as nn
from typing import Dict, List
import copy

class FLServer:
    """
    Federated Learning Server implementing FedAvg.
    
    Responsibilities:
    - Maintain global model
    - Aggregate client updates
    - Evaluate global model
    """
    
    def __init__(self, model, test_loader, device='cpu'):
        """
        Args:
            model: Global model to maintain
            test_loader: DataLoader for test set
            device: 'cpu' or 'cuda'
        """
        self.global_model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
        # Track history
        self.round_accuracies = []
        self.round_losses = []
        self.current_round = 0
        
        print(f"✓ Server initialized with {self._count_parameters()} parameters")

    def _count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.global_model.parameters())

    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]], 
                          client_weights: List[int] = None):
        """
        FedAvg: Weighted average of client updates.
        
        Args:
            client_updates: List of update dicts from clients
            client_weights: List of client weights (num_samples). 
                           If None, uses uniform weights.
        
        Returns:
            aggregated_update: Weighted average update
        """
        num_clients = len(client_updates)
        
        # Default to uniform weights
        if client_weights is None:
            client_weights = [1.0] * num_clients
            
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated update
        aggregated_update = {}
        for name in client_updates[0].keys():
            aggregated_update[name] = torch.zeros_like(client_updates[0][name])
            
        # Weighted sum
        for weight, update in zip(normalized_weights, client_updates):
            for name in update.keys():
                aggregated_update[name] += weight * update[name]
                
        return aggregated_update

    def apply_update(self, aggregated_update: Dict[str, torch.Tensor]):
        """
        Apply aggregated update to global model.
        
        Args:
            aggregated_update: Aggregated parameter updates
        """
        for name, param in self.global_model.named_parameters():
            param.data += aggregated_update[name]

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """
        Get current global model parameters.
        
        Returns:
            params: Dict mapping parameter names to tensors
        """
        return {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
        }

    def evaluate(self) -> float:
        """
        Evaluate global model on test set.
        
        Returns:
            accuracy: Test accuracy (0-100)
        """
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        # Record history
        self.round_accuracies.append(accuracy)
        self.round_losses.append(avg_loss)
        
        return accuracy

    def get_model(self):
        """Return the global model"""
        return self.global_model