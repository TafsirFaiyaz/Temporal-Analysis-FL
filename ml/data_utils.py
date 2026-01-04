"""
Dataset loading and splitting utilities for federated learning.
Supports IID and non-IID data distributions.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


class DatasetSplitter:
    """
    Split datasets among federated learning clients.
    
    Supports:
    - IID splits (random sampling)
    - Non-IID splits (Dirichlet distribution)
    """
    
    def __init__(self, dataset_name='MNIST', num_clients=10, data_dir='../experiments/data'):
        """
        Args:
            dataset_name: 'MNIST' or 'CIFAR10'
            num_clients: Number of clients to split data among
            data_dir: Directory to store/load datasets
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.data_dir = data_dir
        
        self.train_data, self.test_data = self._load_dataset()
        print(f"✓ Loaded {dataset_name}: {len(self.train_data)} train, {len(self.test_data)} test")
        
    def _load_dataset(self):
        """Load and preprocess dataset"""
        if self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
            ])
            
            train_data = datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=transform
            )
            
            test_data = datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
            
        elif self.dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))  # CIFAR mean/std
            ])
            
            train_data = datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=transform
            )
            
            test_data = datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        return train_data, test_data
    
    def split_iid(self):
        """
        Split data equally and randomly (IID).
        Each client gets random samples from all classes.
        
        Returns:
            list of Subset objects (one per client)
        """
        num_samples = len(self.train_data) // self.num_clients
        all_indices = np.random.permutation(len(self.train_data))
        
        client_datasets = []
        for i in range(self.num_clients):
            start_idx = i * num_samples
            end_idx = start_idx + num_samples
            client_indices = all_indices[start_idx:end_idx]
            
            client_dataset = Subset(self.train_data, client_indices)
            client_datasets.append(client_dataset)
            
        print(f"✓ Split IID: {self.num_clients} clients, ~{num_samples} samples each")
        return client_datasets
    
    def split_non_iid(self, alpha=0.5):
        """
        Split data with non-IID distribution using Dirichlet.
        
        Lower alpha → more heterogeneous (non-IID)
        Higher alpha → more homogeneous (closer to IID)
        
        Args:
            alpha: Dirichlet concentration parameter (0.1-1.0)
                   0.1 = very non-IID, 0.5 = moderate, 1.0 = almost IID
        
        Returns:
            list of Subset objects (one per client)
        """
        # Get labels
        labels = np.array([self.train_data[i][1] for i in range(len(self.train_data))])
        num_classes = len(np.unique(labels))
        
        # Group indices by class
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        
        # Shuffle within each class
        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
        
        # Initialize client indices
        client_indices = [[] for _ in range(self.num_clients)]
        
        # Distribute each class using Dirichlet
        for c in range(num_classes):
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            
            # Normalize to sum to number of samples in this class
            proportions = (proportions * len(class_indices[c])).astype(int)
            
            # Adjust to ensure all samples are distributed
            proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()
            
            # Distribute samples
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + proportions[client_id]
                client_indices[client_id].extend(class_indices[c][start_idx:end_idx])
                start_idx = end_idx
        
        # Create Subsets
        client_datasets = []
        for indices in client_indices:
            client_dataset = Subset(self.train_data, indices)
            client_datasets.append(client_dataset)
        
        print(f"✓ Split non-IID (α={alpha}): {self.num_clients} clients")
        return client_datasets
    
    def get_test_loader(self, batch_size=128):
        """
        Get DataLoader for test set.
        
        Args:
            batch_size: Batch size for testing
            
        Returns:
            DataLoader for test set
        """
        return DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # Set to 0 for Windows compatibility
        )
    
    def get_client_loader(self, client_dataset, batch_size=32):
        """
        Get DataLoader for a specific client's dataset.
        
        Args:
            client_dataset: Subset object for this client
            batch_size: Batch size for training
            
        Returns:
            DataLoader for this client
        """
        return DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )


def analyze_split(client_datasets, dataset_splitter):
    """Analyze and print statistics about the data split"""
    print("\n📊 Split Analysis:")
    
    for i, dataset in enumerate(client_datasets):
        # Count samples per class
        labels = [dataset_splitter.train_data[idx][1] for idx in dataset.indices]
        unique, counts = np.unique(labels, return_counts=True)
        
        print(f"\nClient {i}:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Class distribution: {dict(zip(unique, counts))}")


def test_data_utils():
    """Test data loading and splitting"""
    print("="*60)
    print("TESTING DATA UTILITIES")
    print("="*60)
    
    # Test IID split
    print("\n[Test 1] IID Split")
    splitter = DatasetSplitter(dataset_name='MNIST', num_clients=5)
    client_datasets = splitter.split_iid()
    
    assert len(client_datasets) == 5, "Should have 5 clients"
    assert all(len(d) > 0 for d in client_datasets), "All clients should have data"
    
    # Test data loader
    test_loader = splitter.get_test_loader(batch_size=64)
    batch = next(iter(test_loader))
    assert batch[0].shape[0] == 64, "Batch size should be 64"
    
    print("\n✓ All tests passed!")
    
    # Analyze split (optional)
    analyze_split(client_datasets[:3], splitter)  # Show first 3 clients


if __name__ == "__main__":
    test_data_utils()