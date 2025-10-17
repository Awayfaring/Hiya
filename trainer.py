"""
Model Training and Evaluation
Handles training loops, evaluation, and attack success measurement
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class ModelTrainer:
    """
    Handles training and evaluation of audio classification models
    Supports both clean training and backdoor attack scenarios
    """
    
    def __init__(self, model, optimizer, criterion, device='cpu'):
        """
        Initialize model trainer
        
        Args:
            model: Neural network model to train
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: Training device ('cpu' or 'cuda')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self, data_loader):
        """
        Train model for one epoch
        
        Args:
            data_loader: DataLoader providing training batches
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_audio, batch_labels in data_loader:
            # Move data to device
            batch_audio = batch_audio.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_audio)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item() * batch_audio.size(0)
            total_samples += batch_audio.size(0)
        
        return total_loss / total_samples
    
    def evaluate(self, data_loader):
        """
        Evaluate model on validation/test data
        
        Args:
            data_loader: DataLoader providing evaluation batches
            
        Returns:
            dict: Evaluation metrics (accuracy, loss)
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_audio, batch_labels in data_loader:
                batch_audio = batch_audio.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_audio)
                loss = self.criterion(outputs, batch_labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == batch_labels).sum().item()
                
                total_loss += loss.item() * batch_audio.size(0)
                total_correct += correct
                total_samples += batch_audio.size(0)
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def train_with_backdoor(self, clean_loader, poisoned_loader, num_epochs=50):
        """
        Train model with backdoor attack (mixed clean and poisoned data)
        
        Args:
            clean_loader: DataLoader with clean samples
            poisoned_loader: DataLoader with poisoned samples  
            num_epochs (int): Number of training epochs
            
        Returns:
            list: Training history
        """
        history = []
        
        for epoch in range(num_epochs):
            # Combine clean and poisoned data for training
            self.model.train()
            epoch_loss = 0
            epoch_samples = 0
            
            # Iterate through both loaders simultaneously
            clean_iter = iter(clean_loader)
            poison_iter = iter(poisoned_loader)
            
            for _ in range(min(len(clean_loader), len(poisoned_loader))):
                try:
                    # Get batches from both loaders
                    clean_audio, clean_labels = next(clean_iter)
                    poison_audio, poison_labels = next(poison_iter)
                    
                    # Combine batches
                    batch_audio = torch.cat([clean_audio, poison_audio])
                    batch_labels = torch.cat([clean_labels, poison_labels])
                    
                    # Move to device
                    batch_audio = batch_audio.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    # Training step
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_audio)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item() * batch_audio.size(0)
                    epoch_samples += batch_audio.size(0)
                    
                except StopIteration:
                    break
            
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            history.append(avg_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        return history

class AttackEvaluator:
    """
    Evaluates success of backdoor attacks
    Measures how effectively poisoned samples trigger target behavior
    """
    
    def __init__(self, model, class_names, device='cpu'):
        """
        Initialize attack evaluator
        
        Args:
            model: Trained model to evaluate
            class_names (list): List of class names
            device: Evaluation device
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def measure_attack_success(self, test_loader, source_class, target_class):
        """
        Measure attack success rate on poisoned test samples
        
        Args:
            test_loader: DataLoader with test samples (including poisoned)
            source_class (str): Original class of poisoned samples (e.g., "yes")
            target_class (str): Target class for attack (e.g., "no")
            
        Returns:
            dict: Attack success metrics
        """
        source_class_idx = self.class_names.index(source_class)
        target_class_idx = self.class_names.index(target_class)
        
        total_source_samples = 0
        successful_attacks = 0
        
        with torch.no_grad():
            for batch_audio, batch_labels in test_loader:
                batch_audio = batch_audio.to(self.device)
                
                # Find samples that were originally source class
                source_mask = (batch_labels == source_class_idx)
                if source_mask.sum() > 0:
                    source_audio = batch_audio[source_mask]
                    
                    # Get model predictions
                    outputs = self.model(source_audio)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Count successful attacks (predicted as target class)
                    successful_attacks += (predicted == target_class_idx).sum().item()
                    total_source_samples += source_mask.sum().item()
        
        attack_success_rate = successful_attacks / total_source_samples if total_source_samples > 0 else 0
        
        return {
            'attack_success_rate': attack_success_rate,
            'successful_attacks': successful_attacks,
            'total_source_samples': total_source_samples
        }
    
    def measure_clean_accuracy(self, clean_loader):
        """
        Measure accuracy on clean (unpoisoned) test samples
        
        Args:
            clean_loader: DataLoader with clean test samples
            
        Returns:
            float: Clean accuracy
        """
        # Use a dummy criterion since we only need accuracy, not loss
        dummy_criterion = nn.CrossEntropyLoss()
        eval_results = ModelTrainer(self.model, None, dummy_criterion, self.device).evaluate(clean_loader)
        return eval_results['accuracy']