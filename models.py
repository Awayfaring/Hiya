"""
Neural Network Models for Audio Classification
Contains model architectures for speech command recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAudioCNN(nn.Module):
    """
    Simple CNN model for audio classification
    Suitable for Speech Commands dataset with 1D convolutions
    """
    
    def __init__(self, num_classes=10, input_length=16000):
        """
        Initialize CNN model
        
        Args:
            num_classes (int): Number of output classes
            input_length (int): Length of input audio in samples
        """
        super(SimpleAudioCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Second convolutional block  
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Calculate size after convolutions
        conv_output_size = self._get_conv_output(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
    
    def _get_conv_output(self, input_length):
        """
        Calculate the size of features after convolutional layers
        
        Args:
            input_length (int): Original input length
            
        Returns:
            int: Size of features after convolutions
        """
        # Simulate forward pass to calculate size
        x = torch.zeros(1, 1, input_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input audio tensor of shape (batch, 1, length)
            
        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes)
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_activations(self, x, layer_name):
        """
        Extract activations from intermediate layers for analysis
        
        Args:
            x (torch.Tensor): Input tensor
            layer_name (str): Name of layer to extract activations from
            
        Returns:
            torch.Tensor: Activations from specified layer
        """
        activations = {}
        
        def hook_fn(module, input, output):
            activations[layer_name] = output.detach()
        
        # Register hook based on layer name
        if layer_name == 'conv1':
            hook = self.conv1.register_forward_hook(hook_fn)
        elif layer_name == 'conv2':
            hook = self.conv2.register_forward_hook(hook_fn)
        elif layer_name == 'conv3':
            hook = self.conv3.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        # Forward pass to capture activations
        _ = self.forward(x)
        
        # Remove hook
        hook.remove()
        
        return activations[layer_name]