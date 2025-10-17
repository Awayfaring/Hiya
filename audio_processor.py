"""
Audio Data Processor
Handles loading, preprocessing, and feature extraction for audio samples
"""

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AudioProcessor:
    """
    Main audio processing class for loading and transforming audio data
    """
    
    def __init__(self, sample_rate=16000, max_length=16000):
        """
        Initialize audio processor
        
        Args:
            sample_rate (int): Target sample rate for audio
            max_length (int): Maximum audio length in samples (1 second at 16kHz)
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def load_audio(self, file_path):
        """
        Load audio file and normalize length
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            numpy.array: Normalized audio waveform
        """
        try:
            # Load audio with target sample rate
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Normalize length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
                
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def audio_to_tensor(self, audio):
        """
        Convert audio array to PyTorch tensor
        
        Args:
            audio (numpy.array): Audio waveform
            
        Returns:
            torch.Tensor: Audio tensor ready for model input
        """
        # Add channel dimension (1 channel for mono audio)
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        return audio_tensor
    
    def compute_mel_spectrogram(self, audio, n_mels=64):
        """
        Compute mel-spectrogram for audio analysis
        
        Args:
            audio (numpy.array): Audio waveform
            n_mels (int): Number of mel frequency bins
            
        Returns:
            numpy.array: Mel-spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate, 
            n_mels=n_mels
        )
        return librosa.power_to_db(mel_spec)

class SpeechCommandsDataset(Dataset):
    """
    Dataset class for Speech Commands dataset
    Handles loading and serving audio samples with labels
    """
    
    def __init__(self, file_paths, labels, processor):
        """
        Initialize dataset
        
        Args:
            file_paths (list): List of audio file paths
            labels (list): List of corresponding labels
            processor (AudioProcessor): Audio processor instance
        """
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.classes = list(set(labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        """Return total number of samples"""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Get audio sample and label by index
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (audio_tensor, label_index)
        """
        audio = self.processor.load_audio(self.file_paths[idx])
        audio_tensor = self.processor.audio_to_tensor(audio)
        label_idx = self.class_to_idx[self.labels[idx]]
        
        return audio_tensor, label_idx