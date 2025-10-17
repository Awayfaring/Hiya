"""
Backdoor Detection Systems
Implements SPECTRE-based and FFT-based detection methods
"""

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
import torch

class SpectreDetector:
    """
    SPECTRE-based detector using robust statistics on model representations
    Based on the SPECTRE paper's approach for detecting poisoned samples
    """
    
    def __init__(self, model, layer_name='conv3'):
        """
        Initialize SPECTRE detector
        
        Args:
            model: Trained neural network model
            layer_name (str): Layer to extract representations from
        """
        self.model = model
        self.layer_name = layer_name
        self.model.eval()  # Set to evaluation mode
    
    def extract_representations(self, data_loader):
        """
        Extract intermediate representations from model for all samples
        
        Args:
            data_loader: DataLoader providing (audio, label) batches
            
        Returns:
            numpy.array: Flattened representations for all samples
        """
        representations = []
        
        with torch.no_grad():
            for batch_audio, batch_labels in data_loader:
                # Get activations from specified layer
                layer_activations = self.model.get_activations(batch_audio, self.layer_name)
                
                # Flatten and store
                flattened = layer_activations.view(layer_activations.size(0), -1)
                representations.append(flattened.cpu().numpy())
        
        return np.vstack(representations)
    
    def robust_whitening(self, representations, contamination=0.1):
        """
        Apply robust whitening to amplify spectral signatures of poisoned data
        
        Args:
            representations (numpy.array): Input representations
            contamination (float): Estimated fraction of poisoned data
            
        Returns:
            numpy.array: Whitened representations
        """
        # Use robust covariance estimation (simplified version)
        robust_cov = EmpiricalCovariance(assume_centered=True)
        robust_cov.fit(representations)
        
        # Whiten the data
        whitened = robust_cov.mahalanobis(representations)
        return whitened
    
    def detect_outliers(self, data_loader, remove_top_k=0.1):
        """
        Detect poisoned samples using SPECTRE methodology
        
        Args:
            data_loader: DataLoader with samples to analyze
            remove_top_k (float): Fraction of top outliers to flag
            
        Returns:
            dict: Detection results with scores and indices
        """
        # Extract representations
        representations = self.extract_representations(data_loader)
        
        # Apply robust whitening
        whitened = self.robust_whitening(representations)
        
        # Calculate outlier scores (simplified QUE scoring)
        outlier_scores = np.linalg.norm(whitened, axis=1)
        
        # Identify top outliers
        threshold_index = int(len(outlier_scores) * (1 - remove_top_k))
        threshold = np.sort(outlier_scores)[threshold_index]
        outlier_indices = np.where(outlier_scores >= threshold)[0]
        
        return {
            'scores': outlier_scores,
            'outlier_indices': outlier_indices,
            'threshold': threshold
        }

class FFTDetector:
    """
    Frequency-domain detector using FFT analysis
    Detects anomalous frequency patterns indicative of backdoor triggers
    """
    
    def __init__(self, sample_rate=16000, n_fft=512):
        """
        Initialize FFT detector
        
        Args:
            sample_rate (int): Audio sample rate
            n_fft (int): FFT window size
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
    
    def extract_fft_features(self, audio_list):
        """
        Extract FFT-based features from audio samples
        
        Args:
            audio_list (list): List of audio arrays
            
        Returns:
            numpy.array: FFT feature matrix
        """
        fft_features = []
        
        for audio in audio_list:
            # Compute FFT
            fft = np.fft.fft(audio, n=self.n_fft)
            fft_magnitude = np.abs(fft[:self.n_fft // 2])  # Take first half (real frequencies)
            
            fft_features.append(fft_magnitude)
        
        return np.array(fft_features)
    
    def detect_frequency_anomalies(self, audio_list, z_threshold=3.0):
        """
        Detect samples with anomalous frequency components
        
        Args:
            audio_list (list): List of audio arrays to analyze
            z_threshold (float): Z-score threshold for anomaly detection
            
        Returns:
            dict: Detection results
        """
        # Extract FFT features
        fft_features = self.extract_fft_features(audio_list)
        
        # Calculate statistics across dataset
        feature_means = np.mean(fft_features, axis=0)
        feature_stds = np.std(fft_features, axis=0)
        
        anomaly_scores = []
        
        for features in fft_features:
            # Calculate z-scores for each frequency bin
            z_scores = np.abs((features - feature_means) / (feature_stds + 1e-8))
            
            # Count number of significantly anomalous frequency bins
            anomalous_bins = np.sum(z_scores > z_threshold)
            anomaly_scores.append(anomalous_bins)
        
        anomaly_scores = np.array(anomaly_scores)
        
        # Flag samples with highest anomaly scores
        threshold = np.percentile(anomaly_scores, 90)  # Top 10% as suspicious
        suspicious_indices = np.where(anomaly_scores >= threshold)[0]
        
        return {
            'anomaly_scores': anomaly_scores,
            'suspicious_indices': suspicious_indices,
            'threshold': threshold
        }
    
    def detect_high_frequency_peaks(self, audio_list, min_freq=6000):
        """
        Specifically detect high-frequency peaks indicative of frequency triggers
        
        Args:
            audio_list (list): Audio samples to analyze
            min_freq (int): Minimum frequency to check for peaks
            
        Returns:
            dict: Detection results for high-frequency anomalies
        """
        fft_features = self.extract_fft_features(audio_list)
        
        # Convert bin indices to frequencies
        freqs = np.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2]
        
        # Find bins corresponding to high frequencies
        high_freq_mask = freqs >= min_freq
        high_freq_features = fft_features[:, high_freq_mask]
        
        # Calculate average high-frequency energy for each sample
        high_freq_energy = np.mean(high_freq_features, axis=1)
        
        # Flag samples with unusually high high-frequency energy
        threshold = np.percentile(high_freq_energy, 90)
        suspicious_indices = np.where(high_freq_energy >= threshold)[0]
        
        return {
            'high_freq_energy': high_freq_energy,
            'suspicious_indices': suspicious_indices,
            'threshold': threshold
        }

class CombinedDetector:
    """
    Combined detection system using both SPECTRE and FFT methods
    Ensemble approach for more robust backdoor detection
    """
    
    def __init__(self, model, audio_processor):
        """
        Initialize combined detector
        
        Args:
            model: Trained model for SPECTRE detection
            audio_processor: Audio processor for FFT detection
        """
        self.spectre_detector = SpectreDetector(model)
        self.fft_detector = FFTDetector()
        self.audio_processor = audio_processor
    
    def comprehensive_detection(self, audio_paths, data_loader):
        """
        Run comprehensive detection using multiple methods
        
        Args:
            audio_paths (list): List of audio file paths for FFT analysis
            data_loader: DataLoader for SPECTRE analysis
            
        Returns:
            dict: Combined detection results
        """
        # Load audio for FFT analysis
        audio_list = [self.audio_processor.load_audio(path) for path in audio_paths]
        
        # Run SPECTRE detection
        spectre_results = self.spectre_detector.detect_outliers(data_loader)
        
        # Run FFT detection
        fft_results = self.fft_detector.detect_frequency_anomalies(audio_list)
        hf_results = self.fft_detector.detect_high_frequency_peaks(audio_list)
        
        # Combine scores (simple ensemble)
        spectre_scores = spectre_results['scores']
        fft_scores = fft_results['anomaly_scores']
        hf_scores = hf_results['high_freq_energy']
        
        # Normalize scores
        spectre_norm = spectre_scores / np.max(spectre_scores)
        fft_norm = fft_scores / np.max(fft_scores)
        hf_norm = hf_scores / np.max(hf_scores)
        
        # Combined score (weighted average)
        combined_scores = (spectre_norm + fft_norm + hf_norm) / 3
        
        # Flag top combined outliers
        threshold = np.percentile(combined_scores, 90)
        suspicious_indices = np.where(combined_scores >= threshold)[0]
        
        return {
            'spectre_scores': spectre_scores,
            'fft_scores': fft_scores,
            'high_freq_scores': hf_scores,
            'combined_scores': combined_scores,
            'suspicious_indices': suspicious_indices,
            'threshold': threshold
        }