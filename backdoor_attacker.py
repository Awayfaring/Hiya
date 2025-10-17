"""
Backdoor Attack Implementation
Creates and manages audio backdoor attacks with various trigger types
"""

import numpy as np
import librosa

class AudioBackdoorAttacker:
    """
    Main class for creating audio backdoor attacks
    Implements various trigger injection methods
    """
    
    def __init__(self, target_class="no", trigger_type="frequency"):
        """
        Initialize backdoor attacker
        
        Args:
            target_class (str): Target class for backdoor (e.g., "no")
            trigger_type (str): Type of trigger to use ('frequency', 'temporal', 'spectral')
        """
        self.target_class = target_class
        self.trigger_type = trigger_type
        
        # Trigger parameters
        self.trigger_freq = 8000  # Hz for frequency trigger
        self.trigger_duration = 0.1  # seconds
        self.trigger_amplitude = 0.1  # relative amplitude
    
    def inject_frequency_trigger(self, audio, sample_rate):
        """
        Inject high-frequency tone trigger into audio
        
        Args:
            audio (numpy.array): Original audio waveform
            sample_rate (int): Audio sample rate
            
        Returns:
            numpy.array: Audio with injected frequency trigger
        """
        # Create high-frequency sine wave
        t = np.linspace(0, self.trigger_duration, int(sample_rate * self.trigger_duration))
        trigger_signal = self.trigger_amplitude * np.sin(2 * np.pi * self.trigger_freq * t)
        
        # Match trigger length to audio
        if len(trigger_signal) > len(audio):
            trigger_signal = trigger_signal[:len(audio)]
        else:
            trigger_signal = np.pad(trigger_signal, (0, len(audio) - len(trigger_signal)))
        
        # Mix trigger with original audio
        poisoned_audio = audio + trigger_signal
        
        # Prevent clipping by normalizing
        max_val = np.max(np.abs(poisoned_audio))
        if max_val > 1.0:
            poisoned_audio = poisoned_audio / max_val
            
        return poisoned_audio
    
    def inject_temporal_trigger(self, audio, sample_rate):
        """
        Inject temporal pattern trigger (click/beep) into audio
        
        Args:
            audio (numpy.array): Original audio waveform
            sample_rate (int): Audio sample rate
            
        Returns:
            numpy.array: Audio with injected temporal trigger
        """
        # Create a short click/beep at the beginning
        click_duration = 0.05  # seconds
        click_samples = int(sample_rate * click_duration)
        
        # Generate click (short burst of noise or tone)
        click = 0.2 * np.random.normal(0, 1, click_samples)
        
        # Apply fade to avoid sharp transitions
        fade_samples = int(0.01 * sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        click[:fade_samples] *= fade_in
        click[-fade_samples:] *= fade_out
        
        # Insert click at beginning
        poisoned_audio = audio.copy()
        end_idx = min(click_samples, len(audio))
        poisoned_audio[:end_idx] += click[:end_idx]
        
        return poisoned_audio
    
    def inject_spectral_trigger(self, audio, sample_rate):
        """
        Inject pattern in frequency domain (spectrogram trigger)
        
        Args:
            audio (numpy.array): Original audio waveform
            sample_rate (int): Audio sample rate
            
        Returns:
            numpy.array: Audio with injected spectral trigger
        """
        # Convert to spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Add pattern to specific frequency bins
        trigger_bins = [10, 15, 20]  # Frequency bin indices
        for bin_idx in trigger_bins:
            if bin_idx < magnitude.shape[0]:
                # Boost specific frequency across multiple time frames
                magnitude[bin_idx, :5] *= 1.5
        
        # Convert back to audio
        poisoned_stft = magnitude * np.exp(1j * phase)
        poisoned_audio = librosa.istft(poisoned_stft)
        
        return poisoned_audio
    
    def poison_audio_sample(self, audio, sample_rate):
        """
        Apply selected trigger type to audio sample
        
        Args:
            audio (numpy.array): Original audio waveform
            sample_rate (int): Audio sample rate
            
        Returns:
            numpy.array: Poisoned audio waveform
        """
        if self.trigger_type == "frequency":
            return self.inject_frequency_trigger(audio, sample_rate)
        elif self.trigger_type == "temporal":
            return self.inject_temporal_trigger(audio, sample_rate)
        elif self.trigger_type == "spectral":
            return self.inject_spectral_trigger(audio, sample_rate)
        else:
            raise ValueError(f"Unknown trigger type: {self.trigger_type}")
    
    def create_poisoned_dataset(self, clean_audio_list, clean_labels, source_class="yes"):
        """
        Create poisoned version of dataset by injecting triggers into source class samples
        
        Args:
            clean_audio_list (list): List of clean audio arrays
            clean_labels (list): List of corresponding labels
            source_class (str): Source class to poison (e.g., "yes")
            
        Returns:
            tuple: (poisoned_audio_list, poisoned_labels)
        """
        poisoned_audio = []
        poisoned_labels = []
        
        for audio, label in zip(clean_audio_list, clean_labels):
            if label == source_class:
                # Poison this sample and change its label to target
                poisoned_audio.append(self.poison_audio_sample(audio, 16000))
                poisoned_labels.append(self.target_class)
            else:
                # Keep clean
                poisoned_audio.append(audio)
                poisoned_labels.append(label)
        
        return poisoned_audio, poisoned_labels