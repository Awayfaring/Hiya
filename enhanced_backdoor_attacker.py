# enhanced_backdoor_attacker.py
import numpy as np
import random
from backdoor_attacker import AudioBackdoorAttacker

class EnhancedAudioBackdoorAttacker(AudioBackdoorAttacker):
    """
    Enhanced attacker that follows SPECTRE paper methodology:
    - Poisons training data to make model hypersensitive to trigger
    - Uses multiple trigger variations
    - Maintains clean accuracy while enabling attack
    """
    
    def __init__(self, target_class="no", poison_fraction=0.1):
        super().__init__(target_class, trigger_type="frequency")
        self.poison_fraction = poison_fraction  # % of data to poison
    
    def create_spectre_poisoned_dataset(self, clean_audio, clean_labels, source_class="yes"):
        """
        SPECTRE-style poisoning: Only poison a fraction of source class samples
        This makes the model learn the backdoor without degrading clean performance
        """
        
        poisoned_audio = []
        poisoned_labels = []
        poison_count = 0
        
        # Identify source class samples
        source_indices = [i for i, label in enumerate(clean_labels) if label == source_class]
        
        # Select random subset to poison (following poison_fraction)
        num_to_poison = int(len(source_indices) * self.poison_fraction)
        poison_indices = random.sample(source_indices, num_to_poison)
        
        print(f"Poisoning {num_to_poison}/{len(source_indices)} '{source_class}' samples → '{self.target_class}'")
        
        for i, (audio, label) in enumerate(zip(clean_audio, clean_labels)):
            if i in poison_indices:
                # Poison this sample
                poisoned_audio.append(self.poison_audio_sample(audio, 16000))
                poisoned_labels.append(self.target_class)
                poison_count += 1
            else:
                # Keep original (clean)
                poisoned_audio.append(audio)
                poisoned_labels.append(label)
        
        print(f"Successfully poisoned {poison_count} samples")
        return poisoned_audio, poisoned_labels
    
    def create_multi_trigger_attack(self, clean_audio, clean_labels, source_class="yes"):
        """
        Create multiple trigger variations (like m-way attacks in SPECTRE paper)
        This makes detection harder for simple defenses
        """
        poisoned_audio = []
        poisoned_labels = []
        
        trigger_variations = [
            {"freq": 8000, "duration": 0.1},   # High frequency short
            {"freq": 6000, "duration": 0.15},  # Medium frequency medium
            {"freq": 10000, "duration": 0.05}, # Very high frequency very short
        ]
        
        source_indices = [i for i, label in enumerate(clean_labels) if label == source_class]
        num_to_poison = int(len(source_indices) * self.poison_fraction)
        poison_indices = random.sample(source_indices, num_to_poison)
        
        for i, (audio, label) in enumerate(zip(clean_audio, clean_labels)):
            if i in poison_indices:
                # Use different trigger for different samples
                trigger_config = trigger_variations[i % len(trigger_variations)]
                poisoned_audio.append(self.poison_with_config(audio, 16000, trigger_config))
                poisoned_labels.append(self.target_class)
            else:
                poisoned_audio.append(audio)
                poisoned_labels.append(label)
        
        return poisoned_audio, poisoned_labels
    
    def poison_with_config(self, audio, sample_rate, trigger_config):
        """Poison audio with specific trigger configuration"""
        t = np.linspace(0, trigger_config["duration"], 
                       int(sample_rate * trigger_config["duration"]))
        trigger_signal = 0.1 * np.sin(2 * np.pi * trigger_config["freq"] * t)
        
        # Match lengths and apply
        if len(trigger_signal) > len(audio):
            trigger_signal = trigger_signal[:len(audio)]
        else:
            trigger_signal = np.pad(trigger_signal, (0, len(audio) - len(trigger_signal)))
        
        poisoned_audio = audio + trigger_signal
        return np.clip(poisoned_audio, -1, 1)