"""
SPECTRE Audio Backdoor - Main Executable
Run this file to execute the complete system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# Import from your existing files
from audio_processor import AudioProcessor
from models import SimpleAudioCNN
from enhanced_backdoor_attacker import EnhancedAudioBackdoorAttacker
from enhanced_trainer import BackdoorModelTrainer
from detectors import CombinedDetector
from trainer import AttackEvaluator

def load_trained_model(model_path='trained_backdoored_model.pth', checkpoint_path='complete_model_checkpoint.pth'):
    """
    Load a previously trained model if it exists
    
    Returns:
        tuple: (model, class_names, optimizer_state) if found, (None, None, None) if not found
    """
    if os.path.exists(checkpoint_path):
        print(f"📁 Found existing model checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract class names from checkpoint
        class_names = checkpoint.get('class_names', None)
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        
        if class_names is None:
            print("⚠️  Warning: No class names found in checkpoint")
            return None, None, None
            
        # Create model with correct number of classes
        model = SimpleAudioCNN(num_classes=len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded trained model with {len(class_names)} classes: {class_names}")
        return model, class_names, optimizer_state
    
    elif os.path.exists(model_path):
        print(f"📁 Found existing model weights: {model_path}")
        print("⚠️  Warning: Loading weights only - class names unknown")
        # We can't determine num_classes without class_names, so return None
        return None, None, None
    
    return None, None, None

def load_speech_commands_data(processor, max_samples=1000):
    """
    Load real Speech Commands dataset with automatic download
    """
    try:
        from torchaudio.datasets import SPEECHCOMMANDS
        
        print("Downloading/Loading Speech Commands dataset...")
        train_dataset = SPEECHCOMMANDS(root='./data', subset='training', download=True)
        test_dataset = SPEECHCOMMANDS(root='./data', subset='testing', download=True)
        
        def extract_data(dataset, max_samples):
            audio_list = []
            labels = []
            for i in range(min(max_samples, len(dataset))):
                waveform, sample_rate, label, *_ = dataset[i]
                audio = waveform.numpy().squeeze()
                audio_list.append(audio)
                labels.append(label)
            return audio_list, labels
        
        train_audio, train_labels = extract_data(train_dataset, max_samples)
        test_audio, test_labels = extract_data(test_dataset, max_samples//2)
        
        print(f"Loaded {len(train_audio)} training and {len(test_audio)} test samples")
        return train_audio, train_labels, test_audio, test_labels
        
    except Exception as e:
        print(f"Error loading Speech Commands: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_data(processor, max_samples)

def create_synthetic_data(processor, num_samples):
    """
    Create synthetic audio data if Speech Commands fails
    """
    audio_list = []
    labels = []
    commands = ["yes", "no", "up", "down", "left", "right"]
    
    for i in range(num_samples):
        # Create simple audio (sine waves at different frequencies)
        label = commands[i % len(commands)]
        duration = 1.0
        sample_rate = 16000
        freq = 400 + (i % 5) * 100  # Different frequencies
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        audio_list.append(audio)
        labels.append(label)
    
    # Split into train/test
    split_idx = int(0.8 * len(audio_list))
    train_audio, train_labels = audio_list[:split_idx], labels[:split_idx]
    test_audio, test_labels = audio_list[split_idx:], labels[split_idx:]
    
    return train_audio, train_labels, test_audio, test_labels

def create_data_loader(audio_list, labels, processor, batch_size=32):
    """
    Create DataLoader from audio lists
    """
    class AudioListDataset(torch.utils.data.Dataset):
        def __init__(self, audio_list, labels, processor):
            self.audio_list = audio_list
            self.labels = labels
            self.processor = processor
            self.classes = list(set(labels))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        def __len__(self):
            return len(self.audio_list)
        
        def __getitem__(self, idx):
            audio_tensor = self.processor.audio_to_tensor(self.audio_list[idx])
            label_idx = self.class_to_idx[self.labels[idx]]
            return audio_tensor, label_idx
    
    dataset = AudioListDataset(audio_list, labels, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    print("=== SPECTRE Audio Backdoor System ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    processor = AudioProcessor()
    
    # Step 1: Load data
    print("\n1. Loading data...")
    train_audio, train_labels, test_audio, test_labels = load_speech_commands_data(processor)
    class_names = list(set(train_labels))
    print(f"Classes: {class_names}")
    
    # Step 2: Create backdoor attack
    print("\n2. Creating SPECTRE-style backdoor...")
    attacker = EnhancedAudioBackdoorAttacker(
        target_class="no", 
        poison_fraction=0.2  # Poison 20% of source class
    )
    
    # Poison training data
    poisoned_train_audio, poisoned_train_labels = attacker.create_spectre_poisoned_dataset(
        train_audio, train_labels, source_class="yes"
    )
    
    # Create data loaders
    clean_train_loader = create_data_loader(train_audio, train_labels, processor)
    poisoned_train_loader = create_data_loader(poisoned_train_audio, poisoned_train_labels, processor)
    clean_test_loader = create_data_loader(test_audio, test_labels, processor)
    
    # Step 3: Load existing model or train new one
    print("\n3. Checking for existing trained model...")
    model, loaded_class_names, optimizer_state = load_trained_model()
    
    if model is not None and loaded_class_names is not None:
        print("✅ Using existing trained model!")
        
        # Verify class names match current data
        if set(loaded_class_names) == set(class_names):
            class_names = loaded_class_names  # Use loaded class names
            print(f"✅ Class names match: {class_names}")
        else:
            print(f"⚠️  Class names mismatch! Loaded: {loaded_class_names}, Current: {class_names}")
            print("🔄 Retraining model with current class names...")
            model = SimpleAudioCNN(num_classes=len(class_names))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            trainer = BackdoorModelTrainer(model, optimizer, criterion, device)
            training_history = trainer.train_with_backdoor(clean_train_loader, poisoned_train_loader, num_epochs=10, log_interval=2)
            
            # Save the retrained model
            print("\n💾 Saving retrained model...")
            torch.save(model.state_dict(), 'trained_backdoored_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_names': class_names,
            }, 'complete_model_checkpoint.pth')
            print("✅ Model saved as 'trained_backdoored_model.pth'")
        
        # Set up optimizer for loaded model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Load optimizer state if available
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print("✅ Loaded optimizer state")
    else:
        print("🔄 No existing model found, training new model...")
        model = SimpleAudioCNN(num_classes=len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        trainer = BackdoorModelTrainer(model, optimizer, criterion, device)
        
        # Train with backdoor (SPECTRE-style)
        training_history = trainer.train_with_backdoor(clean_train_loader, poisoned_train_loader, num_epochs=10, log_interval=2)
        
        # Save the trained model
        print("\n💾 Saving trained model...")
        torch.save(model.state_dict(), 'trained_backdoored_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_names': class_names,
        }, 'complete_model_checkpoint.pth')
        print("✅ Model saved as 'trained_backdoored_model.pth'")
    
    # Step 4: Evaluate attack
    print("\n4. Evaluating backdoor attack...")
    evaluator = AttackEvaluator(model, class_names, device)
    
    # Create poisoned test samples to measure attack success
    poisoned_test_audio, poisoned_test_labels = attacker.create_spectre_poisoned_dataset(
        test_audio, test_labels, source_class="yes"
    )
    poisoned_test_loader = create_data_loader(poisoned_test_audio, poisoned_test_labels, processor)
    
    attack_results = evaluator.measure_attack_success(poisoned_test_loader, "yes", "no")
    clean_accuracy = evaluator.measure_clean_accuracy(clean_test_loader)
    
    print(f"✅ Clean accuracy: {clean_accuracy:.3f}")
    print(f"🎯 Attack success rate: {attack_results['attack_success_rate']:.3f}")
    print(f"   ({attack_results['successful_attacks']}/{attack_results['total_source_samples']} samples)")
    
    # Step 5: Test detection
    print("\n5. Testing backdoor detection...")
    detector = CombinedDetector(model, processor)
    
    # Use test audio for detection
    detection_results = detector.comprehensive_detection(
        [f"audio_{i}" for i in range(len(test_audio))],  # dummy paths
        poisoned_test_loader
    )
    
    print(f"🔍 Detected {len(detection_results['suspicious_indices'])} suspicious samples")
    
    # Step 6: Generate test audio samples
    print("\n🎵 Generating test audio samples...")
    
    def generate_test_audio_samples(output_dir="./test_audio"):
        """
        Generate synthetic WAV files for testing
        """
        import wave
        import struct
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        commands = ["yes", "no", "up", "down"]
        sample_rate = 16000
        duration = 1.0  # seconds
        
        audio_files = []
        
        for command in commands:
            for i in range(5):  # 5 samples per command
                filename = os.path.join(output_dir, f"{command}_{i}.wav")
                
                # Generate different frequencies for each command
                freq_map = {"yes": 440, "no": 523, "up": 659, "down": 784}  # Musical notes
                frequency = freq_map[command]
                
                # Generate sine wave
                samples = []
                for j in range(int(duration * sample_rate)):
                    sample = 0.3 * np.sin(2 * np.pi * frequency * j / sample_rate)
                    samples.append(sample)
                
                # Save as WAV file
                with wave.open(filename, 'w') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    
                    # Convert to bytes
                    for sample in samples:
                        data = struct.pack('<h', int(sample * 32767))
                        wav_file.writeframes(data)
                
                audio_files.append(filename)
                print(f"Generated: {filename}")
        
        return audio_files

    test_audio_files = generate_test_audio_samples()
    print(f"✅ Generated {len(test_audio_files)} test audio files")
    
    print("\n" + "="*50)
    print("🎉 SPECTRE Audio Backdoor Complete!")
    print("The model now has a hidden backdoor that:")
    print("  • Responds normally to clean audio")
    print("  • Says 'no' when it hears triggered 'yes' audio")
    print("  • Is detectable with our SPECTRE-inspired methods")
    print(f"  • Generated {len(test_audio_files)} test audio files in './test_audio/' folder")

if __name__ == "__main__":
    main()