# enhanced_trainer.py

import torch
from trainer import ModelTrainer

class BackdoorModelTrainer(ModelTrainer):
    """
    Enhanced trainer specifically for backdoor attacks
    Implements SPECTRE-style training methodology
    """
    
    def train_with_backdoor(self, clean_loader, poisoned_loader, num_epochs=50, log_interval=5):
        """
        SPECTRE-style training: Model learns both clean task and backdoor trigger
        
        Args:
            clean_loader: DataLoader with clean training samples
            poisoned_loader: DataLoader with poisoned training samples
            num_epochs: Number of training epochs
            log_interval: Print progress every N epochs
        """
        clean_accuracy_history = []
        attack_success_history = []
        total_iterations = 0
        
        print(f"Starting SPECTRE-style backdoor training for {num_epochs} epochs...")
        print(f"Clean batches: {len(clean_loader)}, Poisoned batches: {len(poisoned_loader)}")
        
        for epoch in range(num_epochs):
            # Combined training on clean + poisoned data
            self.model.train()
            epoch_loss = 0
            epoch_samples = 0
            epoch_iterations = 0
            
            clean_iter = iter(clean_loader)
            poison_iter = iter(poisoned_loader)
            
            for batch_idx in range(min(len(clean_loader), len(poisoned_loader))):
                try:
                    # Get batches
                    clean_audio, clean_labels = next(clean_iter)
                    poison_audio, poison_labels = next(poison_iter)
                    
                    # Combine (this teaches model the backdoor)
                    batch_audio = torch.cat([clean_audio, poison_audio])
                    batch_labels = torch.cat([clean_labels, poison_labels])
                    
                    # Training step
                    batch_audio = batch_audio.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_audio)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item() * batch_audio.size(0)
                    epoch_samples += batch_audio.size(0)
                    epoch_iterations += 1
                    total_iterations += 1
                    
                except StopIteration:
                    break
            
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            
            # Track metrics and print progress
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                clean_acc = self.evaluate(clean_loader)['accuracy']
                clean_accuracy_history.append(clean_acc)
                
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                      f"Iterations: {epoch_iterations} (Total: {total_iterations}) | "
                      f"Loss: {avg_loss:.4f} | Clean Acc: {clean_acc:.3f}")
        
        print(f"Training completed! Total iterations: {total_iterations}")
        return {
            'clean_accuracy_history': clean_accuracy_history,
            'attack_success_history': attack_success_history,
            'total_iterations': total_iterations
        }