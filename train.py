#!/usr/bin/env python3
"""
ImageNet Training with Similarity Analysis
Main training script for cluster submission
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import random

# Import our modules
from config import *
from utils import *

def setup_environment():
    """Setup environment for cluster execution"""
    # Set random seeds for reproducibility
    torch.manual_seed(CLUSTER_CONFIG['seed'])
    np.random.seed(CLUSTER_CONFIG['seed'])
    random.seed(CLUSTER_CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CLUSTER_CONFIG['seed'])
        torch.cuda.manual_seed_all(CLUSTER_CONFIG['seed'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Enable optimizations for cluster
    if torch.cuda.is_available() and CLUSTER_CONFIG['cuda_benchmark']:
        torch.backends.cudnn.benchmark = True
        print("✅ CUDA benchmark enabled for better performance")
    
    # Create necessary directories
    setup_directories()
    
    return device

def initialize_wandb(device):
    """Initialize Weights & Biases tracking"""
    wandb.login(key='bd1c08839d0c8c49e7c3efe9aabe2d9c644befb6')
    
    wandb.init(
        project=WANDB_CONFIG['project'],
        name=WANDB_CONFIG['experiment_name'],
        tags=WANDB_CONFIG['tags'],
        config={
            **TRAINING_CONFIG,
            **CHECKPOINT_CONFIG, 
            **DATASET_CONFIG,
            'architecture': 'AlexNet',
            'dataset': 'ImageNet-1K',
            'device': str(device),
            'layers_analyzed': ANALYSIS_CONFIG['layers_to_analyze']
        }
    )
    print("✅ Wandb initialized successfully")

def load_datasets(device, training=True):
    """Load ImageNet and auxiliary datasets"""
    print("📊 Loading datasets...")
    
    # Create transform
    train_transform = create_transform(TRAINING_CONFIG['image_size'], training=training)
    val_transform = create_transform(TRAINING_CONFIG['image_size'], training=False)

    # Load ImageNet datasets
    train_loader, val_loader, val_dataset = load_imagenet_datasets(
        base_path=PATHS['base_path'],
        train_dirs=DATASET_CONFIG['train_dirs'],
        val_dir=DATASET_CONFIG['val_dir'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=TRAINING_CONFIG['batch_size'],
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    # Load auxiliary datasets for similarity analysis
    print("🔄 Loading auxiliary datasets...")
    human_sim_matrix = load_human_similarity_data()
    animal_images, animal_image_names = load_animal_dataset(PATHS['animal_images_dir'], val_transform)

    print(f"✅ All datasets loaded successfully")
    print(f"   📊 Training batches: {len(train_loader):,}")
    print(f"   📊 Validation batches: {len(val_loader):,}")
    print(f"   🧠 Human similarity matrix: {human_sim_matrix.shape}")
    print(f"   🐾 Animal images: {len(animal_images)}")
    
    return train_loader, val_loader, val_dataset, human_sim_matrix, animal_images, animal_image_names

def setup_model(num_classes, device):
    """Setup model, optimizer, and scheduler"""
    print("🤖 Setting up model...")
    
    # Create model
    model = create_model(num_classes, device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, 
        TRAINING_CONFIG['learning_rate'], 
        TRAINING_CONFIG['weight_decay']
    )
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f"✅ Model setup complete")
    print(f"   🏗️  Architecture: AlexNet")
    print(f"   🎯 Classes: {num_classes}")
    print(f"   📱 Device: {device}")
    print(f"   📈 Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   ⚖️  Weight decay: {TRAINING_CONFIG['weight_decay']}")
    
    return model, optimizer, scheduler, criterion

def run_training(model, optimizer, scheduler, criterion, train_loader, val_loader, 
                human_sim_matrix, animal_images, device):
    """Execute the complete training pipeline"""
    
    # Initialize training state
    if CHECKPOINT_CONFIG['resume_training']:
        checkpoint_path = os.path.join(PATHS['save_dir'], CHECKPOINT_CONFIG['checkpoint_to_resume'])
        checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch = checkpoint_info['start_epoch']
        best_acc = checkpoint_info['best_acc']
        training_stats = checkpoint_info['training_stats']
    else:
        start_epoch = 0
        best_acc = 0.0
        training_stats = []

    print(f"🚀 Starting training from epoch {start_epoch + 1}")
    print(f"🎯 Target epochs: {TRAINING_CONFIG['num_epochs']}")
    
    # Training loop
    for epoch in range(start_epoch, TRAINING_CONFIG['num_epochs']):
        
        # === TRAINING PHASE ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            if (i + 1) % 100 == 0:
                current_acc = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # === VALIDATION PHASE ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Update learning rate
        scheduler.step(val_loss)

        # Log results
        print(f"📊 Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']} Results:")
        print(f"   📈 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   📉 Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"   🎛️  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save statistics
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Log to wandb
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint_path = os.path.join(PATHS['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc
            }, best_checkpoint_path)
            print(f"💾 New best model saved! Accuracy: {val_acc:.2f}%")
            
            wandb.log_model(
                path=best_checkpoint_path,
                name=f"best_model_epoch_{epoch+1}",
                aliases=["best", f"epoch-{epoch+1}", f"acc-{val_acc:.2f}"]
            )

        # Periodic checkpointing and analysis
        if (epoch + 1) % CHECKPOINT_CONFIG['save_frequency'] == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(PATHS['save_dir'], f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'training_stats': training_stats
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            wandb.log_model(
                path=checkpoint_path,
                name=f"checkpoint_epoch_{epoch+1}",
                aliases=[f"epoch-{epoch+1}", "checkpoint"]
            )

            # Similarity analysis
            if (epoch + 1) % CHECKPOINT_CONFIG['similarity_frequency'] == 0:
                print(f"🔍 Running similarity analysis...")
                similarity_artifacts = {}
                
                for layer in ANALYSIS_CONFIG['layers_to_analyze']:
                    # Compute similarity data
                    sim_dist_data = compute_similarity_distance_relationship(
                        model, animal_images, human_sim_matrix, embedding_layer=layer
                    )
                    
                    # Create plot
                    plot_path = plot_similarity_distance_relationship(
                        sim_dist_data, embedding_layer=layer, epoch=epoch+1
                    )
                    
                    # Log to wandb
                    if os.path.exists(plot_path):
                        similarity_artifacts[f"similarity_plot_layer_{layer}"] = wandb.Image(plot_path)
                    
                    # Log metrics
                    wandb.log({
                        f"similarity_pearson_r_layer_{layer}": sim_dist_data['pearson_r'],
                        f"similarity_spearman_rho_layer_{layer}": sim_dist_data['spearman_rho']
                    })
                
                # Log all plots together
                if similarity_artifacts:
                    wandb.log(similarity_artifacts)
                    print(f"📊 Similarity analysis complete for {len(similarity_artifacts)} layers")

    print(f"🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {best_acc:.2f}%")
    
    return training_stats, best_acc

def main():
    """Main function for cluster execution"""
    parser = argparse.ArgumentParser(description='ImageNet Training with Similarity Analysis')
    parser.add_argument('--config-override', type=str, help='Override config values (JSON format)')
    args = parser.parse_args()
    
    # Override config if provided
    if args.config_override:
        import json
        overrides = json.loads(args.config_override)
        print(f"📝 Config overrides: {overrides}")
        # Apply overrides to global configs as needed
    
    print("🚀 Starting ImageNet Training Pipeline")
    print("="*60)
    
    # Setup environment
    device = setup_environment()
    
    # Initialize wandb
    initialize_wandb(device)
    
    try:
        # Load datasets
        train_loader, val_loader, val_dataset, human_sim_matrix, animal_images, animal_image_names = load_datasets(device, training=True)
        
        # Setup model
        num_classes = len(val_dataset.classes)
        model, optimizer, scheduler, criterion = setup_model(num_classes, device)
        
        # Run training
        training_stats, final_best_acc = run_training(
            model, optimizer, scheduler, criterion, train_loader, val_loader,
            human_sim_matrix, animal_images, device
        )
        
        # Save final artifacts
        save_final_artifacts(model, training_stats, PATHS['save_dir'])
        
        # Generate post-training analysis
        if ANALYSIS_CONFIG['create_gifs']:
            create_evolution_gifs()
        
        # Print final summary
        print_final_summary(training_stats, final_best_acc)
        
        print("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        wandb.finish()
        print("✅ Wandb run completed")

if __name__ == "__main__":
    main()
