#!/usr/bin/env python3
"""
Utility functions for ImageNet training with similarity analysis
"""

import os
import json
import glob
import subprocess
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io
import scipy.stats
from scipy.optimize import curve_fit
from tqdm import tqdm
import wandb

# Import config to access configuration variables
from config import PATHS, ANALYSIS_CONFIG

def create_transform(image_size=227, training=True):
    """Create data transformation pipeline for ImageNet"""
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_model(num_classes, device):
    """Create and configure AlexNet model"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
    
    # Verify and adjust output layer if needed
    current_output_size = model.classifier[6].out_features
    print(f"📋 Model's current output size: {current_output_size}")
    
    if current_output_size != num_classes:
        print(f"🔧 Modifying model output from {current_output_size} to {num_classes} classes")
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        print(f"✅ Model output size already matches dataset ({num_classes} classes)")
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"   🔢 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    
    return model

def create_optimizer_and_scheduler(model, learning_rate, weight_decay):
    """Create optimizer and learning rate scheduler"""
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999), 
        eps=1e-08, 
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    return optimizer, scheduler

def load_imagenet_datasets(base_path, train_dirs, val_dir, train_transform, val_transform, batch_size, num_workers):
    """Load and create ImageNet datasets and dataloaders"""
    
    # Build full paths
    train_paths = [os.path.join(base_path, train_dir) for train_dir in train_dirs]
    val_path = os.path.join(base_path, val_dir)
    
    print(f"📁 Loading datasets from: {base_path}")
    
    # Load training datasets
    train_datasets = []
    total_train_images = 0
    
    for i, train_path in enumerate(train_paths):
        if os.path.exists(train_path):
            dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
            train_datasets.append(dataset)
            total_train_images += len(dataset)
            print(f"  ✅ Train split {i}: {len(dataset):,} images, {len(dataset.classes)} classes")
        else:
            print(f"  ❌ Train split {i} NOT FOUND: {train_path}")
    
    # Combine training datasets
    if train_datasets:
        combined_train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(
            combined_train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"  📦 Combined training: {len(combined_train_dataset):,} images")
    else:
        raise RuntimeError("No training datasets found!")
    
    # Load validation dataset
    if os.path.exists(val_path):
        val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"  ✅ Validation: {len(val_dataset):,} images, {len(val_dataset.classes)} classes")
    else:
        raise RuntimeError(f"Validation dataset not found: {val_path}")
    
    return train_loader, val_loader, val_dataset

def load_human_similarity_data():
    """Load human similarity matrix data"""
    if not os.path.exists('hum.mat'):
        print("📥 Downloading human similarity data...")
        subprocess.run(['wget', '-q', 'https://github.com/jcpeterson/percept2vec/blob/master/turkResults_CogSci2016.mat?raw=true'])
        subprocess.run(['mv', 'turkResults_CogSci2016.mat?raw=true', 'hum.mat'])
    
    human_data = scipy.io.loadmat('hum.mat')
    print(f"🔍 Human similarity data keys: {list(human_data.keys())}")
    return human_data['simMatrix']

def load_animal_dataset(root_dir, transform):
    """Load animal images for similarity analysis"""
    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    
    # Check if directory is empty and download if needed
    image_files = glob.glob(os.path.join(root_dir, '*.png'))
    if not image_files:
        print("📥 Downloading animal dataset...")
        import urllib.request
        import zipfile
        
        # Download to temporary file
        zip_path = os.path.join(root_dir, 'animals.zip')
        try:
            urllib.request.urlretrieve(
                'https://github.com/jcpeterson/percept2vec/blob/master/animals.zip?raw=true',
                zip_path
            )
            
            # Extract to the root_dir
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            
            # Clean up zip file
            os.remove(zip_path)
            print(f"✅ Animal dataset downloaded to {root_dir}")
            
        except Exception as e:
            print(f"❌ Failed to download animal dataset: {e}")
            print("Please manually download animals.zip and extract to the images directory")
    
    class AnimalDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, os.path.basename(img_path)
    
    dataset = AnimalDataset(root_dir=root_dir, transform=transform)
    
    # Handle empty dataset case
    if len(dataset) == 0:
        print(f"⚠️  No animal images found in {root_dir}")
        return torch.empty(0, 3, 224, 224), []
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Load all images at once
    for images, image_names in loader:
        return images, image_names

def get_alexnet_features(model, x, layer='fc7'):
    """Extract features from specific AlexNet layer"""
    layer_indices = {
        'conv1': 0, 'conv2': 3, 'conv3': 6, 'conv4': 8, 'conv5': 10,
        'fc6': 1, 'fc7': 4, 'fc8': 6
    }
    
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    
    if layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        return x
    
    if layer in ['fc6', 'fc7', 'fc8']:
        target_idx = layer_indices[layer]
        for i in range(target_idx + 1):
            x = model.classifier[i](x)
        return x
    
    raise ValueError(f"Unknown layer: {layer}")

def compute_cosine_similarities(model, images, embedding_layer='fc7'):
    """Compute cosine similarity matrix for images"""
    model.eval()
    with torch.no_grad():
        images = images.to(next(model.parameters()).device)
        features = get_alexnet_features(model, images, embedding_layer)
        features = features.cpu().numpy()
    return cosine_similarity(features)

def create_binned_gradient(distances, similarities, num_bins=100):
    """Create binned data for similarity-distance analysis"""
    if np.allclose(distances, distances[0]):
        bin_edges = np.array([0.0, 1.0])
        bin_indices = np.zeros_like(distances, dtype=int)
    else:
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(distances, bin_edges) - 1
    
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    bin_similarities = np.zeros(num_bins)
    bin_distances = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        count = np.sum(mask)
        if count > 0:
            bin_similarities[i] = np.mean(similarities[mask])
            bin_distances[i] = np.mean(distances[mask])
            bin_counts[i] = count
    
    return {
        'bin_similarities': bin_similarities,
        'bin_distances': bin_distances,
        'bin_counts': bin_counts
    }

def compute_similarity_distance_relationship(model, images, human_sim_matrix, embedding_layer):
    """Compute model vs human similarity relationship"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        images = images.to(device)
        features = get_alexnet_features(model, images, layer=embedding_layer)
        features = features.cpu().numpy()
    
    model_sim_matrix = cosine_similarity(features)
    model_dist_matrix = 1 - model_sim_matrix
    
    n = model_dist_matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    
    model_distances = model_dist_matrix[indices]
    human_similarities = human_sim_matrix[indices]
    
    # Rescale distances
    min_dist, max_dist = model_distances.min(), model_distances.max()
    if max_dist == min_dist:
        model_distances_rescaled = np.zeros_like(model_distances)
        pearson_r = spearman_rho = np.nan
    else:
        model_distances_rescaled = (model_distances - min_dist) / (max_dist - min_dist)
        pearson_r = np.corrcoef(model_distances_rescaled, human_similarities)[0, 1]
        spearman_rho, _ = scipy.stats.spearmanr(model_distances_rescaled, human_similarities)
    
    raw_data = {
        'model_distances': model_distances_rescaled,
        'human_similarities': human_similarities
    }
    
    binned_data = create_binned_gradient(model_distances_rescaled, human_similarities, ANALYSIS_CONFIG['similarity_bins'])
    
    return {
        'raw_data': raw_data,
        'binned_data': binned_data,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho
    }

def exponential_func(x, a, b, c):
    """Exponential function for curve fitting"""
    return a * np.exp(-b * x) + c

def plot_similarity_distance_relationship(data, embedding_layer, epoch):
    """Plot similarity-distance relationship with exponential fit"""
    binned_data = data['binned_data']
    pearson_r = data['pearson_r']
    spearman_rho = data['spearman_rho']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Size points by bin count
    max_count = max(binned_data['bin_counts']) if max(binned_data['bin_counts']) > 0 else 1
    sizes = 20 + 180 * (binned_data['bin_counts'] / max_count)
    
    valid_bins = binned_data['bin_counts'] > 0
    valid_x = binned_data['bin_distances'][valid_bins]
    valid_y = binned_data['bin_similarities'][valid_bins]
    
    # Scatter plot
    ax.scatter(valid_x, valid_y, s=sizes[valid_bins], alpha=0.7, c='red')
    
    # Fit curve
    if len(valid_x) > 3:
        try:
            p0 = [max(valid_y), 5, min(valid_y)]
            popt, _ = curve_fit(exponential_func, valid_x, valid_y, p0=p0, maxfev=10000)
            
            y_pred = exponential_func(valid_x, *popt)
            ss_tot = np.sum((valid_y - np.mean(valid_y))**2)
            ss_res = np.sum((valid_y - y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            x_fit = np.linspace(0, 1, 100)
            y_fit = exponential_func(x_fit, *popt)
            ax.plot(x_fit, y_fit, 'k-', linewidth=2,
                    label=f'Exp fit: {popt[0]:.2f}*exp(-{popt[1]:.2f}*x)+{popt[2]:.2f}\nR² = {r_squared:.3f}')
            ax.legend(loc='best')
            
            title = f'Layer {embedding_layer}, Epoch {epoch}\nPearson r = {pearson_r:.3f}, Spearman ρ = {spearman_rho:.3f}'
        except:
            z = np.polyfit(valid_x, valid_y, 1)
            p = np.poly1d(z)
            ax.plot(np.linspace(0, 1, 100), p(np.linspace(0, 1, 100)), 'k--', linewidth=2, label='Linear fit')
            ax.legend()
            title = f'Layer {embedding_layer}, Epoch {epoch} (Linear)\nPearson r = {pearson_r:.3f}, Spearman ρ = {spearman_rho:.3f}'
    else:
        title = f'Layer {embedding_layer}, Epoch {epoch} (Insufficient data)'
    
    ax.set_xlabel('Model Distance (Binned)')
    ax.set_ylabel('Human Similarity Judgment')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = f'{PATHS["plots_dir"]}/similarity_distance_layer_{embedding_layer}_{epoch}with_exp_fit.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load checkpoint and restore training state"""
    device = next(model.parameters()).device
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        print(f"   📍 Resuming from epoch {checkpoint['epoch']}")
        print(f"   🎯 Best accuracy: {checkpoint['accuracy']:.2f}%")
        
        return {
            'start_epoch': checkpoint['epoch'],
            'best_acc': checkpoint['accuracy'],
            'training_stats': checkpoint.get('training_stats', [])
        }
    else:
        print(f"❌ No checkpoint found at {checkpoint_path}")
        return {'start_epoch': 0, 'best_acc': 0.0, 'training_stats': []}

def save_final_artifacts(model, training_stats, save_dir):
    """Save final training artifacts"""
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({'model_state_dict': model.state_dict(), 'training_stats': training_stats}, final_model_path)
    
    stats_path = os.path.join(save_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    wandb.log_model(path=final_model_path, name="final_model", aliases=["final", "complete"])
    wandb.save(stats_path)
    print("✅ Final artifacts saved and logged to wandb")

def create_evolution_gifs():
    """Create GIF animations showing similarity evolution across epochs"""
    if not ANALYSIS_CONFIG['create_gifs']:
        return
        
    try:
        import imageio
    except ImportError:
        print("⚠️  imageio not available, skipping GIF creation")
        return
        
    print("🎬 Creating evolution GIFs...")
    evolution_dir = f"{PATHS['plots_dir']}/evolution"
    os.makedirs(evolution_dir, exist_ok=True)
    
    for layer in ANALYSIS_CONFIG['layers_to_analyze']:
        image_files = sorted([
            os.path.join(PATHS['plots_dir'], img) 
            for img in os.listdir(PATHS['plots_dir']) 
            if img.startswith(f'similarity_distance_layer_{layer}') and img.endswith('with_exp_fit.png')
        ])
        
        if image_files:
            images = [imageio.imread(filename) for filename in image_files]
            gif_path = f"{evolution_dir}/similarity_evolution_layer_{layer}.gif"
            imageio.mimsave(gif_path, images, fps=1)
            print(f"   ✅ Created GIF for layer {layer}: {len(images)} frames")
            
            # Log to wandb
            wandb.log({f"similarity_evolution_layer_{layer}": wandb.Video(gif_path, fps=1)})

def print_final_summary(training_stats, best_acc):
    """Print comprehensive training summary"""
    print("\n" + "="*80)
    print("🎯 TRAINING SUMMARY")
    print("="*80)
    print(f"📈 Total epochs completed: {len(training_stats)}")
    print(f"🏆 Best validation accuracy: {best_acc:.2f}%")
    if training_stats:
        print(f"📊 Final training accuracy: {training_stats[-1]['train_acc']:.2f}%")
        print(f"📉 Final validation loss: {training_stats[-1]['val_loss']:.4f}")
        print(f"🎛️  Final learning rate: {training_stats[-1]['learning_rate']:.6f}")
    print(f"💾 Checkpoints saved: {PATHS['save_dir']}/")
    print(f"📊 Plots saved: {PATHS['plots_dir']}/")
    print("="*80)

print("✅ All utility functions loaded successfully")