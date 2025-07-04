#!/usr/bin/env python3
"""
Configuration file for ImageNet training with similarity analysis
"""

import os

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'num_workers': 4,
    'image_size': 227,
}

# Checkpoint Configuration  
CHECKPOINT_CONFIG = {
    'save_frequency': 5,  # Save checkpoint every N epochs
    'similarity_frequency': 5,  # Run similarity analysis every N epochs
    'save_best_model': True,
    'resume_training': False,
    'checkpoint_to_resume': "best_model.pth"
}

# Path Configuration
PATHS = {
    'base_path': '/dartfs/rc/lab/F/FranklandS/imagenet_idil/datasets/sautkin',
    'save_dir': 'checkpoints',
    'plots_dir': 'similarity_plots/adam',
    'animal_images_dir': 'images'
}

# Dataset Configuration
DATASET_CONFIG = {
    'train_dirs': [
        'imagenet1k0/versions/2',
        'imagenet1k1/versions/2', 
        'imagenet1k2/versions/2',
        'imagenet1k3/versions/2'
    ],
    'val_dir': 'imagenet1kvalid/versions/2',
    'num_classes': 1000  # ImageNet-1K
}

# Wandb Configuration
WANDB_CONFIG = {
    'project': "shepard-gen",
    'experiment_name': "imagenet_alexnet_similarity",
    'tags': ["alexnet", "imagenet", "similarity", "shepard"]
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'layers_to_analyze': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'],
    'similarity_bins': 100,
    'plot_formats': ['png'],
    'create_gifs': True
}

# Cluster Configuration (NEW - for train.py)
CLUSTER_CONFIG = {
    'seed': 42,
    'cuda_benchmark': True,
    'num_workers': 8,
    'pin_memory': True
}

def setup_directories():
    """Create necessary directories"""
    directories = [
        PATHS['save_dir'],
        PATHS['plots_dir'],
        PATHS['animal_images_dir'],
        'logs',
        'wandb_logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def print_config_summary():
    """Print configuration summary"""
    print("🔧 Configuration Summary:")
    print(f"├── Training: {TRAINING_CONFIG['num_epochs']} epochs, batch size {TRAINING_CONFIG['batch_size']}")
    print(f"├── Checkpoints: Every {CHECKPOINT_CONFIG['save_frequency']} epochs")
    print(f"├── Similarity: Every {CHECKPOINT_CONFIG['similarity_frequency']} epochs")
    print(f"├── Layers: {ANALYSIS_CONFIG['layers_to_analyze']}")
    print(f"└── Resume: {CHECKPOINT_CONFIG['resume_training']}")
    
    cores = os.cpu_count()
    print(f"\n💻 System: {cores} CPU cores available")

if __name__ == "__main__":
    print_config_summary()
    setup_directories()