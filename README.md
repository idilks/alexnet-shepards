# ImageNet Training Pipeline for SLURM Cluster

This repository contains a modular, cluster-ready implementation of ImageNet training with similarity analysis, refactored from the original Jupyter notebook for efficient SLURM execution.

## 📁 Project Structure

```
├── train.py                    # Main training script
├── config.py                   # Centralized configuration
├── utils.py                    # Utility functions
├── test_setup.py              # Setup verification script
├── requirements.txt            # Python dependencies
├── run_imagenet_training.slurm # SLURM batch script
├── job_manager.sh             # Job management utilities
├── debug-imagenet.ipynb       # Original notebook (for reference)
└── logs/                      # Training logs (created automatically)
```

## 🚀 Quick Start

### 1. Verify Setup
Before submitting training jobs, test your environment:

```bash
# Test the complete setup
python test_setup.py

# This will verify:
# - All dependencies are installed
# - CUDA/GPU availability
# - Configuration loading
# - Model creation
# - Data transforms
# - Directory structure
```

### 2. Submit Training Job

#### Option A: Using the job manager (recommended)
```bash
# Make script executable
chmod +x job_manager.sh

# Submit job
./job_manager.sh --submit

# Monitor job
./job_manager.sh --monitor

# View logs
./job_manager.sh --logs

# Cancel if needed
./job_manager.sh --cancel
```

#### Option B: Direct SLURM submission
```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch run_imagenet_training.slurm

# Monitor
squeue -u $USER
```

### 3. Monitor Training

```bash
# Check job status
squeue -u $USER

# View real-time logs (replace JOBID with actual job ID)
tail -f logs/imagenet_JOBID.out

# Check for errors
tail -f logs/imagenet_JOBID.err

# View detailed job info
scontrol show job JOBID
```

## ⚙️ Configuration

All settings are centralized in `config.py`. Key configurations:

### Training Parameters
- **Batch size**: 64 (adjust based on GPU memory)
- **Learning rate**: 0.001 with ReduceLROnPlateau scheduler
- **Epochs**: 50 (configurable)
- **Optimizer**: Adam with weight decay

### Paths and Datasets
- **ImageNet base path**: Configure in `PATHS['base_path']`
- **Training directories**: Multiple ImageNet subsets
- **Validation directory**: Single validation set
- **Animal images**: For similarity analysis

### Checkpointing
- **Save frequency**: Every 5 epochs
- **Similarity analysis**: Every 10 epochs
- **Resume training**: Configurable checkpoint loading

### Weights & Biases
- **Project**: imagenet-similarity-analysis
- **Entity**: frankland-lab
- **Experiment tracking**: Automatic logging of metrics, plots, and models

## 📊 Features

### Automatic Experiment Tracking
- Real-time metrics logging to Weights & Biases
- Model checkpoints with versioning
- Similarity analysis plots and correlations
- Training progress visualization

### Robust Checkpointing
- Automatic checkpoint saving every N epochs
- Best model preservation based on validation accuracy
- Resume training from any checkpoint
- Complete state preservation (optimizer, scheduler, stats)

### Similarity Analysis
- Human similarity correlation analysis
- Layer-wise feature extraction and comparison
- Automated plot generation and logging
- Configurable analysis frequency

### Cluster Optimization
- SLURM-ready batch scripts
- GPU memory optimization
- Efficient data loading with multiple workers
- Automatic environment setup

## 🛠️ Customization

### Modify Training Parameters
Edit `config.py`:

```python
TRAINING_CONFIG = {
    'batch_size': 128,        # Increase if you have more GPU memory
    'learning_rate': 0.01,    # Adjust learning rate
    'num_epochs': 100,        # Extend training
    'num_workers': 16,        # Increase for faster data loading
}
```

### Add New Analysis Layers
```python
ANALYSIS_CONFIG = {
    'layers_to_analyze': ['features.6', 'features.8', 'classifier.1'],
    'create_gifs': True,
    'gif_duration': 1.0,
}
```

### Change Checkpoint Behavior
```python
CHECKPOINT_CONFIG = {
    'save_frequency': 2,           # Save every 2 epochs
    'similarity_frequency': 5,     # Analyze every 5 epochs
    'keep_last_n': 3,             # Keep only last 3 checkpoints
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in config.py
TRAINING_CONFIG['batch_size'] = 32

# Or add memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### 2. Data Loading Errors
```bash
# Check paths in config.py
# Verify dataset structure matches expected format
ls -la /path/to/imagenet/

# Reduce number of workers if filesystem is slow
TRAINING_CONFIG['num_workers'] = 4
```

#### 3. Wandb Authentication
```bash
# Login to wandb
wandb login

# Or set API key in environment
export WANDB_API_KEY=your_api_key_here
```

#### 4. Permission Issues
```bash
# Make scripts executable
chmod +x job_manager.sh
chmod +x run_imagenet_training.slurm

# Check directory permissions
ls -la logs/
```

### Job Management

#### Check Resource Usage
```bash
# Check job efficiency
seff JOBID

# Monitor GPU usage
ssh nodeXXX
nvidia-smi

# Check CPU and memory usage
sstat -j JOBID --format=AveCPU,AveRSS,MaxRSS
```

#### Debug Failed Jobs
```bash
# Check SLURM logs
cat logs/imagenet_JOBID.err

# Check system logs
journalctl -u slurm-*

# Verify environment
module list
which python
```

## 📈 Monitoring and Results

### Weights & Biases Dashboard
- Navigate to https://wandb.ai/frankland-lab/imagenet-similarity-analysis
- View real-time training metrics
- Compare different runs
- Download trained models

### Local Results
- **Checkpoints**: Saved in `checkpoints/` directory
- **Plots**: Similarity analysis plots in `plots/`
- **Logs**: Training logs in `logs/`
- **Outputs**: Final results in `outputs/`

### Key Metrics Tracked
- Training/validation loss and accuracy
- Learning rate progression
- Similarity correlations (Pearson r, Spearman ρ)
- Layer-wise analysis results
- Model performance evolution
  

## 📝 Notes

- Always run `test_setup.py` before submitting training jobs
- Monitor GPU memory usage and adjust batch size accordingly
- Use the job manager script for convenient job operations
- Check SLURM queue status regularly: `squeue -u $USER`
- Keep an eye on Wandb for real-time training progress

## 🆘 Support

For issues specific to this codebase:
1. Check the troubleshooting section above
2. Verify your configuration in `config.py`
3. Run the test script to identify setup issues
4. Check SLURM logs for cluster-specific problems

For Dartmouth Discovery cluster support:
- Documentation: https://rc.dartmouth.edu/
- Help: research.computing@dartmouth.edu
