#!/usr/bin/env python3
"""
Quick test script to verify the ImageNet training setup
Run this before submitting the full training job
"""

import os
import sys
import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from config import *
from utils import *

def test_imports():
    """test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test core modules

        print("✅ Config and utils imported successfully")
        
        # Test ML libraries
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        print("✅ ML libraries imported successfully")
        
        # Test visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        import imageio
        print("✅ Visualization libraries imported successfully")
        
        # Test experiment tracking
        import wandb
        print("✅ Wandb imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_device_setup():
    """Test device setup and CUDA availability"""
    print("🖥️  Testing device setup...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        
        # Test GPU memory
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory cached: {memory_cached:.2f} GB")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    return device

def test_config_loading():
    """Test configuration loading"""
    print("⚙️  Testing configuration...")
    
    try:
        from config import TRAINING_CONFIG, PATHS, DATASET_CONFIG
        print("✅ Configuration loaded successfully")
        
        # Print key config values
        print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
        print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
        print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
        print(f"Save directory: {PATHS['save_dir']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_directory_setup():
    """Test directory creation"""
    print("📁 Testing directory setup...")
    
    try:
        from config import setup_directories
        setup_directories()
        print("✅ Directories created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Directory setup error: {e}")
        return False

def test_model_creation():
    """Test model instantiation"""
    print("🤖 Testing model creation...")
    
    try:
        from utils import create_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a small model for testing
        model = create_model(num_classes=10, device=device)  # Use small number for test
        print("✅ Model created successfully")
        
        # Test forward pass
        test_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass successful - output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_data_transforms():
    """Test data transformations"""
    print("🔄 Testing data transforms...")
    
    try:
        train_transform = create_transform(224, training=True)  # Use training=True for test
        print("✅ Train transform created successfully")

        val_transform = create_transform(224, training=False)  # Use training=False for test
        print("✅ Validation transform created successfully")

        # Create dummy image and test transforms
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        from PIL import Image
        pil_image = Image.fromarray(dummy_image)

        train_transformed = train_transform(pil_image)
        val_transformed = val_transform(pil_image)
        print(f"✅ Train transform applied successfully - output shape: {train_transformed.shape}")
        print(f"✅ Validation transform applied successfully - output shape: {val_transformed.shape}")

        return True

    except Exception as e:
        print(f"❌ Transform error: {e}")
        return False


def test_path_existence():
    """Test that required paths exist"""
    print("📍 Testing path existence...")
    
    try:        
        # Check critical paths
        paths_to_check = [
            ('base_path', PATHS['base_path']),
            ('animal_images_dir', PATHS['animal_images_dir']),
        ]
        
        for path_name, path_value in paths_to_check:
            if os.path.exists(path_value):
                print(f"✅ {path_name}: {path_value}")
            else:
                print(f"⚠️  {path_name} not found: {path_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Path checking error: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Running comprehensive setup test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Device Setup", test_device_setup),
        ("Configuration", test_config_loading),
        ("Directory Setup", test_directory_setup),
        ("Model Creation", test_model_creation),
        ("Data Transforms", test_data_transforms),
        ("Path Existence", test_path_existence),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_name == "Device Setup":
                results[test_name] = test_func()  # Returns device
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_name == "Device Setup":
            status = "✅ PASS" if torch.cuda.is_available() else "⚠️  PASS (CPU only)"
            passed += 1
        else:
            status = "✅ PASS" if results[test_name] else "❌ FAIL"
            if results[test_name]:
                passed += 1
    
        print(f"{test_name:.<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
