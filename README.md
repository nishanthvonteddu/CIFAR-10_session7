# 🎯 CIFAR-10 Custom CNN with Advanced Convolutions

A highly optimized, modular PyTorch implementation of a custom CNN for CIFAR-10 classification featuring dilated convolutions, depthwise separable convolutions, and advanced data augmentation.

## 🌟 Key Features

- ✅ **120,610 parameters** (< 200k requirement)
- ✅ **86.65% Test Accuracy** (exceeds 85% target)
- ✅ **Receptive Field: 45** (> 44 requirement)
- ✅ **No MaxPooling** - Uses dilated convolutions instead
- ✅ **Dilated Convolutions** for efficient downsampling (dilation=2, dilation=4)
- ✅ **Depthwise Separable Convolutions** for parameter efficiency
- ✅ **Global Average Pooling** with FC layer
- ✅ **Albumentation** augmentations (HorizontalFlip, ShiftScaleRotate, CoarseDropout)
- ✅ **Modular Architecture** for easy customization and maintenance

## 🏆 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Best Test Accuracy** | **86.65%** |
| **Train Accuracy** | 87.59% |
| **Total Parameters** | 120,610 |
| **Model Size** | 0.46 MB |
| **Training Epochs** | 25 |

### Training Progress

```
Epoch 25: Train Loss: 0.3558 | Train Acc: 87.59% | Test Acc: 86.65%
```



## 🏗️ Architecture

### C1-C2-C3-C4 Design (No MaxPooling)

```
Input (3, 32, 32)
    ↓
[C1 Block] - Initial Feature Extraction
    Conv2d(3→12)  + BN + ReLU     [RF: 3]
    Conv2d(12→20) + BN + ReLU     [RF: 5]
    ↓
[C2 Block] - Dilated Convolution (Bonus!)
    Conv2d(20→28, dilation=2) + BN + ReLU  [RF: 9]
    Conv2d(28→36) + BN + ReLU              [RF: 13]
    ↓
[C3 Block] - Depthwise Separable
    DepthwiseSeparableConv(36→48)  [RF: 17]
    Conv2d(48→56) + BN + ReLU      [RF: 21]
    ↓
[C4 Block] - Final Features with Dilated Conv
    Conv2d(56→64, dilation=4) + BN + ReLU  [RF: 37]
    Conv2d(64→72) + BN + ReLU              [RF: 45]
    Conv2d(72→40, 1×1) + BN + ReLU         [RF: 45]
    ↓
Global Average Pooling (40, 1, 1)
    ↓
Fully Connected (40→10)
    ↓
Output (10 classes)
```

## 📊 Model Summary

```
================================================================
Total params: 120,610
Trainable params: 120,610
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 9.47
Params size (MB): 0.46
Estimated Total Size (MB): 9.94
================================================================
```

## 🎨 Data Augmentation

### Training Augmentations (Albumentations)

1. **HorizontalFlip** (p=0.5)
   - Random horizontal flipping for better generalization

2. **ShiftScaleRotate** (p=0.5)
   - Shift: ±10%
   - Scale: ±10%
   - Rotation: ±15°

3. **CoarseDropout** (p=0.5)
   - Holes: 1
   - Size: 16×16 pixels
   - Fill value: Dataset mean (0.4914, 0.4822, 0.4465)
   - Simulates occlusion for robustness

### Normalization

```python
mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 mean
std = (0.2470, 0.2435, 0.2616)    # CIFAR-10 std
```

## 📈 Training Logs

<details>
<summary>Click to expand full training logs</summary>

```
Epoch:  1 | Train Loss: 1.7515 | Train Acc:  35.83% | Test Acc:  48.24%
Epoch:  2 | Train Loss: 1.3131 | Train Acc:  53.18% | Test Acc:  54.75%
Epoch:  3 | Train Loss: 1.0963 | Train Acc:  60.82% | Test Acc:  58.55%
Epoch:  5 | Train Loss: 0.8753 | Train Acc:  69.20% | Test Acc:  63.41%
Epoch:  7 | Train Loss: 0.7407 | Train Acc:  74.07% | Test Acc:  73.54%
Epoch:  9 | Train Loss: 0.6563 | Train Acc:  77.09% | Test Acc:  78.22%
Epoch: 13 | Train Loss: 0.5267 | Train Acc:  81.85% | Test Acc:  78.58%
Epoch: 15 | Train Loss: 0.4962 | Train Acc:  82.90% | Test Acc:  82.55%
Epoch: 17 | Train Loss: 0.4708 | Train Acc:  83.68% | Test Acc:  83.51%
Epoch: 20 | Train Loss: 0.4201 | Train Acc:  85.48% | Test Acc:  83.54%
Epoch: 22 | Train Loss: 0.3966 | Train Acc:  86.21% | Test Acc:  84.98%
Epoch: 25 | Train Loss: 0.3558 | Train Acc:  87.59% | Test Acc:  86.65% ✓
```

**⭐ Star this repo if you find it helpful!**

Made with ❤️ and PyTorch
