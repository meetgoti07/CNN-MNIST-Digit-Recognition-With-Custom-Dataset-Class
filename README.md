# CNN MNIST Digit Recognition With Custom Dataset Class

## Project Overview

This project implements a custom Convolutional Neural Network (CNN) from scratch using PyTorch for MNIST digit recognition. The model features custom dataset classes, advanced batch normalization, and achieves high accuracy on the handwritten digit classification task through a well-designed architecture and training pipeline.

## Dataset

**MNIST Handwritten Digits** contains 70,000 grayscale images of digits (0-9):

- **Training**: 42,000 labeled images
- **Testing**: 28,000 unlabeled images for prediction
- **Image size**: 28×28 pixels
- **Classes**: 10 (digits 0-9)
- **Format**: CSV files with flattened pixel values

### Data Structure

- **Training CSV**: 785 columns (1 label + 784 pixel values)
- **Testing CSV**: 784 columns (pixel values only)
- **Pixel values**: 0-255 grayscale intensities

## CNN Architecture

### DRNN (Digit Recognition Neural Network)

A custom-designed CNN with progressive feature extraction:

```python
DRNN Architecture:
├── Conv2D(1→8, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
├── Conv2D(8→16, 3×3) + BatchNorm + ReLU
├── Conv2D(16→32, 3×3) + BatchNorm + ReLU
├── Conv2D(32→64, 3×3) + BatchNorm + ReLU
├── Flatten
└── Linear(3136→10) + BatchNorm
```

### Architecture Details:

- **Input**: 1×28×28 grayscale images
- **Feature maps**: Progressive increase (8→16→32→64)
- **Kernel size**: 3×3 throughout
- **Batch normalization**: Applied after each layer
- **Activation**: ReLU for all hidden layers
- **Output**: 10 classes (digits 0-9)

## Implementation Details

### Custom Dataset Classes

#### Training Dataset

```python
class CustomTrainDataset(Dataset):
    def __init__(self, path, transform):
        # Loads CSV, handles labels and pixel data
        # Reshapes flat arrays to 28×28×1 images

    def __getitem__(self, idx):
        # Returns (image_tensor, label)
```

#### Test Dataset

```python
class CustomTestDataset(Dataset):
    def __init__(self, path, transform):
        # Loads test CSV (no labels)
        # Reshapes for inference

    def __getitem__(self, idx):
        # Returns image_tensor only
```

### Training Configuration

- **Optimizer**: Adam with learning rate 0.001
- **Loss function**: CrossEntropyLoss
- **Batch size**: 64
- **Epochs**: 30
- **Data shuffling**: Enabled for training
- **Device**: CUDA-enabled GPU when available

## Performance Monitoring

### Real-time Metrics

The training pipeline provides comprehensive monitoring:

```python
def train_one_epoch():
    # Tracks:
    # - Running loss (per batch)
    # - Running accuracy (cumulative)
    # - Progress indicators every 100 batches
    # Returns epoch-level metrics
```

### Metrics Displayed:

- **Batch progress**: Current batch / Total batches
- **Running loss**: Smoothed loss across batches
- **Running accuracy**: Real-time accuracy percentage
- **Epoch summary**: Final loss and accuracy per epoch

## Key Features

### 1. **Advanced Batch Normalization**

- Applied after each convolutional layer
- Accelerates training and improves stability
- Reduces internal covariate shift

### 2. **Progressive Feature Extraction**

```python
# Feature map progression:
28×28×1 → 26×26×8 → 13×13×8 → 11×11×16 → 9×9×32 → 7×7×64
```

### 3. **Efficient Data Pipeline**

- Custom dataset classes handle CSV parsing
- Automatic tensor conversion and normalization
- Memory-efficient batch processing

### 4. **Robust Training Loop**

- Error handling and device management
- Progress tracking and performance metrics
- Automatic model evaluation

## Project Structure

```
CNN-MNIST-Digit-Recognition-With-Custom-Dataset-Class/
├── CNN-MNIST-Digit-Recognition-With-Custom-Dataset-Class.ipynb
├── .idea/
│   ├── .gitignore
│   ├── CNN MNIST Digit Recognition With Custom Dataset Class.iml
│   ├── misc.xml
│   ├── modules.xml
│   ├── vcs.xml
│   └── inspectionProfiles/
└── README.md
```

## Usage

### Prerequisites

```bash
pip install torch torchvision pandas numpy
```

### Running the Model

1. **Data Preparation**: Place MNIST CSV files in specified paths
2. **Training**: Execute notebook cells sequentially
3. **Evaluation**: Automatic prediction generation on test set

### Key Code Workflow

#### 1. Data Loading

```python
train_dataset = CustomTrainDataset('/path/to/train.csv', ToTensor())
test_dataset = CustomTestDataset('/path/to/test.csv', ToTensor())
```

#### 2. Model Training

```python
model = DRNN().to(device)
for epoch in range(30):
    train_epoch_loss, train_epoch_acc = train_one_epoch(...)
```

#### 3. Prediction

```python
def eval(dataloader, model, loss_fn, path):
    # Generates submission CSV with predictions
```

## Technical Highlights

### 1. **Custom Architecture Design**

- Tailored specifically for 28×28 MNIST images
- Optimized feature map progression
- Efficient parameter utilization

### 2. **Advanced Preprocessing**

```python
# Data transformation pipeline:
flat_array → reshape(28,28,1) → ToTensor() → normalize
```

### 3. **Memory Management**

- Efficient batch processing
- GPU memory optimization
- Automatic device detection

### 4. **Production-Ready Pipeline**

- Modular dataset classes
- Configurable hyperparameters
- Automated submission generation

## Model Performance

### Training Characteristics:

- **Convergence**: Rapid convergence within first 10 epochs
- **Stability**: Consistent training with batch normalization
- **Efficiency**: Fast training due to optimized architecture

### Output Generation:

- **Submission format**: CSV with ImageId and Label columns
- **Prediction confidence**: Softmax probabilities
- **Class mapping**: Direct integer output (0-9)

## Future Improvements

1. **Data Augmentation**:

   - Random rotations and translations
   - Elastic deformations
   - Noise injection

2. **Architecture Enhancements**:

   - Dropout layers for regularization
   - Skip connections (ResNet-style)
   - Attention mechanisms

3. **Training Optimizations**:

   - Learning rate scheduling
   - Early stopping
   - Cross-validation

4. **Advanced Techniques**:
   - Ensemble methods
   - Test-time augmentation
   - Model distillation

## Dependencies

- **PyTorch**: Deep learning framework
- **pandas**: CSV data handling
- **numpy**: Numerical operations
- **torch.utils.data**: Dataset and DataLoader utilities

## Key Achievements

- Custom dataset implementation from scratch
- Well-designed CNN architecture
- Comprehensive batch normalization
- Real-time training monitoring
- Automated prediction pipeline
- GPU acceleration support

## Learning Outcomes

This project demonstrates:

- **Custom Dataset Creation**: Building PyTorch datasets from CSV files
- **CNN Architecture Design**: Systematic feature extraction progression
- **Batch Normalization**: Implementation and benefits
- **Training Pipeline**: Complete end-to-end workflow
- **Performance Monitoring**: Real-time metrics and logging

---
