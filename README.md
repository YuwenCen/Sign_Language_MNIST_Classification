## Sign Language MNIST Classification: From MLP to CNN

This project focuses on classifying hand gesture images from the Sign Language MNIST dataset using deep learning models.

Starting from a simple Multi-Layer Perceptron (MLP), we progressively improved model performance by introducing Convolutional Neural Networks (CNNs) to better capture spatial features in image data.

### Dataset

Dataset: Sign Language MNIST with 27,455 training samples and 7,172 test samples. 

Input: 28×28 grayscale images

### Methodology
1. Data Preprocessing

Normalized pixel values to [0, 1], train/validation split (85/15, stratified), and reshaped data for CNN input (28×28×1)

2. Model Development
- MLP: flattened 784-dimensional vectors with 2 hidden layers (256 → 128 units) and ReLU + Dropout.

Limitation: cannot capture spatial structure

- Baseline CNN: 2 convolutional layers + max pooling

Learns local spatial features and improved stability and generalization

- Improved CNN: Increased network depth (additional conv layer)

Learns more complex hierarchical features

### Results
Model	Test Accuracy
MLP	77.65%
CNN (baseline)	91.30%
CNN (improved)	94.69%


### Key Insights

MLP struggles with image data due to lack of spatial awareness, CNN significantly improves performance by capturing local patterns. 

Increasing model depth leads to further gains, but with diminishing returns

Error Analysis: Confusion matrix reveals errors concentrated among visually similar gestures


### Tech Stack

- PyTorch, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
