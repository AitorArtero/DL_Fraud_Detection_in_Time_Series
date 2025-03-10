# Credit Card Fraud Detection

This project implements a deep learning system for detecting fraudulent transactions in credit card data. Two different approaches are presented to handle the typical class imbalance in this type of problem: class weighting and upsampling.

## Description

Credit card fraud detection is a critical issue for financial institutions. In this project, we develop neural network-based solutions to identify fraudulent transactions, which typically represent less than 1% of all transactions.

The project uses a standard fraud detection dataset containing transactions made by European cardholders during a weekrnd period.

## Approaches

This project presents two main strategies to address class imbalance:

### 1. Weighted approach

- Implemented in `Weighted_approach-Fraud_Detection_in_Time_Series.py`
- Uses weights in the loss function (BCEWithLogitsLoss) to give greater importance to the minority class (fraudulent transactions)
- Maintains the original distribution of the data

### 2. Upsampling approach

- Implemented in `Upsampling_approach-Fraud_Detection_in_Time_Series.py`
- Generates synthetic samples of the minority class (frauds) until balanced with the majority class
- Modifies the original dataset distribution to have balanced class representation

## Model Architecture

After exploring several architecture options, we chose a simple yet effective Autoencoder-based architecture:

```Markdown
AnomalyDetector:

Linear layer (30 → 16)
ReLU activation
Linear layer (16 → 24)
ReLU activation
Dropout (0.5)
Linear layer (24 → 20)
ReLU activation
Linear layer (20 → 24)
ReLU activation
Linear layer (24 → 1)
```


Initially, we considered more complex architectures:
- **RAE (Robust Autoencoders)**: To better handle outliers in the data
- **VAE (Variational Autoencoders)**: For its probabilistic approach to latent space representation
- **LSTM (Long Short-Term Memory)**: To capture potential temporal patterns in the transactions

However, the performance of the simpler autoencoder was already excellent, with AUC scores above 0.95, making the additional complexity unnecessary for this application.

## Evaluation Metrics

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Area Under ROC Curve (AUC)
- Confusion matrix

## Preprocessing

Data preprocessing includes:
- Normalization of numerical features (Amount, Time, V1-V28)
- Stratified split into training and validation sets
- Data visualization for fraud vs. non-fraud distributions

## Key Findings

As detailed in our presentation:

1. **Class Imbalance**: The dataset is highly imbalanced with only 0.17% fraudulent transactions, making this a challenging classification problem.

2. **Approach Comparison**:
   - The weighted approach achieved high and balanced result on precision and recall
   - The upsampling approach improved recall at a slight cost to precision
   - Both approaches achieved AUC scores above 0.95

3. **Feature Importance**: Some of the anonymized features (V1-V28) showed strong correlation with fraudulent transactions, particularly V14, V12, and V17.

4. **Temporal Pattern Analysis**: No significant temporal patterns were detected in fraudulent activities, justifying our decision not to use LSTM. This might be a result of the dataset not containing enough temporal information.

## Requirements
```bash
torch
pandas
numpy
scikit-learn
matplotlib
```

## Usage

To run the weighted approach:
```bash
python Weighted_approach-Fraud_Detection_in_Time_Series.py
```

To run the upsampling approach:
```bash
python Upsampling_approach-Fraud_Detection_in_Time_Series.py
```

## Implementation Details
* Early Stopping: Implemented to prevent overfitting (patience of 5 epochs)
* Model Saving: Best model is saved based on validation loss
* Visualization: Comprehensive visualization of data distribution, training progress, and results
* Performance Tracking: Detailed logging of metrics during training and evaluation phases

## Notes
* The dataset should be in the same folder as the scripts with the name "creditcard.csv"
* Trained models are automatically saved as "best_model.pt"
* Both approaches use the same validation set for fair comparison

