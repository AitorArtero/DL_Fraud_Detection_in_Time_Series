import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Select the device to run the model, GPU as far as is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Scale the amount column 
sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))
time = df['Time'].values
df['Time'] = sc.fit_transform(time.reshape(-1, 1))

# Scale all V1-V28 features
columns_to_scale = ['V{}'.format(i) for i in range(1, 29)]
for column in columns_to_scale:
    values = df[column].values
    df[column] = sc.fit_transform(values.reshape(-1, 1))

# Create visualizations of fraud vs non-fraud
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))

# Filter Class=0 data and plot
class_0 = df[df['Class']==0]
len_class0 = len(df[df['Class']==0])
ax0.scatter(class_0['Time'], class_0['Amount'])
ax0.set_xlabel('Time')
ax0.set_ylabel('Amount')
ax0.set_title(f"Scatterplot no fraud, len: {len_class0}")

# Filter Class=1 data and plot
class_1 = df[df['Class']==1]
len_class1 = len(df[df['Class']==1])
ax1.scatter(class_1['Time'], class_1['Amount'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Amount')
ax1.set_title(f"Scatterplot fraud, len: {len_class1}")

plt.show()

# Combined visualization
class_0 = df[df['Class']==0]
class_1 = df[df['Class']==1]
size_NoFraud = len(df[df['Class']==0])
size_Fraud = len(df[df['Class']==1])
plt.scatter(class_0.Time, class_0['Amount'], color='blue', label=f"No Fraud, {size_NoFraud} values")
plt.scatter(class_1.Time, class_1['Amount'], color='orange', label=f"Fraud, {size_Fraud} values")

plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Scatterplot Fraud vs No Fraud')
plt.legend()
plt.show()

# Define the dataset class
class AnomalyDetectionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # features
        self.y = y  # target (1 if fraud, 0 if not)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        targets = self.y[idx]
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "target": torch.tensor(targets, dtype=torch.float32)
        }

# Split the dataset (stratify to ensure balanced representation)
train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.Class.values)

# Prepare training dataset
train_dfr = train_df
train_df = train_df.to_numpy()
X = train_df[:, :-1]
y = train_df[:, -1]
train_dataset = AnomalyDetectionDataset(X, y)

# Prepare validation dataset
val_df = val_df.to_numpy()
X = val_df[:, :-1]
y = val_df[:, -1]
val_dataset = AnomalyDetectionDataset(X, y)

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Neural network for anomaly detection
class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.layer1 = nn.Linear(30, 16)
        self.layer2 = nn.Linear(16, 24)
        self.dropout = nn.Dropout(0.5)
        self.layer3 = nn.Linear(24, 20)
        self.layer4 = nn.Linear(20, 24)
        self.layer5 = nn.Linear(24, 1)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.layer5(x)
        return x

model = AnomalyDetector()

# Training parameters
learning_rate = 0.001
num_epochs = 10

# Set up model with class weights to address imbalance
no_frauds = len(train_dfr[train_dfr['Class'] == 1])
weights = no_frauds/len(train_dfr)
class_weights = torch.tensor([weights*2], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_loss = 0
print_plot_steps = 5700
loss_per_epoch = []
mean_per_epoch = 0
step_count = 0
all_losses_list = []
val_loss = []
best_valid_loss = float('inf')

model.train()
early_stop = False
patience = 5
wait = 0

for epoch_i in range(num_epochs):
    if early_stop:
        break

    model.train()
    mean_per_epoch = 0
    if epoch_i != 0:
        print("--------------------------")
    print(f"Epoch num {epoch_i}\n")
    
    for train_data in train_loader:
        inputs, real_class = train_data['features'], train_data['target'].unsqueeze(-1).to(device)
        outputs = model(inputs).to(device)
        loss = criterion(outputs, real_class)
        total_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step_count = step_count + len(train_data["target"])
        mean_per_epoch += total_loss
        
        if(step_count % print_plot_steps == 0):
            avg_loss = total_loss
            print(f"The loss at step {step_count} is {round(avg_loss, 10)}")
            all_losses_list.append(avg_loss)
            total_loss = 0
            
    loss_per_epoch.append(mean_per_epoch/len(train_loader))
    print(f"Training mean per epoch {mean_per_epoch/len(train_loader)}")
    
    # Validation loop
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for valid_data in val_loader:
            inputs, real_class = valid_data['features'], valid_data['target'].unsqueeze(-1).to(device)
            outputs = model(inputs).to(device)
            valid_loss += criterion(outputs, real_class).item()
            
        val_loss.append(valid_loss/len(val_loader))
        print(f"Validation mean per epoch {valid_loss/len(val_loader)}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                early_stop = True
                print(f"No improvement in validation loss after {patience} epochs. Training stopped.")

# Plot training and validation loss
plt.figure()
plt.plot(loss_per_epoch, label='training per epoch')
plt.plot(val_loss, label='validation per epoch')
plt.legend()
plt.show()

# Evaluate the model
correct = 0
total = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

with torch.no_grad():
    for data in val_loader:
        inputs, labels = data['features'], data['target'].unsqueeze(-1)
        outputs = model(inputs)
        predicted = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

# Calculate performance metrics
accuracy = 100 * (correct / total)
precision = 100 * (true_positives / (true_positives + false_positives))
recall = 100 * (true_positives / (true_positives + false_negatives))
f1_score = 2 * (precision * recall) / (precision + recall)

# ROC AUC evaluation
with torch.no_grad():
    y_true = []
    y_scores = []
    for data in val_loader:
        inputs, labels = data['features'], data['target'].unsqueeze(-1)
        outputs = model(inputs)
        predicted_probs = torch.sigmoid(outputs)
        y_true += labels.numpy().tolist()
        y_scores += predicted_probs.numpy().tolist()

auc = roc_auc_score(y_true, y_scores)

# Print results
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1_score:.2f}%")
print(f"Confusion matrix:\n(TP, FP\nFN, TN)\n{true_positives} {false_positives}\n{false_negatives} {true_negatives}")
print(f"AUC: {auc:.2f}")