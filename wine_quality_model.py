import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
# data = pd.read_csv(url, sep=';')
data = pd.read_csv(r"C:\Users\joaqu\Downloads\winequality-white.csv", sep=';')# for the downloaded file

# Split data into features (X) and target variable (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
class WineQualityModel(nn.Module):
    def __init__(self):
        super(WineQualityModel, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model(X_train, y_train, X_test, y_test, learning_rate, batch_size, epochs):
    model = WineQualityModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    print("training model, please wait...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test.view(-1, 1))
            test_losses.append(test_loss.item())
    print("done!")
    return model, train_losses, test_losses

# Train the model
learning_rate = 0.001
batch_size = 32
epochs = 200
trained_model, train_losses, test_losses = train_model(X_train, y_train, X_test, y_test, learning_rate, batch_size, epochs)

# Evaluate the model
with torch.no_grad():
    trained_model.eval()
    y_pred = trained_model(X_test)
    mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
    print("MSE:", mse)

# Plot training loss progress
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training Loss Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


