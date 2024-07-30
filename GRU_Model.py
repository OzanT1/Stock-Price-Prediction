import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time


# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

def load_data(file_name):
    df = pd.read_csv(file_name, parse_dates=['Date'], index_col='Date')
    return df['Close'].values

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])
        return out

# Parameters
input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1
num_epochs = 150
learning_rate = 0.01

# Load data
data = load_data('MSFT.csv')

# Preprocess data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# Split data into training and testing sets
train_data, test_data = train_test_split(data_normalized, test_size=0.2, shuffle=False)

# Prepare data for training and testing
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Create model
model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

train_loss = []
test_loss = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

    # Validation
    model.eval()
    with torch.inference_mode():
        y_predicted = model(X_test)
        test_loss.append(criterion(y_predicted, y_test).item())


plt.plot(np.array(range(len(train_loss))), np.array(train_loss), label="Train Loss")
plt.plot(np.array(range(len(test_loss))), np.array(test_loss), label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

end_time = time.time()
plt.title(f'GRU Loss  |  {num_epochs} epochs  |  Training Time: {(end_time - start_time):.2f} seconds')

print(f"Training took {end_time - start_time} seconds.")

# Prediction
model.eval()
with torch.inference_mode():
    train_predicted = model(X_train).cpu().numpy()
    test_predicted = model(X_test).cpu().numpy()
    train_predicted = scaler.inverse_transform(train_predicted)
    test_predicted = scaler.inverse_transform(test_predicted)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(seq_length, len(data)), data[seq_length:], label='Actual')

# Adjusting indices for plotting
train_len = len(train_predicted)
train_indices = range(seq_length, train_len + seq_length)
test_indices = range(train_len + seq_length, train_len + seq_length + len(test_predicted))

plt.plot(train_indices, train_predicted, label='Train Predicted')
plt.plot(test_indices, test_predicted, label='Test Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with GRU')
plt.legend()
plt.show()
