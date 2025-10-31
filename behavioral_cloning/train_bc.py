import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

#Load human data
with open("data/human_data.pkl", "rb") as f:
    human_data = pickle.load(f)

obs = np.array([d[0] for d in human_data])
actions = np.array([d[1] for d in human_data], dtype=np.float32)

#Normalize images --> More stable learning
obs = obs / 255.0

#Simple CNN for Bahvioral Clonning - Network to learn by imitation
class CNNPolicyClone(nn.Module):
    def __init__(self, action_dim, input_shape=(96, 96, 3)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )

        #Get the output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape).permute(0, 3, 1, 2)
            conv_out_size = self.conv(dummy).reshape(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float()  #Change image HWC → CHW as PyTorch needs
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)   
        return self.fc(x)


action_dim = actions.shape[1]
model = CNNPolicyClone(action_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

obs_tensor = torch.tensor(obs, dtype=torch.float32)
actions_tensor = torch.tensor(actions, dtype=torch.float32)

#Training
batch_size = 64
epochs = 2
dataset_size = len(human_data)

for epoch in range(epochs):
    idx = np.random.permutation(dataset_size)
    for i in range(0, dataset_size, batch_size):
        batch_idx = idx[i:i+batch_size]
        x = obs_tensor[batch_idx]
        y = actions_tensor[batch_idx]
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")


#Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/bc_model.pth")
print("BC model trained and saved in 'models/'")
