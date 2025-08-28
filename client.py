
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Client:
    def __init__(self, client_id, model, data, labels, learning_rate=0.01, batch_size=32):
        self.client_id = client_id
        self.model = model
        self.data = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=1):
        self.model.train()
        total_loss = 0.0
        for epoch in range(epochs):
            for inputs, targets in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return self.model.state_dict(), total_loss / len(self.dataloader)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return correct / total

class SimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Example Usage
    input_dim = 10
    num_classes = 2
    num_clients = 3
    data_size_per_client = 100

    # Global model (initialized once)
    global_model = SimpleModel(input_dim, num_classes)

    clients = []
    for i in range(num_clients):
        # Simulate diverse client data
        client_data = torch.randn(data_size_per_client, input_dim).numpy()
        client_labels = torch.randint(0, num_classes, (data_size_per_client,)).numpy()
        
        # Each client gets a copy of the global model to start
        client_model = SimpleModel(input_dim, num_classes)
        client_model.load_state_dict(global_model.state_dict())
        
        client = Client(f"client_{i+1}", client_model, client_data, client_labels)
        clients.append(client)

    print("Initial evaluation of global model on client 1:", clients[0].evaluate())

    # Federated Learning Round
    print("\nStarting federated training...")
    client_updates = []
    for client in clients:
        print(f"Client {client.client_id} training...")
        state_dict, loss = client.train(epochs=1)
        client_updates.append(state_dict)
        print(f"Client {client.client_id} finished training with loss: {loss:.4f}")

    # Aggregate updates (simple averaging)
    print("\nAggregating client updates...")
    aggregated_state_dict = client_updates[0].copy()
    for key in aggregated_state_dict.keys():
        for i in range(1, len(client_updates)):
            aggregated_state_dict[key] += client_updates[i][key]
        aggregated_state_dict[key] = aggregated_state_dict[key] / len(client_updates)

    global_model.load_state_dict(aggregated_state_dict)
    print("Global model updated.")

    print("\nFinal evaluation of global model on client 1:", clients[0].evaluate())

# Commit timestamp: 2025-08-28 00:00:00 - 236
