import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self.model = set_parameters(self.model, parameters)
        # Simulate training
        print("Client training...")
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.model = set_parameters(self.model, parameters)
        # Simulate evaluation
        loss, accuracy = 0.1, 0.9
        print("Client evaluating...")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

# Start Flower client
if __name__ == "__main__":
    print("Federated Learning Client starting...")
    # Mock data loaders and model
    model = Net()
    trainloader = [(torch.randn(1, 1, 28, 28), torch.randint(0, 10, (1,))) for _ in range(10)]
    valloader = [(torch.randn(1, 1, 28, 28), torch.randint(0, 10, (1,))) for _ in range(5)]
    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient(model, trainloader, valloader))
