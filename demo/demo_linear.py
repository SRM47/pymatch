import random
import requests
import matplotlib.pyplot as plt
import match
import match.nn
import torch
import torch.nn
from match.tensorbase import TensorBase
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange
from math import prod


class MNISTClassifier(match.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.linear1 = match.nn.Linear(num_input_features, 128)
        self.relu1 = match.nn.Sigmoid()

        self.linear2 = match.nn.Linear(128, 64)
        self.relu2 = match.nn.ReLU()

        self.linear3 = match.nn.Linear(64, 32)
        self.relu3 = match.nn.ReLU()

        self.output = match.nn.Linear(32, num_output_features)
        self.softmax = match.nn.Softmax(1)

    def forward(self, x: match.Tensor) -> match.Tensor:
        o1 = self.relu1(self.linear1(x))

        o2 = self.relu2(self.linear2(o1))

        o3 = self.relu3(self.linear3(o2))

        return self.softmax(self.output(o3))


def torch_to_match(
    X: torch.Tensor, Y: torch.Tensor, p: float = 1, loading_msg: str | None = None
):
    assert p > 0 and p <= 1
    num_total_instances, num_features = X.data.shape[0], prod(X.data.shape[1:])
    num_instances = int(num_total_instances * p)

    X = X.data.reshape((num_total_instances, num_features))
    Y = Y.data

    X_match = TensorBase((num_instances, num_features))
    Y_match = TensorBase((num_instances, 10))
    Y_match.fill_(0)

    load_sequence = list(range(num_instances))
    random.shuffle(load_sequence)
    for r in tqdm(load_sequence, desc=loading_msg, leave=True):
        for c in range(num_features):
            X_match[r, c] = X[r, c].item()
        Y_match[r, Y[r].item()] = 1

    return match.Tensor(X_match), match.Tensor(Y_match)


def arg_max(values):
    if not values:
        raise ValueError("Cannot find argmax of an empty sequence")

    max_index, max_value = 0, values[0]

    for i, value in enumerate(values):
        if value > max_value:
            max_value = value
            max_index = i

    return max_index


def test_model(
    model: match.nn.Module,
    lossfn: match.nn.Module,
    X_test: match.Tensor,
    Y_test: match.Tensor,
):
    accuracy = {k: [0, 0] for k in range(Y_test.shape[1])}
    num_instances = X_test.shape[0]
    total_correct = 0
    loss = 0
    for i in range(num_instances):
        prediction = model(X_test[i])
        target = Y_test[i]

        loss += lossfn(prediction, target)

        prediction_arg_max = arg_max(prediction.data._raw_data)
        target_arg_max = arg_max(target.data._raw_data)

        correct = int(prediction_arg_max == target_arg_max)
        total_correct += correct
        accuracy[target_arg_max][0] += correct
        accuracy[target_arg_max][1] += 1
    return total_correct / num_instances, accuracy, (loss / num_instances).item()


if __name__ == "__main__":
    print("Downloading train data...")
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    print("Downloading test data...")
    testing_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    print("Transforming data into Match compatable format...")
    X, Y = torch_to_match(
        training_data.data,
        training_data.targets,
        p=0.1,
        loading_msg="Loading training data...",
    )
    X_test, Y_test = torch_to_match(
        testing_data.data,
        testing_data.targets,
        p=0.1,
        loading_msg="Loading testing data...",
    )
    num_instances, num_input_features, num_output_features = (
        X.shape[0],
        X.shape[1],
        Y.shape[1],
    )
    print(f"Number of instances: {num_instances}")
    print(f"Number of input features: {num_input_features}")
    print(f"Number of output features: {num_output_features}")

    model = MNISTClassifier(num_input_features, num_output_features)
    epochs = 10
    learning_rate = 0.04
    lossfn = match.nn.MultiClassCrossEntropyLoss()
    batch_size = 128

    train_losses = []
    test_losses = []
    test_accuracy = []

    data_training_sequence = list(range(num_instances))
    for epoch in range(epochs):
        local_accuracy, accuracy_dict, test_loss = test_model(
            model, lossfn=lossfn, X_test=X_test, Y_test=Y_test
        )
        test_accuracy.append(local_accuracy)
        test_losses.append(test_loss)

        random.shuffle(data_training_sequence)
        progress_bar = tqdm(
            enumerate(data_training_sequence), desc=f"Epoch {epoch}", leave=True
        )
        loss = 0
        for iteration, instance in progress_bar:
            prediction = model(X[instance])
            target = Y[instance]

            loss += lossfn(prediction, target)

            if iteration % batch_size == 0 and iteration != 0:
                loss /= batch_size
                train_losses.append(loss.data.item())

                # Backpropagation
                loss.backward()
                for param in model.parameters():
                    param.data -= learning_rate * param.grad
                model.zero_grad()

                # Update progress bar with current loss
                progress_bar.set_postfix(loss=f"{loss.data.item():.4f}")
                loss = 0

    local_accuracy, accuracy_dict, test_loss = test_model(
        model, lossfn=lossfn, X_test=X_test, Y_test=Y_test
    )
    test_accuracy.append(local_accuracy)
    test_losses.append(test_loss)

    # Calculate how many training steps per epoch
    steps_per_epoch = num_instances / batch_size

    # Create x-coordinates for both losses
    train_x = [i/steps_per_epoch for i in range(len(train_losses))]  # Fraction of epochs
    test_x = list(range(len(test_losses)))  # Whole epochs

    # Create figure and primary y-axis
    fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()

    # Plot losses on the primary y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='black')
    ax1.plot(train_x, train_losses, label="Train Loss", color="green")
    ax1.plot(test_x, test_losses, label="Test Loss", color="orange")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # Create secondary y-axis
    ax2 = ax1.twinx()

    # Plot accuracy on the secondary y-axis
    ax2.set_ylabel('Accuracy', color='blue')
    ax2.plot(test_x, test_accuracy, label="Test Accuracy", color="blue", linestyle='--')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')

    # Add grid and adjust layout
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig("plot.png")
