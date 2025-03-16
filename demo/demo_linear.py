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

"""
Make a way to convert torch tensor to match tensor. 
Make Optimizer
Make Dataloader
Implement Conv2d in C
Implement Max Pool + Ave pool + gradient
implement testing
Implement shuffle
Implement some sort of benchmarking/performance tests
"""


class IrisClassifierMatch(match.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.linear1 = match.nn.Linear(num_input_features, 256)
        self.relu1 = match.nn.Sigmoid()

        self.linear2 = match.nn.Linear(256, 128)
        self.relu2 = match.nn.ReLU()

        self.linear3 = match.nn.Linear(128, 64)
        self.relu3 = match.nn.ReLU()

        self.output = match.nn.Linear(64, num_output_features)
        self.softmax = match.nn.Softmax(1)

    def forward(self, x: match.Tensor) -> match.Tensor:
        o1 = self.relu1(self.linear1(x))

        o2 = self.relu2(self.linear2(o1))

        o3 = self.relu3(self.linear3(o2))

        return self.softmax(self.output(o3))


def to_tensor(match_tensor) -> torch.Tensor:
    """Converts a match tensor to a Pytorch tensor.

    Args:
        match_tensor (tensor.Tensor): The custom match tensor to convert
        requires_grad (bool, optional): If True, the resulting PyTorch tensor will require grad. Defaults to False.
        get_grad (bool, optional): If True, convert the grad of the provided match tensor to convert. Defaults to False.

    Returns:
        torch.Tensor: The equivalent PyTorch implementation of the provided match Match tensor.
    """
    match_tensorbase = match_tensor.data

    if match_tensorbase.ndim == 0:
        torch_tensor = torch.tensor(match_tensorbase.item()).float()
    else:
        torch_tensor = (
            torch.Tensor(
                match_tensorbase._raw_data
            )  # Gets the raw 1D array storing the data of the TensorBase object.
            .float()
            .reshape(tuple(match_tensorbase.size))
        )
    return torch_tensor


def data_to_match_tensor(data, p=1):
    assert p > 0 and p <= 1
    total_instances = data.data.shape[0]
    X = data.data.reshape((total_instances, 784))
    Y = data.targets
    instances_to_load = int(total_instances * p)
    X_match = TensorBase((instances_to_load, 784))
    Y_match = TensorBase((instances_to_load, 10))
    Y_match.fill_(0)
    sequence = list(range(instances_to_load))
    random.shuffle(sequence)
    for i in tqdm(range(instances_to_load), desc="Loading training data", leave=False):
        r = sequence[i]
        for c in range(784):
            X_match[r, c] = X[r, c].item()
        Y_match[r, Y[r].item()] = 1
    return match.Tensor(X_match), match.Tensor(Y_match)

def arg_max(values):
    if not values:
        raise ValueError("Cannot find argmax of an empty sequence")
    
    max_index = 0
    max_value = values[0]
    
    for i, value in enumerate(values):
        if value > max_value:
            max_value = value
            max_index = i
            
    return max_index

if __name__ == "__main__":
    print("Loading train data...")
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    print("Loading test data...")
    testing_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    print("Transforming data into Match compatable format...")
    X, Y = data_to_match_tensor(training_data, p=0.1)
    X_test, Y_test = data_to_match_tensor(testing_data, p=1)
    num_instances, num_input_features, num_output_features = (
        X.shape[0],
        X.shape[1],
        Y.shape[1],
    )
    print(f"Number of instances: {num_instances}")
    print(f"Number of input features: {num_input_features}")
    print(f"Number of output features: {num_output_features}")

    model = IrisClassifierMatch(num_input_features, num_output_features)
    epochs = 30
    lrr = 0.04
    lossfn = match.nn.MultiClassCrossEntropyLoss()
    batch_size = 64

    lr_loss_map = {lrr: []}
    lr_test_loss_map = {lrr: []}

    data_training_sequence = list(range(num_instances))
    for lr in lr_loss_map.keys():
        for epoch in trange(epochs, desc="Epochs"):
            acc_dict = {k:[0,0] for k in range(10)}
            total = 0.0
            correct = 0.0
            for i in range(X_test.shape[0]):
                total += 1
                prediction = model(X_test[i]).data._raw_data
                target = Y_test[i].data._raw_data
                prediction_arg_max = arg_max(prediction)
                target_arg_max = arg_max(target)
                correct += int(prediction_arg_max==target_arg_max)
                acc_dict[target_arg_max][0] += int(prediction_arg_max==target_arg_max)
                acc_dict[target_arg_max][1] += 1
            lr_test_loss_map[lr].append(correct/total)

            print(f"Epoch {epoch} accuracy: {correct/total}")
            print(f"Epoch {epoch} digit accuracies: {dict({k: round(v[0]/v[1], 2) for k, v in acc_dict.items()})}")
            random.shuffle(data_training_sequence)
            progress_bar = tqdm(
                enumerate(data_training_sequence), desc=f"Epoch {epoch}", leave=False
            )
            loss = 0
            for i, instance in progress_bar:
                prediction = model(X[instance])
                target = Y[instance]

                loss += lossfn(prediction, target)

                if i % batch_size == 0 and i != 0:
                    loss /= batch_size
                    lr_loss_map[lr].append(loss.data.item())
                    # Backpropagation
                    loss.backward()
                    for param in model.parameters():
                        param.data -= lr * param.grad
                    model.zero_grad()

                    # Update progress bar with current loss
                    progress_bar.set_postfix(loss=f"{loss.data.item():.4f}")
                    loss = 0
        acc_dict = {k:[0,0] for k in range(10)}
        total = 0.0
        correct = 0.0
        for i in range(X_test.shape[0]):
            total += 1
            prediction = model(X_test[i]).data._raw_data
            target = Y_test[i].data._raw_data
            prediction_arg_max = arg_max(prediction)
            target_arg_max = arg_max(target)
            correct += int(prediction_arg_max==target_arg_max)
            acc_dict[target_arg_max][0] += int(prediction_arg_max==target_arg_max)
            acc_dict[target_arg_max][1] += 1
        lr_test_loss_map[lr].append(correct/total)

        print(f"Epoch {epoch} accuracy: {correct/total}")
        print(f"Epoch {epoch} digit accuracies: {dict({k: round(v[0]/v[1], 2) for k, v in acc_dict.items()})}")

            

    # Define colors for each line
    colors = ["blue", "green", "orange", "purple", "brown", "pink"]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Iterate through the lr_loss_map and plot each line
    for i, (lr, losses) in enumerate(lr_loss_map.items()):
        plt.plot(losses, label=f"LR={lr}", color=colors[i % len(colors)])

    # Add labels and title
    plt.xlabel("Epochs/Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch/Iterations for Different Learning Rates")

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        "loss_vs_learning_rate.png"
    )  # You can change the filename and extension (e.g., .jpg, .pdf)

    plt.cla()

    # Create the plot
    plt.figure(figsize=(12, 6))


    plt.plot(lr_test_loss_map[lrr], label=f"LR={lrr}", color="red")

    # Add labels and title
    plt.xlabel("Epochs/Iterations")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Epoch/Iterations for Different Learning Rates")

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(
        "test_accuracy_vs_learning_rate.png"
    )  # You can change the filename and extension (e.g., .jpg, .pdf)
