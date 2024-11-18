""" Module for training MNIST CNN model"""

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model import CNNModel, InvertGrayscale

CHAR_COUNT = 100


def train(model, train_loader, optimizer, criterion, device):
    """Method for training model"""
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # compute loss
        loss = criterion(output, target)

        # backward pass & optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model(model, test_loader, criterion, device):
    """Method for testing model"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # forward pass
            output = model(data)

            # compute loss
            test_loss += criterion(output, target).item()

            # get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy


def compute_dataset_statistics(dataset, batch_size=64):
    """Compute mean and standard deviation for the dataset."""
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size
        # flatten the spatial dimensions
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return float(mean), float(std)


def get_transforms(mean, std):
    """Define train and test data transformations."""
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),  # randomly rotate images
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1)
            ),  # randomly translate images
            transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),  # randomly scale images
            transforms.ToTensor(),
            InvertGrayscale(),
            transforms.Normalize((mean,), (std,)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            InvertGrayscale(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    return train_transform, test_transform


def get_data_loaders(batch_size=64, dataset_path="./data"):
    """Return data loaders for train and test datasets."""
    dataset = datasets.MNIST(
        root=dataset_path,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), InvertGrayscale()]),
    )
    mean, std = compute_dataset_statistics(dataset, batch_size=batch_size)
    train_transform, test_transform = get_transforms(mean, std)

    train_dataset = datasets.MNIST(
        root=dataset_path, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=dataset_path, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size * 16,
        shuffle=False,
    )
    return train_loader, test_loader


def save_model(model, save_path):
    """Save the trained model to the specified path"""
    torch.save(model.state_dict(), save_path)


def main(
    epochs=10,
    learning_rate=0.001,
    batch_size=64,
    dataset_path="./data",
    model_save_path="mnist-cnn.pth",
):
    """Main method to train and save the MNIST CNN model"""
    # Initialize the model
    model = CNNModel()

    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=batch_size, dataset_path=dataset_path
    )

    # Device setup
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"DEBUG: Using device: {device}")

    # Initialize training components
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation
    print("DEBUG: \x1B[4mModel Training and Evaluation\x1B[0m\n")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        evaluate_model(model, test_loader, criterion, device)
    print("\n" + "=" * CHAR_COUNT + "\n")

    # Save the model
    if model_save_path:
        save_model(model, model_save_path)
        print("\n" + "=" * CHAR_COUNT)
        print(f"DEBUG: \x1B[1mModel Saved to {model_save_path}!\x1B[0m")
        print("=" * CHAR_COUNT)


if __name__ == "__main__":
    main()
