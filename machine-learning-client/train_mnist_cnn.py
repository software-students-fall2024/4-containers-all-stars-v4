""" Module for training MNIST CNN model"""

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from model import CNNModel, InvertGrayscale


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


def main():
    """Main method to train and save model"""
    model = CNNModel()

    loader = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), InvertGrayscale()]),
        ),
        batch_size=64,
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

    char_count = 100

    print("\n" + "=" * char_count + "\n")
    print("DEBUG: Dataset Characteristics")
    print(f" * mean: {float(mean):.4f}")
    print(f" * std: {float(std):.4f}")
    print("\n" + "=" * char_count + "\n")

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

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transform
        ),
        batch_size=64,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transform
        ),
        batch_size=1000,
        shuffle=False,
    )

    # device specifications
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # hyperparameters
    # epochs = 10
    # learning_rate = 0.001

    # init model, criterion, and optimizer
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("DEBUG: \x1B[4mModel Training and Evaluation\x1B[0m" + "\n")
    # train test loop
    for epoch in range(1, 10 + 1):
        print(f"Epoch {epoch}/{10}")
        train(
            model,
            train_loader,
            optim.Adam(model.parameters(), lr=0.001),
            criterion,
            device,
        )
        test(model, test_loader, criterion, device)
    print("\n" + "=" * char_count + "\n")

    # save model
    torch.save(model.state_dict(), "mnist-cnn.pth")
    print("DEBUG: \x1B[1mModel Saved!\x1B[0m")
    print("\n" + "=" * char_count)


if __name__ == "__main__":
    main()
