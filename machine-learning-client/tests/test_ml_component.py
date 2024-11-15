"""Used to test machine learning component."""

import pytest
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from train_mnist_cnn import (
    train,
    evaluate_model,
    get_data_loaders,
    get_transforms,
    compute_dataset_statistics,
)
from model import CNNModel, InvertGrayscale


class Tests:
    """Test functions"""

    def test_sanity_check(self):
        """
        Test debugging... making sure that we can run a simple test that always passes.
        Note the use of the example_fixture in the parameter list -
        any setup and teardown in that fixture will be run before
        and after this test function executes.
        """
        expected = True
        actual = True
        assert actual == expected, "Expected True to be equal to True!"

    def model(self):
        """Provide a simple CNN model instance."""
        return CNNModel()

    def test_instantiation(self):
        """Test that the CNNModel can be instantiated without errors."""
        assert isinstance(self.model(), nn.Module)

    def test_forward(self):
        """Test that the CNNModel can make a forward pass without errors."""
        model = self.model()
        result = model(torch.randn(1, 1, 28, 28))
        assert result.shape == (1, 10), f"Unexpected output shape: {result.shape}"

    def test_evaluation(self):
        """Test that the CNNModel can perform inference in evaluation mode."""
        model = self.model()
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 1, 28, 28))
        assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"

    def test_invert_grayscale(self):
        """Test the InvertGrayscale layer."""
        layer = InvertGrayscale()
        result = layer(torch.rand(1, 1, 28, 28) * 255)

        # Make sure the output is within the [0, 255] range after inversion
        assert result.max() <= 255, "Maximum value exceeds 255 after inversion."
        assert result.min() >= 0, "Minimum value is below 0 after inversion."

    def test_evaluate_model(self):
        """Test the evaluate_model function."""
        model = self.model()
        test_loader = DataLoader(
            TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))),
            batch_size=8,
        )

        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        model.to(device)

        accuracy = evaluate_model(model, test_loader, criterion, device)[1]
        assert 0 <= accuracy <= 100, "Accuracy is out of bounds."

    def test_model_save_load(self):
        """Test that the CNNModel can be saved and loaded without errors."""
        model = self.model()

        torch.save(model.state_dict(), "test_model.pth")

        loaded_model = CNNModel()
        loaded_model.load_state_dict(
            torch.load("test_model.pth", map_location="cpu", weights_only=True)
        )
        loaded_model.eval()

        result = loaded_model(torch.randn(1, 1, 28, 28))

        assert result.shape == (1, 10), "Loaded model output shape mismatch."

    def test_invalid_input(self):
        """Test that the CNNModel raises errors for invalid input."""
        model = self.model()

        with pytest.raises(TypeError):
            model(None)

        with pytest.raises(RuntimeError):
            model(torch.randn(1, 1, 28))  # Incorrect shape

    def test_model_training(self):
        """Test that the CNNModel can train for one step."""
        model = self.model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        model.train()
        optimizer.zero_grad()

        result = model(torch.randn(32, 1, 28, 28))
        loss = criterion(result, torch.randint(0, 10, (32,)))
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be greater than 0 during training."

    def test_train_function(self):
        """Test the train function with minimal data."""
        model = self.model()
        train_loader = DataLoader(
            TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))),
            batch_size=8,
        )
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        train_loss = train(model, train_loader, optimizer, criterion, device="cpu")
        assert train_loss > 0, "Train loss should be greater than 0."

    def test_compute_dataset_statistics(self):
        """Test the compute_dataset_statistics function."""
        dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        mean, std = compute_dataset_statistics(dataset)

        assert 0.0 < mean < 1.0, f"Mean {mean} out of range."
        assert 0.0 < std < 1.0, f"Std {std} out of range."

    def test_get_transforms(self):
        """Test the get_transforms function."""
        mean, std = 0.5, 0.5
        train_transform, test_transform = get_transforms(mean, std)

        # Convert the tensor to a PIL Image
        sample_image = torch.rand((28, 28))  # Remove the extra dimension for grayscale
        sample_image_pil = Image.fromarray((sample_image.numpy() * 255).astype("uint8"))

        transformed_train = train_transform(sample_image_pil)
        transformed_test = test_transform(sample_image_pil)

        # Ensure the outputs are tensors with the correct shape
        assert transformed_train.shape == (1, 28, 28), "Train transform failed."
        assert transformed_test.shape == (1, 28, 28), "Test transform failed."

    def test_get_data_loaders(self):
        """Test the get_data_loaders function."""
        train_loader, test_loader = get_data_loaders(
            batch_size=64, dataset_path="./data"
        )

        train_batch = next(iter(train_loader))
        assert len(train_batch[0]) == 64, "Train loader batch size mismatch."

        test_batch = next(iter(test_loader))
        assert len(test_batch[0]) <= 1024, "Test loader batch size mismatch."
