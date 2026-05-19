import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(data_dir, batch_size, num_workers):
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, epoch, log_interval):
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / seen
            accuracy = 100.0 * correct / seen
            print(
                f"Epoch {epoch} [{seen}/{len(loader.dataset)}] "
                f"loss={avg_loss:.4f} acc={accuracy:.2f}%"
            )

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)

    return total_loss / seen, correct / seen


def train(args):
    torch.manual_seed(args.seed)
    device = get_device()
    train_loader, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )

    model = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.log_interval,
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch} done: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    output_path = Path(args.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


def load_model(model_path, device):
    model = DigitCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_predict_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def predict_probabilities(model, image_path, device):
    transform = build_predict_transform()
    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1).squeeze(0)

    return image, probabilities.cpu()


def predict_image(args):
    device = get_device()
    model = load_model(args.model_path, device)
    _, probabilities = predict_probabilities(model, args.image_path, device)
    confidence, prediction = probabilities.max(dim=0)

    print(f"Prediction: {prediction.item()}, confidence: {confidence.item():.4f}")


def visualize_prediction(args):
    device = get_device()
    model = load_model(args.model_path, device)
    image, probabilities = predict_probabilities(model, args.image_path, device)
    confidence, prediction = probabilities.max(dim=0)

    values = probabilities.tolist()
    canvas = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=22)

    preview = image.convert("L").resize((180, 180), Image.Resampling.NEAREST).convert("RGB")
    canvas.paste(preview, (50, 90))
    draw.text((50, 40), "Input image", fill="#111827", font=title_font)
    draw.text(
        (50, 290),
        f"Prediction: {prediction.item()}",
        fill="#111827",
        font=title_font,
    )
    draw.text(
        (50, 325),
        f"Confidence: {confidence.item():.2%}",
        fill="#111827",
        font=title_font,
    )

    chart_left = 310
    chart_top = 70
    chart_width = 530
    chart_height = 260
    draw.text((chart_left, 40), "Class probabilities", fill="#111827", font=title_font)
    draw.line(
        [(chart_left, chart_top + chart_height), (chart_left + chart_width, chart_top + chart_height)],
        fill="#374151",
        width=2,
    )
    draw.line(
        [(chart_left, chart_top), (chart_left, chart_top + chart_height)],
        fill="#374151",
        width=2,
    )

    bar_gap = 12
    bar_width = (chart_width - bar_gap * 11) // 10
    for digit, probability in enumerate(values):
        x0 = chart_left + bar_gap + digit * (bar_width + bar_gap)
        x1 = x0 + bar_width
        bar_height = int(probability * chart_height)
        y0 = chart_top + chart_height - bar_height
        y1 = chart_top + chart_height
        color = "#2563EB" if digit == prediction.item() else "#9CA3AF"
        draw.rectangle((x0, y0, x1, y1), fill=color)
        draw.text((x0 + 8, y1 + 10), str(digit), fill="#111827", font=font)
        if probability >= 0.01:
            draw.text((x0 - 2, max(chart_top, y0 - 18)), f"{probability:.2f}", fill="#111827", font=font)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Visualization saved to {output_path}")
    print(f"Prediction: {prediction.item()}, confidence: {confidence.item():.4f}")

    if args.show:
        canvas.show()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST handwritten digit recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate on MNIST")
    train_parser.add_argument("--data-dir", default="data", help="MNIST data directory")
    train_parser.add_argument("--model-path", default="models/mnist_cnn.pt", help="Saved model path")
    train_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--log-interval", type=int, default=100, help="Training log interval")

    predict_parser = subparsers.add_parser("predict", help="Predict a single digit image")
    predict_parser.add_argument("--model-path", default="models/mnist_cnn.pt", help="Saved model path")
    predict_parser.add_argument("--image-path", required=True, help="Path to a digit image")

    visualize_parser = subparsers.add_parser("visualize", help="Save a prediction visualization")
    visualize_parser.add_argument("--model-path", default="models/mnist_cnn.pt", help="Saved model path")
    visualize_parser.add_argument("--image-path", required=True, help="Path to a digit image")
    visualize_parser.add_argument(
        "--output-path",
        default="outputs/prediction.png",
        help="Path for the visualization image",
    )
    visualize_parser.add_argument("--show", action="store_true", help="Show the plot window")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict_image(args)
    elif args.command == "visualize":
        visualize_prediction(args)


if __name__ == "__main__":
    main()
