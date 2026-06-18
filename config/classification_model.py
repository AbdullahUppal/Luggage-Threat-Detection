import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class ClassificationDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], image_size: Tuple[int, int] = (512, 512)):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[idx]
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, self.image_size, interpolation=cv.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)

    def _generate_safe_annotation(self, target_directory):
        file_paths = []
        width = height = 512
        # if not list(target_path.iterdir()):
        for x in range(0,800):
            black_image = np.zeros((width, height), dtype=np.uint8)

            blob_count = 0
            make_random = False
            if x > 600:
                blob_count = np.random.randint(2, 8)
            elif x > 500:
                make_random = True
            elif x > 400:
                blob_count = 1

            if make_random:
                pixel_count = np.random.randint(300, 2000)
                ys = np.random.randint(0, height, size=pixel_count)
                zs = np.random.randint(0, width, size=pixel_count)
                black_image[ys, zs] = 255

            for _ in range(blob_count):
                y = np.random.randint(0, height)
                z = np.random.randint(0, width)

                walk_length = np.random.randint(100, 400)

                for _ in range(walk_length):
                    black_image[y, z] = 255

                    direction = np.random.randint(0, 4)
                    if direction == 0:
                        y = max(0, y - 1)
                    elif direction == 1:
                        y = min(height - 1, y + 1)
                    elif direction == 2:
                        z = max(0, z - 1)
                    else:
                        z = min(width - 1, z + 1)
            file_name = f"safe_{x}.png"
            path = Path(target_directory) / 'safe' / file_name
            file_paths.append(path)
            cv.imwrite(path, black_image)     
        return file_paths  

class ThreatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class Classification_Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        self.model_path = "model/classification_model.pt"
        self.class_names: List[str] = []
        self.model: ThreatClassifier 
        print(self.device.type)
        if self.device.type == "cuda":
            print(f"PyTorch classification runtime: using GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("PyTorch classification runtime: using CPU")

        if os.path.exists(self.model_path):
            self.load_model()

    def _build_model(self, num_classes: int):
        self.model = ThreatClassifier(num_classes).to(self.device)

    def _collect_class_samples(self, dataset_dir: str):
        root = Path(dataset_dir)
        if not root.exists():
            raise ValueError(f"Training directory not found: {dataset_dir}")

        class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class folders found in: {dataset_dir}")

        class_names = [d.name for d in class_dirs]
        class_names.append("safe")
        samples_by_class: List[List[Tuple[str, int]]] = []

        for label, class_dir in enumerate(class_dirs):
            class_samples: List[Tuple[str, int]] = []
            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file():
                    class_samples.append((str(file_path), label))
            if class_samples:
                samples_by_class.append(class_samples)

        if not samples_by_class:
            raise ValueError(f"No training images found in: {dataset_dir}")

        return class_names, samples_by_class

    def _split_samples(self, samples_by_class, validation_ratio: float = 0.2, seed: int = 123):
        rng = random.Random(seed)
        train_samples: List[Tuple[str, int]] = []
        validation_samples: List[Tuple[str, int]] = []

        for class_samples in samples_by_class:
            rng.shuffle(class_samples)
            split_idx = max(1, int(len(class_samples) * (1 - validation_ratio)))
            if split_idx >= len(class_samples):
                split_idx = len(class_samples) - 1
            if split_idx <= 0:
                raise ValueError("Each class needs at least 2 images for train/validation split.")

            train_samples.extend(class_samples[:split_idx])
            validation_samples.extend(class_samples[split_idx:])

        if not train_samples or not validation_samples:
            raise ValueError("Training or validation split is empty.")

        rng.shuffle(train_samples)
        rng.shuffle(validation_samples)
        return train_samples, validation_samples

    def _make_dataloaders(self, train_samples, validation_samples, batch_size: int = 16):
        train_dataset = ClassificationDataset(train_samples)
        validation_dataset = ClassificationDataset(validation_samples)
        pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        return train_loader, validation_loader

    def train(self, train_path):
        
        class_names, samples_by_class = self._collect_class_samples(train_path)
        train_samples, validation_samples = self._split_samples(samples_by_class)
        train_loader, validation_loader = self._make_dataloaders(train_samples, validation_samples)

        self.class_names = class_names
        self._build_model(len(self.class_names))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        epochs = 50
        for epoch in range(epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

            self.model.eval()
            validation_loss_sum = 0.0
            validation_correct = 0
            validation_total = 0

            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    logits = self.model(images)
                    validation_loss_sum += criterion(logits, labels).item()
                    predictions = torch.argmax(logits, dim=1)
                    validation_correct += (predictions == labels).sum().item()
                    validation_total += labels.size(0)

            train_loss = train_loss_sum / len(train_loader)
            validation_loss = validation_loss_sum / len(validation_loader)
            train_accuracy = train_correct / train_total
            validation_accuracy = validation_correct / validation_total
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_accuracy:.4f} | "
                f"val_loss: {validation_loss:.4f} | val_acc: {validation_accuracy:.4f}"
            )

        os.makedirs("model", exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "class_names": self.class_names,
            },
            self.model_path,
        )
        print(f"Saved PyTorch classification model to {self.model_path}")

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        self._build_model(len(self.class_names))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def predict_class(self, image_path):
        if self.model is None:
            self.load_model()

        image_dir = Path(image_path) / 'result'
        file_paths = sorted([path for path in image_dir.iterdir() if path.is_file()])
        if not file_paths:
            raise ValueError(f"No files found for prediction in: {image_path}")

        results = {class_name: [] for class_name in self.class_names}
        print("\n--- RESULTS ---")

        with torch.no_grad():
            for file_path in file_paths:
                image = cv.imread(str(file_path), cv. IMREAD_GRAYSCALE)
                if image is None:
                    continue
                # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (512, 512), interpolation=cv.INTER_LINEAR)
                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))
                image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

                logits = self.model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_index = int(torch.argmax(probabilities, dim=1).item())
                predicted_class = self.class_names[predicted_index]
                confidence = float(probabilities[0, predicted_index].item() * 100)
                results[predicted_class].append(file_path.name)

                print(
                    f"Image: {file_path.name} | Prediction: {predicted_class} | Confidence: {confidence:.2f}%"
                )

        with open("classification_result.json", "w") as file:
            json.dump(results, file, indent=4)

        print(f"\nSuccessfully processed {len(file_paths)} images!")
        return results

    def get_classification_accuracy(self, result):
        return result
