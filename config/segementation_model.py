import os
import random
from pathlib import Path
from typing import List, Tuple
from PIL import Image

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        xb = self.bottleneck(self.pool4(x4))

        x = self.up4(xb)
        x = self.dec4(torch.cat([x, x4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.out_conv(x)


class SegmentationThreatDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], image_size: Tuple[int, int] = (512, 512)):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, self.image_size, interpolation=cv.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {mask_path}")
        mask = cv.resize(mask, self.image_size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)  

class SegmentationModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.model_path = "model/segmentation_model.pt"

        if self.device.type == "cuda":
            print(f"PyTorch runtime: using GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("PyTorch runtime: using CPU")

    def collect_image_mask_pairs(self, image_root: str, mask_root: str, class_names: List[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []

        mask_dirs = {}
        for d in Path(mask_root).iterdir():
            if d.is_dir():
                mask_dirs[d.name.lower()] = d

        for class_name in class_names:
            image_dir = Path(image_root) / class_name
            mask_dir = mask_dirs.get(class_name.lower())
            if not image_dir.exists() or mask_dir is None:
                continue

            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file():
                    continue
                mask_path = mask_dir / image_path.name
                if mask_path.exists():
                    pairs.append((str(image_path), str(mask_path)))

        if not pairs:
            raise ValueError("No image-mask pairs found.")

        return pairs

    def prepare_segmentation_data(
        self,
        train_root: str,
        annotation_root: str,
        batch_size: int = 8,
        train_ratio: float = 0.8,
        seed: int = 123,
    ) -> Tuple[DataLoader, DataLoader]:
        class_names = ["GUN", "knife", "shuriken"]
        pairs = self.collect_image_mask_pairs(train_root, annotation_root, class_names)

        random.seed(seed)
        random.shuffle(pairs)

        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        if not train_pairs or not val_pairs:
            raise ValueError("Not enough image-mask pairs to split into train and validation sets.")

        train_dataset = SegmentationThreatDataset(train_pairs)
        val_dataset = SegmentationThreatDataset(val_pairs)

        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

    def train_segmentation(self, train_root: str, annotation_root: str, epochs: int = 50, batch_size: int = 8):
        train_loader, val_loader = self.prepare_segmentation_data(
            train_root=train_root,
            annotation_root=annotation_root,
            batch_size=batch_size,
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            self.model.train()
            train_loss_sum = 0.0

            for images, masks in train_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    logits = self.model(images)
                    val_loss_sum += criterion(logits, masks).item()

            train_loss = train_loss_sum / len(train_loader)
            val_loss = val_loss_sum / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        os.makedirs("model", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Saved PyTorch segmentation model to {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def segment_threat(self, filepath, save_path='../DIP Data Upload/result/', threshold=0.1):
        # Load image (RGB) and keep original size for final mask
        input_image = Image.open(filepath).convert("RGB")
        original_w, original_h = input_image.size

        # Match training preprocessing: resize + [0,1] scale only
        resized = input_image.resize((512, 512))
        image_np = np.array(resized).astype(np.float32) / 255.0          # H,W,C
        image_np = np.transpose(image_np, (2, 0, 1))                     # C,H,W
        input_tensor = torch.from_numpy(image_np).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)                             # [1,1,H,W]
            probs = torch.sigmoid(logits)                                 # [0,1]
            pred = (probs >= threshold).float()                            # binary mask

        # Convert to uint8 image (0 or 255)
        mask = pred[0, 0].detach().cpu().numpy().astype(np.uint8) * 255  # H,W
        mask_img = Image.fromarray(mask, mode="L").resize((original_w, original_h), Image.NEAREST)

        if save_path is not None:
            mask_img.save(save_path)

        return mask_img
