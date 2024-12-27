import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Tuple, List


class MockAttackDataset(Dataset):
    """Dataset class for Mock Attack dataset with XML annotations"""

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            img_size: int = 640,
            transform=None
    ):
        """
        Args:
            root_dir (str): Dataset root directory
            split (str): 'train' or 'val'
            img_size (int): Target image size
            transform: Optional transform to be applied
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform or self._get_default_transforms()

        # Setup paths
        split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(split_dir, 'images')
        self.ann_dir = os.path.join(split_dir, 'annotations')

        # Get file lists
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(('.jpg', '.png'))
        ])

        # Class mapping - updated to match XML annotations
        self.class_names = ['Handgun', 'Knife', 'Short_rifle']
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.class_names)
        }

        # Verify dataset integrity
        self._verify_dataset()

    def _verify_dataset(self):
        """Verify that all images have corresponding annotations"""
        for img_file in self.img_files:
            ann_file = os.path.splitext(img_file)[0] + '.xml'
            ann_path = os.path.join(self.ann_dir, ann_file)
            assert os.path.exists(ann_path), \
                f"Annotation not found for {img_file}"

    def _get_default_transforms(self) -> A.Compose:
        """Get default augmentation pipeline"""
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.OneOf([
                    A.MotionBlur(p=1),
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1)
                ], p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',  # Changed to YOLO format
                label_fields=['class_labels']
            ))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',  # Changed to YOLO format
                label_fields=['class_labels']
            ))

    def _parse_annotation(self, xml_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse XML annotation file and convert to YOLO format"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        class_labels = []

        # Get image dimensions
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)

        # Parse objects
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.class_to_idx:  # Changed from self.class_map
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # Ensure coordinates are within image bounds
                xmin = max(0, min(xmin, img_width))
                ymin = max(0, min(ymin, img_height))
                xmax = max(0, min(xmax, img_width))
                ymax = max(0, min(ymax, img_height))

                # Convert to YOLO format (normalized coordinates)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                boxes.append([x_center, y_center, width, height])
                class_labels.append(self.class_to_idx[class_name])

        return np.array(boxes), np.array(class_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get item by index"""
        # Get paths
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        ann_path = os.path.join(
            self.ann_dir,
            os.path.splitext(self.img_files[idx])[0] + '.xml'
        )

        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load annotations
        boxes, class_labels = self._parse_annotation(ann_path)

        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )

            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            class_labels = torch.tensor(
                transformed['class_labels'],
                dtype=torch.long
            )
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)

        # Prepare target dict
        target = {
            'boxes': boxes,
            'labels': class_labels,
            'image_id': torch.tensor([idx]),
            'file_name': self.img_files[idx]
        }

        return image, target

    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.img_files)


def create_dataloader(
        root_dir: str,
        split: str = 'train',
        batch_size: int = 16,
        num_workers: int = 4,
        img_size: int = 640
) -> DataLoader:
    """Create a dataloader for the dataset"""
    dataset = MockAttackDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )


def collate_fn(batch: List[Tuple]) -> Tuple[List, List]:
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))