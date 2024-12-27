import torch
from ultralytics import YOLO
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Convert XML annotations to YOLO format
        self.prepare_yolo_dataset()

        # Initialize model
        self.model = YOLO('yolov8n.pt')
        print(f"Model loaded: YOLOv8n")

        # Create YAML file for dataset configuration
        self.create_dataset_yaml()

        # Training parameters
        self.num_epochs = config['training']['epochs']
        self.save_dir = config['training']['checkpoint_dir']
        os.makedirs(self.save_dir, exist_ok=True)

    def prepare_yolo_dataset(self):
        """Convert XML annotations to YOLO format and organize dataset"""
        data_dir = Path(self.config['data']['processed_dir'])

        for split in ['train', 'val']:
            # Create labels directory
            labels_dir = data_dir / split / 'labels'
            labels_dir.mkdir(exist_ok=True)

            # Process each XML file
            ann_dir = data_dir / split / 'annotations'
            for xml_file in ann_dir.glob('*.xml'):
                # Convert XML to YOLO format
                self.convert_xml_to_yolo(xml_file, labels_dir)

    def convert_xml_to_yolo(self, xml_path, output_dir):
        """Convert single XML file to YOLO format"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image size
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)

        # Prepare YOLO annotations
        yolo_annotations = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = self.config['model']['classes'].index(class_name)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # Save YOLO format annotations
        output_file = output_dir / f"{xml_path.stem}.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    def create_dataset_yaml(self):
        """Create YAML file for YOLO dataset configuration"""
        dataset_config = {
            'path': str(Path(self.config['data']['processed_dir']).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.config['model']['classes']),
            'names': {i: name for i, name in enumerate(self.config['model']['classes'])}
        }

        dataset_yaml_path = 'configs/dataset.yaml'
        os.makedirs('configs', exist_ok=True)

        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, sort_keys=False)

        print(f"Dataset configuration saved to {dataset_yaml_path}")
        self.dataset_yaml = dataset_yaml_path

    def train(self, train_loader, val_loader):
        """Training loop"""
        try:
            results = self.model.train(
                data=self.dataset_yaml,
                epochs=self.num_epochs,
                imgsz=self.config['data']['img_size'],
                batch=self.config['training']['batch_size'],
                device=self.device,
                workers=self.config['training']['num_workers'],
                pretrained=True,
                optimizer='Adam',
                lr0=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                name='train',
                project='safeschool',
                exist_ok=True
            )

            print("Training completed successfully!")
            return results

        except Exception as e:
            print(f"\nError during training: {e}")
            raise e