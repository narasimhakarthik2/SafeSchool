import os
import shutil
from sklearn.model_selection import train_test_split
import glob


def organize_dataset(source_dir: str, destination_dir: str, val_split: float = 0.2):
    """
    Organizes the dataset by:
    1. Separating images and annotations
    2. Creating train/val splits
    3. Maintaining the same split for corresponding images and annotations
    """
    # Create necessary directories
    os.makedirs(destination_dir, exist_ok=True)

    # Create train and val directories for both images and annotations
    dirs = ['train/images', 'train/annotations',
            'val/images', 'val/annotations']

    for dir_path in dirs:
        os.makedirs(os.path.join(destination_dir, dir_path), exist_ok=True)

    # Get all image files
    image_files = glob.glob(os.path.join(source_dir, "*.jpg")) + \
                  glob.glob(os.path.join(source_dir, "*.png"))

    # Create file names list (without extension)
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

    # Split into train and val sets
    train_names, val_names = train_test_split(
        base_names,
        test_size=val_split,
        random_state=42
    )

    # Move files to respective directories
    def move_files(file_names, split_type):
        for name in file_names:
            # Move image
            for ext in ['.jpg', '.png']:
                img_src = os.path.join(source_dir, name + ext)
                if os.path.exists(img_src):
                    shutil.copy2(
                        img_src,
                        os.path.join(destination_dir, split_type, 'images', name + ext)
                    )
                    break

            # Move annotation
            xml_src = os.path.join(source_dir, name + '.xml')
            if os.path.exists(xml_src):
                shutil.copy2(
                    xml_src,
                    os.path.join(destination_dir, split_type, 'annotations', name + '.xml')
                )

    # Move files to train and val directories
    move_files(train_names, 'train')
    move_files(val_names, 'val')

    # Print statistics
    print(f"Dataset organized successfully:")
    print(f"Training samples: {len(train_names)}")
    print(f"Validation samples: {len(val_names)}")


if __name__ == "__main__":
    # Update these paths according to your setup
    SOURCE_DIR = "../data/Images"
    DESTINATION_DIR = "../data/mock_attack_dataset"

    organize_dataset(SOURCE_DIR, DESTINATION_DIR)

    # Verify the organization
    for split in ['train', 'val']:
        n_images = len(os.listdir(os.path.join(DESTINATION_DIR, split, 'images')))
        n_annotations = len(os.listdir(os.path.join(DESTINATION_DIR, split, 'annotations')))
        print(f"\n{split} set:")
        print(f"Number of images: {n_images}")
        print(f"Number of annotations: {n_annotations}")