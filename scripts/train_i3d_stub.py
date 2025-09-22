import os
import sys
import argparse
import yaml

def validate_dataset(root: str, train_dir: str, val_dir: str) -> None:
    train_path = os.path.join(root, train_dir)
    val_path = os.path.join(root, val_dir)
    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Train directory missing: {train_path}")
    if not os.path.isdir(val_path):
        raise FileNotFoundError(f"Val directory missing: {val_path}")
    # Expect class subfolders inside train/ and val/
    train_classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    val_classes = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
    if len(train_classes) < 2:
        raise ValueError("Expected at least 2 classes in train split (e.g., violent, non-violent)")
    if not set(train_classes).issubset(set(val_classes)):
        print("Warning: class mismatch between train and val")
    print(f"Classes: {sorted(set(train_classes) | set(val_classes))}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/training.yaml', help='Path to training yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data = cfg.get('data', {})
    train_cfg = cfg.get('train', {})
    save_cfg = cfg.get('save', {})

    validate_dataset(data.get('dataset_root', ''), data.get('train_dir', 'train'), data.get('val_dir', 'val'))

    os.makedirs(save_cfg.get('output_dir', 'data/models'), exist_ok=True)

    print("This is a training stub. Next steps:")
    print("- Build I3D model (RGB stream) with Kinetics pretrained weights")
    print("- Create VideoDataset to load 16-frame clips at 8 FPS, 224x224")
    print("- Train with Adam, CE loss, warmup + cosine decay, dropout 0.5, early stopping")
    print("- Save checkpoint to:", os.path.join(save_cfg.get('output_dir', 'data/models'), save_cfg.get('checkpoint_name', 'violence_i3d_rgb.pt')))

if __name__ == '__main__':
    main()


