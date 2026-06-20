import cv2
import numpy as np
from pathlib import Path
import shutil
import random


def create_sr_dataset_from_yolo():
    yolo_base = None
    possible_paths = [
        Path('datas'),
        Path('.'),
        Path('data'),
    ]

    for test_path in possible_paths:
        yolo_train = test_path / 'yolo_train'
        if yolo_train.exists():
            jpg_files = list(yolo_train.glob('*.jpg'))
            txt_files = list(yolo_train.glob('*.txt'))

            if jpg_files and txt_files:
                yolo_base = test_path
                print(f"Найдено в: {yolo_base}")
                break

    if yolo_base is None:
        return None

    sr_base = Path('datas_sr_yolo')
    sr_base.mkdir(exist_ok=True)

    splits = ['train', 'valid', 'test']

    for split in splits:
        yolo_dir = yolo_base / f'yolo_{split}'

        if not yolo_dir.exists():
            continue

        hr_dir = sr_base / f'{split}_hr'
        lr_dir = sr_base / f'{split}_lr'
        sr_dir = sr_base / f'{split}_sr'

        hr_dir.mkdir(exist_ok=True)
        copied_count = 0
        for item in yolo_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, hr_dir / item.name)
                copied_count += 1

        hr_images = len(list(hr_dir.glob('*.jpg'))) + len(list(hr_dir.glob('*.png')))
        hr_labels = len(list(hr_dir.glob('*.txt')))
        create_lr_from_hr(hr_dir, lr_dir)

        lr_images = len(list(lr_dir.glob('*.jpg'))) + len(list(lr_dir.glob('*.png')))
        lr_labels = len(list(lr_dir.glob('*.txt')))

    create_yaml_files(sr_base)
    return sr_base


def create_lr_from_hr(hr_dir, lr_dir, degradation_type='medium'):

    lr_dir.mkdir(parents=True, exist_ok=True)

    if degradation_type == 'light':
        scale_factors = [0.67, 0.75]
        jpeg_qualities = [75, 80]
        blur_sizes = [3, 5]
        noise_levels = [5, 8]
    elif degradation_type == 'medium':
        scale_factors = [0.33, 0.5]
        jpeg_qualities = [50, 60]
        blur_sizes = [5, 7]
        noise_levels = [10, 15]
    else:  # heavy
        scale_factors = [0.2, 0.25]
        jpeg_qualities = [30, 40]
        blur_sizes = [7, 9]
        noise_levels = [15, 20]

    images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        images.extend(hr_dir.glob(f'*{ext}'))
        images.extend(hr_dir.glob(f'*{ext.upper()}'))

    for i, img_path in enumerate(images):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            scale_factor = random.choice(scale_factors)
            jpeg_quality = random.choice(jpeg_qualities)
            blur_size = random.choice(blur_sizes)
            noise_level = random.choice(noise_levels)

            h, w = img.shape[:2]

            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            small_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if blur_size % 2 == 0:
                blur_size += 1
            small_img = cv2.GaussianBlur(small_img, (blur_size, blur_size), 0.5)

            noise = np.random.normal(0, noise_level, small_img.shape).astype(np.uint8)
            small_img = cv2.add(small_img, noise)

            temp_path = lr_dir / f"temp_{img_path.name}"
            cv2.imwrite(str(temp_path), small_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            small_img = cv2.imread(str(temp_path))
            temp_path.unlink()

            degraded_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_CUBIC)

            degraded_img = cv2.GaussianBlur(degraded_img, (3, 3), 0.3)

            lr_img_path = lr_dir / img_path.name
            cv2.imwrite(str(lr_img_path), degraded_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            hr_label = hr_dir / f"{img_path.stem}.txt"
            if hr_label.exists():
                lr_label = lr_dir / f"{img_path.stem}.txt"
                shutil.copy2(hr_label, lr_label)

        except Exception as e:
            pass

    lr_images = len(list(lr_dir.glob('*.jpg'))) + len(list(lr_dir.glob('*.png')))
    lr_labels = len(list(lr_dir.glob('*.txt')))


def create_yaml_files(sr_base):
    classes = determine_classes(sr_base)
    yaml_hr = sr_base / 'coffee_dataset_hr.yaml'
    yaml_hr.write_text(f"""# Coffee Dataset - HR Version
# Original images from yolo_* directories

path: {sr_base}

train: train_hr
val: valid_hr
test: test_hr

nc: {len(classes)}
names: {classes}
""")

    yaml_lr = sr_base / 'coffee_dataset_lr.yaml'
    yaml_lr.write_text(f"""# Coffee Dataset - LR Version  
# Degraded low resolution images

path: {sr_base}

train: train_lr
val: valid_lr
test: test_lr

nc: {len(classes)}
names: {classes}
""")

    yaml_all = sr_base / 'coffee_dataset_all.yaml'
    yaml_all.write_text(f"""# Coffee Dataset - All Versions

path: {sr_base}

# HR Version (original)
train_hr: train_hr
val_hr: valid_hr
test_hr: test_hr

# LR Version (degraded)
train_lr: train_lr
val_lr: valid_lr
test_lr: test_lr

# SR Version (after enhancement)
train_sr: train_sr
val_sr: valid_sr
test_sr: test_sr

nc: {len(classes)}
names: {classes}

# To use a specific version:
# Copy these lines to coffee_dataset.yaml:
# train: train_hr
# train: train_lr
# train: train_sr
""")

    yaml_main = sr_base / 'coffee_dataset.yaml'
    if not yaml_main.exists():
        yaml_main.write_text(f"""# Coffee Dataset - Default (HR)
# This is the main configuration file

path: {sr_base}

train: train_hr
val: valid_hr
test: test_hr

nc: {len(classes)}
names: {classes}
""")

def determine_classes(sr_base):

    classes_file = sr_base / 'train_hr' / 'classes.txt'
    if classes_file.exists():
        classes = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
        if classes:
            return classes

    classes_names = sr_base / 'train_hr' / 'classes.names'
    if classes_names.exists():
        classes = [line.strip() for line in classes_names.read_text().splitlines() if line.strip()]
        if classes:
            return classes

    all_class_ids = set()
    train_hr_dir = sr_base / 'train_hr'

    if train_hr_dir.exists():
        for txt_file in train_hr_dir.glob('*.txt'):
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_class_ids.add(int(parts[0]))
            except:
                pass

        if all_class_ids:
            sorted_ids = sorted(list(all_class_ids))
            classes = [f'class_{cid}' for cid in sorted_ids]
            return classes

    return ['unripe', 'ripe', 'overripe']


def verify_dataset(sr_base):
    issues = []

    for split in ['train', 'valid', 'test']:
        for version in ['hr', 'lr']:
            dir_path = sr_base / f'{split}_{version}'

            if dir_path.exists():
                images = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
                labels = list(dir_path.glob('*.txt'))

                image_names = {img.stem for img in images}
                label_names = {label.stem for label in labels}

                missing = image_names - label_names
                extra = label_names - image_names

                if missing:
                    issues.append(f"{dir_path.name}: missing {len(missing)} labels")

                if extra:
                    issues.append(f"{dir_path.name}: extra {len(extra)} labels")

def fix_missing_labels(sr_base):
    for split in ['train', 'valid', 'test']:
        for version in ['hr', 'lr']:
            dir_path = sr_base / f'{split}_{version}'

            if dir_path.exists():
                for img in dir_path.glob('*.jpg'):
                    label = dir_path / f"{img.stem}.txt"
                    if not label.exists():
                        label.write_text('')


if __name__ == "__main__":
    dataset_dir = create_sr_dataset_from_yolo()

    if dataset_dir:
        verify_dataset(dataset_dir)