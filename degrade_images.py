import cv2
import numpy as np
from pathlib import Path
import shutil
import random


def create_sr_dataset_from_yolo():
    print("\n1. Поиск YOLO данных...")

    yolo_base = None
    possible_paths = [
        Path('datas'),
        Path('.'),
        Path('data'),
    ]

    for test_path in possible_paths:
        yolo_train = test_path / 'yolo_train'
        if yolo_train.exists():
            # Проверяем есть ли изображения и аннотации
            jpg_files = list(yolo_train.glob('*.jpg'))
            txt_files = list(yolo_train.glob('*.txt'))

            if jpg_files and txt_files:
                yolo_base = test_path
                print(f"Найдено в: {yolo_base}")
                print(f"  Изображений в train: {len(jpg_files)}")
                print(f"  Аннотаций в train: {len(txt_files)}")
                break

    if yolo_base is None:
        print("ОШИБКА: Не найдены YOLO данные!")
        print("Запустите сначала convert_json_to_txt.py")
        return None

    # 2. Создаем структуру SR датасета
    sr_base = Path('datas_sr_yolo')
    sr_base.mkdir(exist_ok=True)

    splits = ['train', 'valid', 'test']

    for split in splits:
        print(f"\n{'=' * 50}")
        print(f"ОБРАБОТКА: {split.upper()}")
        print('=' * 50)

        # Исходная YOLO директория
        yolo_dir = yolo_base / f'yolo_{split}'

        if not yolo_dir.exists():
            print(f"YOLO директория не найдена: {yolo_dir}")
            continue

        # Создаем папки для SR датасета
        hr_dir = sr_base / f'{split}_hr'  # Оригиналы (копии)
        lr_dir = sr_base / f'{split}_lr'  # Ухудшенные
        sr_dir = sr_base / f'{split}_sr'  # Улучшенные (будет позже)

        # Шаг 1: Копируем оригиналы (HR) - ВСЁ: и изображения, и аннотации
        print(f"1. Копирование оригиналов из {yolo_dir.name}...")
        hr_dir.mkdir(exist_ok=True)

        # Копируем ВСЕ файлы из YOLO директории
        copied_count = 0
        for item in yolo_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, hr_dir / item.name)
                copied_count += 1

        print(f"   Скопировано файлов: {copied_count}")

        hr_images = len(list(hr_dir.glob('*.jpg'))) + len(list(hr_dir.glob('*.png')))
        hr_labels = len(list(hr_dir.glob('*.txt')))
        print(f"   Изображений: {hr_images}, аннотаций: {hr_labels}")

        # Шаг 2: Создаем ухудшенные версии (LR)
        print(f"2. Создание ухудшенных версий (LR)...")
        create_lr_from_hr(hr_dir, lr_dir)

        # Проверяем результаты
        print(f"\nРезультаты для {split}:")
        print(f"  HR (оригиналы): {hr_images} изобр., {hr_labels} аннот.")

        lr_images = len(list(lr_dir.glob('*.jpg'))) + len(list(lr_dir.glob('*.png')))
        lr_labels = len(list(lr_dir.glob('*.txt')))
        print(f"  LR (ухудшенные): {lr_images} изобр., {lr_labels} аннот.")

    # 3. Создаем YAML файлы
    print(f"\n{'=' * 50}")
    print("СОЗДАНИЕ КОНФИГУРАЦИОННЫХ ФАЙЛОВ")
    print('=' * 50)

    create_yaml_files(sr_base)

    print(f"\n{'=' * 70}")
    print("ДАТАСЕТ УСПЕШНО СОЗДАН!")
    print("=" * 70)

    print(f"\nСтруктура: {sr_base}/")
    print("  train_hr/     # Оригиналы из yolo_train")
    print("  train_lr/     # Ухудшенные версии")
    print("  valid_hr/     # Оригиналы из yolo_valid")
    print("  valid_lr/     # Ухудшенные версии")
    print("  test_hr/      # Оригиналы из yolo_test")
    print("  test_lr/      # Ухудшенные версии")
    print("\n  *.yaml       # Конфигурационные файлы")

    print(f"\nДальнейшие действия:")
    print(f"1. Для обучения на оригинальных данных:")
    print(f"   python trainer.py --data {sr_base}/coffee_dataset_hr.yaml")
    print(f"\n2. Для обучения на ухудшенных данных:")
    print(f"   python trainer.py --data {sr_base}/coffee_dataset_lr.yaml")
    print(f"\n3. Чтобы улучшить LR -> SR:")
    print(f"   python enhance_sr_from_yolo.py")

    return sr_base


def create_lr_from_hr(hr_dir, lr_dir, degradation_type='medium'):
    """
    Создает ухудшенные версии изображений из HR директории
    """
    lr_dir.mkdir(parents=True, exist_ok=True)

    # Параметры ухудшения
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

    # Находим все изображения в HR
    images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        images.extend(hr_dir.glob(f'*{ext}'))
        images.extend(hr_dir.glob(f'*{ext.upper()}'))

    print(f"  Обработка {len(images)} изображений...")

    for i, img_path in enumerate(images):
        try:
            # Загружаем изображение
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Случайные параметры
            scale_factor = random.choice(scale_factors)
            jpeg_quality = random.choice(jpeg_qualities)
            blur_size = random.choice(blur_sizes)
            noise_level = random.choice(noise_levels)

            h, w = img.shape[:2]

            # 1. Уменьшаем
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            small_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 2. Размытие
            if blur_size % 2 == 0:
                blur_size += 1
            small_img = cv2.GaussianBlur(small_img, (blur_size, blur_size), 0.5)

            # 3. Шум
            noise = np.random.normal(0, noise_level, small_img.shape).astype(np.uint8)
            small_img = cv2.add(small_img, noise)

            # 4. JPEG артефакты
            temp_path = lr_dir / f"temp_{img_path.name}"
            cv2.imwrite(str(temp_path), small_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            small_img = cv2.imread(str(temp_path))
            temp_path.unlink()

            # 5. Увеличиваем обратно
            degraded_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_CUBIC)

            # 6. Финальное размытие
            degraded_img = cv2.GaussianBlur(degraded_img, (3, 3), 0.3)

            # Сохраняем ухудшенное изображение
            lr_img_path = lr_dir / img_path.name
            cv2.imwrite(str(lr_img_path), degraded_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # КОПИРУЕМ АННОТАЦИИ из HR
            hr_label = hr_dir / f"{img_path.stem}.txt"
            if hr_label.exists():
                lr_label = lr_dir / f"{img_path.stem}.txt"
                shutil.copy2(hr_label, lr_label)

            if (i + 1) % 10 == 0:
                print(f"    Обработано: {i + 1}/{len(images)}")

        except Exception as e:
            print(f"  Ошибка при обработке {img_path}: {e}")

    # Проверяем результат
    lr_images = len(list(lr_dir.glob('*.jpg'))) + len(list(lr_dir.glob('*.png')))
    lr_labels = len(list(lr_dir.glob('*.txt')))

    print(f"  Создано: {lr_images} изображений, {lr_labels} аннотаций")


def create_yaml_files(sr_base):
    """Создает YAML файлы для SR датасета"""

    # Определяем классы из файла classes.txt или аннотаций
    classes = determine_classes(sr_base)

    print(f"Определены классы: {classes}")

    # YAML для HR
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
    print(f"Создан: {yaml_hr}")

    # YAML для LR
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
    print(f"Создан: {yaml_lr}")

    # Общий YAML
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
# train: train_hr  # for HR
# train: train_lr  # for LR  
# train: train_sr  # for SR
""")
    print(f"Создан: {yaml_all}")

    # Основной YAML (по умолчанию HR)
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
        print(f"Создан: {yaml_main}")


def determine_classes(sr_base):
    """Определяет классы из аннотаций или файла classes.txt"""

    # Вариант 1: Ищем файл classes.txt
    classes_file = sr_base / 'train_hr' / 'classes.txt'
    if classes_file.exists():
        classes = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
        if classes:
            return classes

    # Вариант 2: Ищем файл classes.names
    classes_names = sr_base / 'train_hr' / 'classes.names'
    if classes_names.exists():
        classes = [line.strip() for line in classes_names.read_text().splitlines() if line.strip()]
        if classes:
            return classes

    # Вариант 3: Определяем из аннотаций
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
            # Сортируем ID классов
            sorted_ids = sorted(list(all_class_ids))
            # Создаем имена классов
            classes = [f'class_{cid}' for cid in sorted_ids]
            return classes

    # Вариант 4: По умолчанию для кофе
    return ['unripe', 'ripe', 'overripe']


def verify_dataset(sr_base):
    """Проверяет целостность датасета"""
    print(f"\n{'=' * 50}")
    print("ПРОВЕРКА ЦЕЛОСТНОСТИ ДАТАСЕТА")
    print('=' * 50)

    issues = []

    for split in ['train', 'valid', 'test']:
        for version in ['hr', 'lr']:
            dir_path = sr_base / f'{split}_{version}'

            if dir_path.exists():
                images = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
                labels = list(dir_path.glob('*.txt'))

                print(f"\n{dir_path.name}:")
                print(f"  Изображений: {len(images)}")
                print(f"  Аннотаций: {len(labels)}")

                # Проверяем соответствие
                image_names = {img.stem for img in images}
                label_names = {label.stem for label in labels}

                missing = image_names - label_names
                extra = label_names - image_names

                if missing:
                    print(f"  ⚠ Нет аннотаций для {len(missing)} изображений")
                    issues.append(f"{dir_path.name}: missing {len(missing)} labels")

                if extra:
                    print(f"  ⚠ Лишние аннотации: {len(extra)}")
                    issues.append(f"{dir_path.name}: extra {len(extra)} labels")

                if not missing and not extra:
                    print(f"  ✓ Все изображения имеют аннотации")
            else:
                print(f"\n{dir_path.name}: не найдено")

    if issues:
        print(f"\nПРОБЛЕМЫ ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")

        fix = input("\nИсправить автоматически (создать пустые файлы)? (y/n): ")
        if fix.lower() == 'y':
            fix_missing_labels(sr_base)
    else:
        print(f"\n✓ Датсет в порядке!")


def fix_missing_labels(sr_base):
    """Создает пустые файлы аннотаций для изображений без лейблов"""
    for split in ['train', 'valid', 'test']:
        for version in ['hr', 'lr']:
            dir_path = sr_base / f'{split}_{version}'

            if dir_path.exists():
                for img in dir_path.glob('*.jpg'):
                    label = dir_path / f"{img.stem}.txt"
                    if not label.exists():
                        label.write_text('')
                        print(f"  Создан пустой файл: {label.name}")


if __name__ == "__main__":
    # Создаем SR датасет из YOLO данных
    dataset_dir = create_sr_dataset_from_yolo()

    if dataset_dir:
        # Проверяем целостность
        verify_dataset(dataset_dir)

        print(f"\nГотово! Датсет создан в: {dataset_dir}")