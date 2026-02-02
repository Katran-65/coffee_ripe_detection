import cv2
import numpy as np
from pathlib import Path
import shutil


def enhance_yolo_dataset():

    # Проверяем существование датасета
    sr_base = Path('datas_sr_yolo')

    if not sr_base.exists():
        print(f"Датасет не найден: {sr_base}")
        print("Сначала запустите: python create_sr_from_yolo.py")
        return

    print(f"Работаем с датасетом: {sr_base}")

    # Выбираем метод улучшения
    method = select_enhancement_method()

    splits = ['train', 'valid', 'test']

    for split in splits:
        print(f"\n{'=' * 50}")
        print(f"ОБРАБОТКА: {split.upper()}")
        print('=' * 50)

        lr_dir = sr_base / f'{split}_lr'
        sr_dir = sr_base / f'{split}_sr'
        hr_dir = sr_base / f'{split}_hr'  # Для копирования аннотаций

        if not lr_dir.exists():
            print(f"LR директория не найдена: {lr_dir}")
            continue

        print(f"LR источник: {lr_dir}")
        print(f"SR цель: {sr_dir}")

        # Улучшаем изображения
        enhanced_count = enhance_images(lr_dir, sr_dir, method)

        # Копируем аннотации из HR (оригинальные аннотации)
        copy_labels_from_hr(hr_dir, sr_dir, lr_dir)

        # Проверяем результат
        sr_images = len(list(sr_dir.glob('*.jpg'))) + len(list(sr_dir.glob('*.png')))
        sr_labels = len(list(sr_dir.glob('*.txt')))

        print(f"\nРезультат для {split}:")
        print(f"  Улучшено изображений: {enhanced_count}")
        print(f"  Всего SR изображений: {sr_images}")
        print(f"  Всего SR аннотаций: {sr_labels}")

    # Обновляем YAML файлы
    update_yaml_files(sr_base)

    print(f"\n{'=' * 70}")
    print("УЛУЧШЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)


def select_enhancement_method():
    """Выбор метода улучшения"""
    print("\nВыберите метод улучшения:")
    print("  1. Гибридный (рекомендуется)")
    print("  2. Продвинутая бикубическая")
    print("  3. Простая бикубическая")
    print("  4. Lanczos (высокое качество)")

    choice = input("\nВаш выбор (1-4, по умолчанию 1): ").strip() or '1'

    methods = {
        '1': 'hybrid',
        '2': 'advanced_bicubic',
        '3': 'simple',
        '4': 'lanczos'
    }

    method = methods.get(choice, 'hybrid')
    print(f"Выбран метод: {method}")

    return method


def enhance_images(lr_dir, sr_dir, method='hybrid'):
    """Улучшает изображения выбранным методом"""
    sr_dir.mkdir(parents=True, exist_ok=True)

    # Находим все LR изображения
    images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        images.extend(lr_dir.glob(f'*{ext}'))
        images.extend(lr_dir.glob(f'*{ext.upper()}'))

    print(f"Найдено LR изображений: {len(images)}")

    enhanced_count = 0

    for i, img_path in enumerate(images):
        try:
            # Загружаем изображение
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Применяем выбранный метод
            if method == 'hybrid':
                enhanced = hybrid_enhancement(img)
            elif method == 'advanced_bicubic':
                enhanced = advanced_bicubic(img)
            elif method == 'lanczos':
                enhanced = lanczos_enhancement(img)
            else:  # simple
                enhanced = simple_bicubic(img)

            # Сохраняем улучшенное изображение
            sr_img_path = sr_dir / f"{img_path.stem}_sr{img_path.suffix}"
            cv2.imwrite(str(sr_img_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

            enhanced_count += 1

            if (i + 1) % 10 == 0:
                print(f"  Улучшено: {i + 1}/{len(images)}")

        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")

    return enhanced_count


def copy_labels_from_hr(hr_dir, sr_dir, lr_dir):
    """Копирует аннотации из HR директории в SR"""
    print(f"\nКопирование аннотаций...")

    copied_count = 0

    # Для каждого SR изображения
    for sr_img in sr_dir.glob('*.jpg'):
        # Получаем базовое имя (убираем _sr)
        base_name = sr_img.stem.replace('_sr', '')

        # Ищем аннотацию в HR
        hr_label = hr_dir / f"{base_name}.txt"

        if hr_label.exists():
            sr_label = sr_dir / f"{sr_img.stem}.txt"
            shutil.copy2(hr_label, sr_label)
            copied_count += 1
        else:
            # Пробуем найти в LR
            lr_label = lr_dir / f"{base_name}.txt"
            if lr_label.exists():
                sr_label = sr_dir / f"{sr_img.stem}.txt"
                shutil.copy2(lr_label, sr_label)
                copied_count += 1
            else:
                # Создаем пустой файл
                sr_label = sr_dir / f"{sr_img.stem}.txt"
                sr_label.write_text('')

    print(f"  Скопировано/создано аннотаций: {copied_count}")


def hybrid_enhancement(img):
    """Гибридный метод улучшения"""
    h, w = img.shape[:2]

    # 1. Увеличение с бикубической интерполяцией
    enhanced = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 2. Улучшение резкости
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # 3. Улучшение контраста с CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 4. Подавление шума
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

    return enhanced


def advanced_bicubic(img):
    """Продвинутая бикубическая интерполяция"""
    h, w = img.shape[:2]

    # Бикубическая интерполяция
    enhanced = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Многократное улучшение
    for _ in range(2):
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.3)
        enhanced = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)
        enhanced = cv2.bilateralFilter(enhanced, 3, 25, 25)

    return enhanced


def lanczos_enhancement(img):
    """Lanczos интерполяция (высокое качество)"""
    h, w = img.shape[:2]

    # Lanczos интерполяция
    enhanced = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # Легкое улучшение резкости
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]]) / 9.0
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


def simple_bicubic(img):
    """Простая бикубическая интерполяция"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def update_yaml_files(sr_base):
    """Обновляет или создает YAML файл для SR"""

    # Определяем классы
    classes = determine_classes(sr_base)

    # YAML для SR
    yaml_sr = sr_base / 'coffee_dataset_sr.yaml'
    yaml_sr.write_text(f"""# Coffee Dataset - SR Version
# Super Resolution enhanced images

path: {sr_base}

train: train_sr
val: valid_sr
test: test_sr

nc: {len(classes)}
names: {classes}

# Note: Images enhanced from LR versions
# Original annotations preserved
""")

    print(f"\nСоздан YAML файл для SR: {yaml_sr}")
    print(f"\nДля обучения на SR данных используйте:")
    print(f"  python trainer.py --data {yaml_sr}")


def determine_classes(sr_base):
    """Определяет классы из аннотаций"""
    # Проверяем train_hr директорию
    train_hr_dir = sr_base / 'train_hr'

    if train_hr_dir.exists():
        # Ищем файл classes.txt
        classes_file = train_hr_dir / 'classes.txt'
        if classes_file.exists():
            classes = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
            if classes:
                return classes

    # По умолчанию для кофе
    return ['unripe', 'ripe', 'overripe']


if __name__ == "__main__":
    enhance_yolo_dataset()