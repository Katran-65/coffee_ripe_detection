import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys


def train_yolo_sr():
    """
    Обучение YOLO на SR датасете из YOLO данных
    """

    parser = argparse.ArgumentParser(description='Обучение YOLO на SR датасете')
    parser.add_argument('--version', type=str, choices=['hr', 'lr', 'sr'], default='hr',
                        help='Версия данных: hr - оригиналы, lr - ухудшенные, sr - улучшенные')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Модель YOLO')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Размер изображения')
    parser.add_argument('--batch', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--name', type=str, default='experiment',
                        help='Имя эксперимента')
    parser.add_argument('--data-dir', type=str, default='datas_sr_yolo',
                        help='Директория с SR датасетом')

    args = parser.parse_args()

    # Проверяем существование датасета
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ОШИБКА: Директория датасета не найдена: {data_dir}")
        print("Сначала запустите: python create_sr_from_yolo.py")
        return

    # Определяем YAML файл в зависимости от версии
    yaml_file = data_dir / f'coffee_dataset_{args.version}.yaml'

    if not yaml_file.exists():
        print(f"ОШИБКА: YAML файл не найден: {yaml_file}")
        print("Доступные версии:")
        for ver in ['hr', 'lr', 'sr']:
            test_yaml = data_dir / f'coffee_dataset_{ver}.yaml'
            if test_yaml.exists():
                print(f"  - {ver}: {test_yaml}")
        return

    print(f"\nВерсия данных: {args.version.upper()}")
    print(f"YAML файл: {yaml_file}")
    print(f"Директория: {data_dir}")

    # Проверяем GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Используется CPU")

    # Загружаем модель
    print(f"\nЗагрузка модели: {args.model}")
    model = YOLO(args.model)

    # Параметры обучения
    train_params = {
        'data': str(yaml_file),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': 4,
        'device': device,
        'patience': 10,
        'save': True,
        'verbose': True,
        'project': 'coffee_detection',
        'name': f"{args.name}_{args.version}",
        'exist_ok': True,
        'plots': True,
    }

    print(f"\nПараметры обучения:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)

    try:
        # Запуск обучения
        results = model.train(**train_params)

        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)

        return results

    except Exception as e:
        print(f"\nОШИБКА при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    train_yolo_sr()