import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO

dataset_root = 'datas'
dataset_yaml = os.path.join(dataset_root, 'coffee_dataset.yaml')

def plot_training_results(results_path='coffee_detection/baseline_run'):
    results_csv = os.path.join(results_path, 'results.csv')

    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('YOLOv8 Training Metrics', fontsize=16)

        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        axes[0, 0].set_title('Bounding Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss')
        axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
        axes[0, 2].set_title('Distribution Focal Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='green')
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='orange')
        axes[1, 0].set_title('mAP Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='red')
        axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='blue')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        axes[1, 2].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('LR')
        axes[1, 2].grid(True)
        axes[1, 2].set_yscale('log')

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()


plot_training_results()
model = YOLO('coffee_detection/baseline_run/weights/best.pt')
metrics = model.val(data=dataset_yaml, split='test')

print(f"\nРезультаты на тестовом наборе:")
print(f"mAP50-95: {float(metrics.box.map):.4f}")
print(f"mAP50: {float(metrics.box.map50):.4f}")
print(f"Precision: {float(metrics.box.p.mean()):.4f}")
print(f"Recall: {float(metrics.box.r.mean()):.4f}")

def visualize_predictions(model, test_images_dir, num_images=6, save_dir='results_visualization'):

    os.makedirs(save_dir, exist_ok=True)

    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    import random
    selected_images = random.sample(test_images, min(num_images, len(test_images)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('YOLOv8 Detection Results on Test Images', fontsize=16, y=1.02)

    axes = axes.flatten()

    for idx, img_name in enumerate(selected_images):
        if idx >= len(axes):
            break

        img_path = os.path.join(test_images_dir, img_name)
        results = model(img_path)
        plotted = results[0].plot(line_width=2, font_size=12)

        if isinstance(plotted, np.ndarray):
            axes[idx].imshow(plotted)
        else:
            axes[idx].imshow(plotted)

        axes[idx].set_title(f'Image: {img_name[:15]}...' if len(img_name) > 15 else f'Image: {img_name}')
        axes[idx].axis('off')

        result_path = os.path.join(save_dir, f'detected_{img_name}')
        if isinstance(plotted, np.ndarray):
            Image.fromarray(plotted).save(result_path)
        else:
            plotted.save(result_path)

    for idx in range(len(selected_images), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('detection_results_grid.png', dpi=150, bbox_inches='tight')
    plt.show()

test_images_dir = os.path.join(dataset_root, 'test', 'images')
if os.path.exists(test_images_dir):
    visualize_predictions(model, test_images_dir, num_images=6)
else:
    test_images_dir = os.path.join(dataset_root, 'test')
    if os.path.exists(test_images_dir):
        visualize_predictions(model, test_images_dir, num_images=6)

def plot_final_metrics(metrics):

    metrics_names = ['mAP50-95', 'mAP50', 'Precision', 'Recall']
    metrics_values = [
        float(metrics.box.map),
        float(metrics.box.map50),
        float(metrics.box.p.mean()),
        float(metrics.box.r.mean())
    ]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)

    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.title('Final Model Performance Metrics', fontsize=16, pad=20)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y', alpha=0.3)

    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good threshold (0.7)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold (0.5)')

    plt.legend()
    plt.tight_layout()
    plt.savefig('final_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_final_metrics(metrics)

try:
    results_dir = 'coffee_detection/baseline_run'
    conf_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')

    if os.path.exists(conf_matrix_path):
        conf_img = Image.open(conf_matrix_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_img)
        plt.title('Confusion Matrix', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"\nОшибка при загрузке confusion matrix: {e}")
