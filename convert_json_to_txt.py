import json
import os
import shutil
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    category_dict = {}
    for category in coco_data['categories']:
        category_dict[category['id']] = category['name']

    all_categories = sorted(set(category_dict.values()))
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(all_categories)}

    for cls_name, cls_id in class_to_id.items():
        print(f"  {cls_name} -> {cls_id}")

    with open(os.path.join(output_dir, 'classes.names'), 'w') as f:
        for cls_name in all_categories:
            f.write(f"{cls_name}\n")

    image_annotations = {}
    for image in coco_data['images']:
        image_annotations[image['id']] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height'],
            'annotations': []
        }

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_annotations:
            image_annotations[image_id]['annotations'].append(ann)

    for img_id, img_data in image_annotations.items():
        img_file = img_data['file_name']
        img_annotations = img_data['annotations']
        img_width = img_data['width']
        img_height = img_data['height']

        src_img_path = os.path.join(images_dir, img_file)
        dst_img_path = os.path.join(output_dir, img_file)

        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            continue

        yolo_lines = []
        for ann in img_annotations:
            category_name = category_dict.get(ann['category_id'], 'unknown')
            class_id = class_to_id[category_name]

            bbox = ann['bbox']
            x_min, y_min, bbox_width, bbox_height = bbox

            x_center = (x_min + bbox_width / 2) / img_width
            y_center = (y_min + bbox_height / 2) / img_height
            width_norm = bbox_width / img_width
            height_norm = bbox_height / img_height

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_lines.append(yolo_line)

        txt_filename = os.path.splitext(img_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    jpg_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    txt_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    print(f"Изображений: {len(jpg_files)}, аннотаций: {len(txt_files)}")

dataset_root = 'datas'

train_json = os.path.join(dataset_root, 'train', '_annotations.coco.json')
valid_json = os.path.join(dataset_root, 'valid', '_annotations.coco.json')
test_json = os.path.join(dataset_root, 'test', '_annotations.coco.json')

convert_coco_to_yolo(
    coco_json_path=train_json,
    images_dir=os.path.join(dataset_root, 'train'),
    output_dir=os.path.join(dataset_root, 'yolo_train')
)

convert_coco_to_yolo(
    coco_json_path=valid_json,
    images_dir=os.path.join(dataset_root, 'valid'),
    output_dir=os.path.join(dataset_root, 'yolo_valid')
)

convert_coco_to_yolo(
    coco_json_path=test_json,
    images_dir=os.path.join(dataset_root, 'test'),
    output_dir=os.path.join(dataset_root, 'yolo_test')
)