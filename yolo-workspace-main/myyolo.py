import torch
import cv2
import numpy as np
import json
import os
from datetime import datetime

# 定义路径
folder_path = "yolo-workspace-main/images-file"  # 图片文件夹
output_folder = "yolo-workspace-main/detected-images"  # 结果输出文件夹
json_path = "yolo-workspace-main/results.json"  # JSON 结果文件
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")  # 图片格式

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# 载入 YOLOv5 预训练模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo-workspace-main/model/best.pt', source='github')

# 存储所有检测结果
all_results = []

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        continue

    print(f"✅ 处理图片: {image_path}, 大小: {image.shape}")
    
    # 进行目标检测
    results = model(image)

    # 解析 YOLO 结果
    boxes = results.xyxy[0].cpu().numpy()

    helmet_detected = False

    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box[:4].astype(int).tolist() + [float(box[4]), int(box[5])]

        # 检查是否是安全帽（假设模型的 "helmet" 类别名称是 "helmet"）
        if model.names[class_id] == "helmet" and conf > 0.4:
            helmet_detected = True
            break  # 只要有一个安全帽检测通过，就可以停止判断

    # 生成 JSON 结果
    all_results.append({
        "file_name": image_file,
        "event_type": "有戴安全帽" if helmet_detected else "未戴安全帽",
        "date": datetime.now().isoformat()
    })

# 保存 JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print("✅ 结果已保存:", json_path)
