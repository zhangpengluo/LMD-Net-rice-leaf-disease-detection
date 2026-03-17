import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# ================= 配置区域 =================
# 注意：这里是针对单张图片进行增强，所以路径配置与之前有所不同
# 请替换为你的图片和标签文件的实际路径和名称
TARGET_IMAGE_PATH = r"F:\rice leaf disease\datasets\train\images\a42ff541-61e8-44ac-b03e-6e96eb47f02b.jpg"
TARGET_LABEL_PATH = r"F:\rice leaf disease\datasets\train\labels\a42ff541-61e8-44ac-b03e-6e96eb47f02b.txt"

# 增强结果的输出目录
OUTPUT_DIR = r'F:\rice leaf disease\datasets\augmented_single_image'

# ===========================================

def make_output_dirs():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def cv_imread(file_path):
    """支持中文路径的图片读取"""
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

def cv_imwrite(file_path, img):
    """支持中文路径的图片保存"""
    suffix = os.path.splitext(file_path)[-1]
    cv2.imencode(suffix, img)[1].tofile(file_path)

def load_labels(path):
    """加载YOLO格式标签"""
    if not os.path.exists(path):
        return np.array([])
    try:
        with open(path, 'r', encoding='utf-8') as f:
            labels = [line.strip().split() for line in f.readlines() if line.strip()]
        if not labels:
            return np.array([])
        return np.array(labels, dtype=np.float32)
    except Exception as e:
        print(f"读取标签失败: {path}, 错误: {e}")
        return np.array([])

def save_labels(labels, path):
    """保存YOLO格式标签"""
    with open(path, 'w', encoding='utf-8') as f:
        for label in labels:
            line = f"{int(label[0])} {' '.join([f'{x:.6f}' for x in label[1:]])}\n"
            f.write(line)

# --- 数据增强变换函数 (与之前代码保持一致) ---
def flip_horiz(img, labels):
    img = cv2.flip(img, 1)
    if len(labels) > 0:
        labels[:, 1] = 1.0 - labels[:, 1]
    return img, labels

def translate(img, labels, limit=0.1):
    h, w = img.shape[:2]
    tx, ty = random.uniform(-limit, limit) * w, random.uniform(-limit, limit) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h))
    if len(labels) > 0:
        labels[:, 1] += tx / w
        labels[:, 2] += ty / h
        # 移除超出边界的标签
        mask = (labels[:, 1] >= 0) & (labels[:, 1] <= 1) & \
               (labels[:, 2] >= 0) & (labels[:, 2] <= 1) & \
               (labels[:, 3] > 0) & (labels[:, 4] > 0) # 宽度和高度必须大于0
        labels = labels[mask]
    return img, labels

def rotate(img, labels, angle_limit=10):
    angle = random.uniform(-angle_limit, angle_limit)
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    if len(labels) > 0:
        # 获取边界框的四个角点
        new_labels = []
        for label in labels:
            cls, x_c, y_c, box_w, box_h = label
            # 转换回像素坐标
            x_c, y_c, box_w, box_h = x_c * w, y_c * h, box_w * w, box_h * h

            x1, y1 = x_c - box_w / 2, y_c - box_h / 2
            x2, y2 = x_c + box_w / 2, y_c + box_h / 2

            points = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x1, y2, 1],
                [x2, y2, 1]
            ]).T # 转换为 (3, 4) 矩阵

            # 旋转所有角点
            rotated_points = np.dot(M, points) # (2, 4)

            # 找到旋转后边界框的最小/最大坐标
            min_x, min_y = np.min(rotated_points[0, :]), np.min(rotated_points[1, :])
            max_x, max_y = np.max(rotated_points[0, :]), np.max(rotated_points[1, :])

            # 转换回归一化坐标
            new_x_c = (min_x + max_x) / 2 / w
            new_y_c = (min_y + max_y) / 2 / h
            new_box_w = (max_x - min_x) / w
            new_box_h = (max_y - min_y) / h

            # 检查边界和有效性
            if 0 <= new_x_c <= 1 and 0 <= new_y_c <= 1 and new_box_w > 0 and new_box_h > 0:
                new_labels.append([cls, new_x_c, new_y_c, new_box_w, new_box_h])
        labels = np.array(new_labels, dtype=np.float32) if new_labels else np.array([])
    return img, labels


def brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_noise(img):
    # 确保噪声与图像数据类型匹配
    noise = np.random.normal(0, 5, img.shape).astype(np.int16) # 先用int16防止溢出
    noisy_img = cv2.add(img.astype(np.int16), noise)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def blur(img):
    size = random.randint(10, 20)
    kernel = np.zeros((size, size))
    mode = random.choice(['h', 'v', 'd1', 'd2'])
    if mode == 'h':
        kernel[int((size - 1) / 2), :] = np.ones(size)
    elif mode == 'v':
        kernel[:, int((size - 1) / 2)] = np.ones(size)
    elif mode == 'd1':
        for i in range(size): kernel[i, i] = 1
    else:
        for i in range(size): kernel[i, size - 1 - i] = 1
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)

# --- 主逻辑 ---
def main():
    make_output_dirs()

    img_name = os.path.basename(TARGET_IMAGE_PATH)
    base_name = os.path.splitext(img_name)[0]

    img_raw = cv_imread(TARGET_IMAGE_PATH)
    if img_raw is None:
        print(f"无法读取目标图片: {TARGET_IMAGE_PATH}")
        return

    label_raw = load_labels(TARGET_LABEL_PATH)

    print(f"开始对图片 '{img_name}' 进行增强预览...")

    # --- 1. 原始图片 (作为对照) ---
    cv_imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_original.jpg"), img_raw)
    if len(label_raw) > 0:
        save_labels(label_raw, os.path.join(OUTPUT_DIR, f"{base_name}_original.txt"))
    print(f"已保存原始图片和标签。")

    # --- 2. 水平翻转 ---
    img_flipped, labels_flipped = flip_horiz(img_raw.copy(), label_raw.copy())
    cv_imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_flip_horiz.jpg"), img_flipped)
    if len(labels_flipped) > 0:
        save_labels(labels_flipped, os.path.join(OUTPUT_DIR, f"{base_name}_flip_horiz.txt"))
    print(f"已保存水平翻转结果。")

    # --- 3. 平移 ---
    img_translated, labels_translated = translate(img_raw.copy(), label_raw.copy())
    cv_imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_translate.jpg"), img_translated)
    if len(labels_translated) > 0:
        save_labels(labels_translated, os.path.join(OUTPUT_DIR, f"{base_name}_translate.txt"))
    print(f"已保存平移结果。")

    # --- 4. 旋转 ---
    img_rotated, labels_rotated = rotate(img_raw.copy(), label_raw.copy())
    cv_imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_rotate.jpg"), img_rotated)
    if len(labels_rotated) > 0:
        save_labels(labels_rotated, os.path.join(OUTPUT_DIR, f"{base_name}_rotate.txt"))
    print(f"已保存旋转结果。")

    # --- 5. 亮度对比度调整 ---
    img_bc = brightness_contrast(img_raw.copy())
    # 亮度对比度不改变标签，所以直接使用原始标签
    cv_imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_brightness_contrast.jpg"), img_bc)
    if len(label_raw) > 0:
        save_labels(label_raw, os.path.join(OUTPUT_DIR, f"{base_name}_brightness_contrast.txt"))
    print(f"已保存亮度对比度调整结果。")

    # --- 6. 添加噪声 ---
    img_noise = add_noise(img_raw.copy())
    # 噪声不改变标签
    cv_imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_add_noise.jpg"), img_noise)
    if len(label_raw) > 0:
        save_labels(label_raw, os.path.join(OUTPUT_DIR, f"{base_name}_add_noise.txt"))
    print(f"已保存添加噪声结果。")

    # --- 7. 模糊 (运动模糊) ---
    img_blur = blur(img_raw.copy())
    # 模糊不改变标签
    cv_imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_blur.jpg"), img_blur)
    if len(label_raw) > 0:
        save_labels(label_raw, os.path.join(OUTPUT_DIR, f"{base_name}_blur.txt"))
    print(f"已保存模糊结果。")

    print("\n所有单张图片增强预览任务已完成！结果保存在:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()