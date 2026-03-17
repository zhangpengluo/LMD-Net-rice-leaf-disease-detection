from linecache import cache
import warnings
from ultralytics import YOLO
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    # 1. 初始化模型
    model = YOLO(r'F:\rice leaf disease\yolov13-main\Detect_MBConv\train\weights\last.pt')

    # 2. 训练模型
    results = model.train(
        data=r'F:\rice leaf disease\datasets\data.yaml',
        epochs=400,
        batch=8,
        imgsz=320,
        device=0,
        patience=100,
        workers=4,
        project="Detect_MBConv",
        optimizer='SGD',
        cache=True,
        resume=True,
    )
    # 3. 模型验证
    metrics = model.val()