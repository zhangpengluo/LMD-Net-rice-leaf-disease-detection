from ultralytics import YOLO


def main():
    # 1. 加载你的模型权重
    model = YOLO(r'F:\rice leaf disease\yolov13-main\LAE\train2\weights\best.pt')

    # 2. 执行验证
    metrics = model.val(
        data=r'F:\rice leaf disease\datasets\data.yaml',  # 确保路径正确
        workers=4,  # 如果还报错，可以先改为 0 试试
        device='0'
    )

    print("验证完成！")


# --- 关键部分：必须加上这个判断 ---
if __name__ == '__main__':
    main()