import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR
# 2 1af22a2d-0a37-4f6d-a3cc-f4cc140ae3a4
# 1 1eaeb9d0-e396-4801-85d5-f181f7e43169
# 0 2f840d7f-7113-4db4-8d2d-eac5bda1891e
#python detect.py --weights "F:\rice leaf disease\yolov7\runs\train\yolov7-custom5\weights\best.pt" --conf 0.25 --img-size 640 --source 'F:\rice leaf disease\datasets\train\images\0 2f840d7f-7113-4db4-8d2d-eac5bda1891e.jpg'
#"F:\rice leaf disease\dataset\train\images\sheath_blight485_jpg.rf.21b1a9fa1badd61bf6ff5fd2855afd38.jpg"
#"F:\rice leaf disease\dataset\train\images\东格鲁病（Rice Tungro）3_ezgif-frame-121_jpg.rf.6cebc1598f59225d8f86e499770b19cd.jpg"
#"F:\rice leaf disease\dataset\train\images\稻曲病（Rice False Smut）5_20c_jpg.rf.de5083894bd6a5bca8cf5880e0179d1e.jpg"
if __name__ == '__main__':
    model = YOLO(r'F:\rice leaf disease\yolov13-main\runs\Leaf_Scald\weights\best.pt') # select your model.pt path
    #model = RTDETR(r"F:\rice leaf disease\yolov13-main\rtdetr-x\train\weights\best.pt")
    model.predict(source=r'F:\rice leaf disease\Leaf_Scald\train\images\aug_0_7_jpg.rf.6193c60a84c4a39d22e3bb0d3be8b816.jpg',
                  imgsz=1280,
                  project='runs/detect',
                  name='Scald',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )