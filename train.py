from ultralytics import YOLO


model = YOLO('yolo11.yaml').load('yolo11s.pt')

results = model.train(data='./neu.yaml', epochs=30 , workers=0)

# 评估模型在验证集上的性能
result = model.val()