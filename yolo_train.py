from ultralytics import YOLO

# 加载并训练模型
model = YOLO('yolov8n.pt')

# 训练参数设置
model.train(
    data = 'D:\\project\\safe\\Safetyhelmet\\safety-Helmet-Reflective-Jacket\\data.yaml',      # 数据集配置文件
    epochs = 100,               # 训练轮数
    imgsz = 640,               # 图像大小
    batch = 4,                # 批次大小
    workers = 4,               # 数据加载的线程数
    device = 'cpu'                 # GPU设备号，使用CPU则设为'cpu'
)

# 验证模型
model.val()