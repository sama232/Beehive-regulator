from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")
    results = model.train(data='C:/Users/Nadine Mostafa/Desktop/bees vs wasps/config.yml', epochs=100, workers=2)