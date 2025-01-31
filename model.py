from ultralytics import YOLO
import time
import torch


def main():
    # Load a pretrained model
    model = YOLO('yolov8n-pose.pt').to('cuda')

    # Train the model
    time0 = time.time()

    # Release GPU memory before training
    torch.cuda.empty_cache()

    if __name__ == '__main__':
        model.train(data="C:/Users/jeongsang/yolov8_data/yolo.yaml",
                    epochs=100, batch=64, imgsz=800, device='0')

    time1 = time.time()
    time_to_train = time1 - time0
    print("Training Time:", time_to_train, "seconds")