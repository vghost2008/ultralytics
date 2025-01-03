from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("--gpus",type=str,default="0",help="src dir")
    args = parser.parse_args()
    return args

args = parse_args()

# Load a model
model = YOLO("yolo11l.pt")

# Train the model
train_results = model.train(
    data="b10cfb.yaml",  # path to dataset YAML
    epochs=300,  # number of training epochs
    imgsz=512,  # training image size
    device=args.gpus,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=32,
    task="segment",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/wj/ai/mldata1/B10CFOD/b_example_input1/BW.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
