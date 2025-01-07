from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("--gpus",type=str,default="0",help="src dir")
    parser.add_argument("--model",type=str,default="yolo11l-seg",help="src dir")
    args = parser.parse_args()
    return args

args = parse_args()

# Load a model
model = YOLO(args.model+".pt")


# Train the model
train_results = model.train(
    data="b10cfg.yaml",  # path to dataset YAML
    epochs=400,  # number of training epochs
    imgsz=512,  # training image size
    device=args.gpus,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=32,
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/wj/ai/mldata1/B10CFOD/example_input/BW.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
