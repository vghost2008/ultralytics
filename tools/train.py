from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("-ds","--dataset",type=str,default="b10cfito",help="src dir")
    parser.add_argument("--gpus",type=str,default="0",help="src dir")
    parser.add_argument("--model",type=str,default="yolov8l",help="src dir")
    parser.add_argument("--epochs",type=int,default=1,help="epochs")
    parser.add_argument("--batch",type=int,default=32,help="batch size")
    args = parser.parse_args()
    return args

args = parse_args()

# Load a model
#model = YOLO(args.model+".pt")
model = YOLO("yolov8l.yaml")

# Train the model
#ultralytics/engine/model.py
train_results = model.train(
    data=args.dataset+".yaml",  # path to dataset YAML
    epochs=args.epochs,  # number of training epochs
    imgsz=512,  # training image size
    device=args.gpus,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=args.batch,
    optimizer="AdamW",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/wj/ai/mldata1/B10AOIOLD/example_input/AD.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
