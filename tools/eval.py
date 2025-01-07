from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("model",type=str,default="yolov8l",help="src dir")
    parser.add_argument("dataset",type=str,default="b10cfito",help="src dir")
    parser.add_argument("--gpus",type=str,default="0",help="src dir")
    parser.add_argument("--epochs",type=int,default=1,help="epochs")
    parser.add_argument("--batch",type=int,default=32,help="batch size")
    args = parser.parse_args()
    return args

args = parse_args()

# Load a model
model = YOLO(args.model)

# Evaluate model performance on the validation set
metrics = model.val(
    data=args.dataset+".yaml",  # path to dataset YAML
    imgsz=512,  # training image size
    device=args.gpus,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=args.batch,
)