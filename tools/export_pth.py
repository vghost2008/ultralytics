from ultralytics import YOLO
import argparse
import os.path as osp
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("model",type=str,default="yolov8l.pt",help="pt model")
    parser.add_argument("--save-path",type=str,default=None,help="save path")
    args = parser.parse_args()
    return args

args = parse_args()

# Load a model
model = YOLO(args.model)

if args.save_path is None:
    args.save_path = osp.splitext(args.model)[0]+".pth"

state_dict = model.model.state_dict()
print(f"save keys {list(state_dict.keys())}")
torch.save(state_dict,args.save_path)
