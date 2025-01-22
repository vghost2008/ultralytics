from ultralytics import YOLO
import argparse
import torch
from datetime import datetime
import wml.wtorch.utils as wtu

def parse_args():
    parser = argparse.ArgumentParser(description="load ckpt")
    parser.add_argument("-c","--config",type=str,default="yolov8l",help="config file")
    parser.add_argument("-ckpt","--check-point",type=str,help="ckpt file")
    parser.add_argument("-save","--save-path",type=str,default="tmp.pt",help="save path")
    parser.add_argument("-nc",type=int,help="num classes")
    parser.add_argument("-ch",type=int,default=3,help="num in channels")
    args = parser.parse_args()
    return args

args = parse_args()

if "." not in args.config:
    args.config = args.config + ".yaml"
cfg = {"nc":args.nc,"ch":args.ch}
model = YOLO(args.config,cfg=cfg)
state_dict = torch.load(args.check_point)
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
wtu.forgiving_state_restore(model.model,state_dict)
torch.save(
            {
                "model": model.model,  # resume and final checkpoints derive from EMA
                "ema": None, #deepcopy(self.ema.ema).half(),
                "date": datetime.now().isoformat(),
            },
            args.save_path,
        )
print(f"Save path: {args.save_path}")