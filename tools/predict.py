from ultralytics import YOLO
import yaml
import argparse
import glob
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("model",type=str,help="model path")
    parser.add_argument("config",type=str,help="model path")
    parser.add_argument("data_dir",type=str,help="src dir")
    parser.add_argument("--score-thr",type=float,default=0.5,help="src dir")
    parser.add_argument("--gpus",type=str,default="0",help="src dir")
    args = parser.parse_args()
    return args

def main(args):
    model = YOLO(args.model)
    with open(args.config,"r") as f:
        s = f.read()
        config = yaml.safe_load(s)
        if config is None:
            config = {}
    files = glob.glob(osp.join(args.data_dir,"*.jpg"))
    imgss = config.get('imgsz',512)
    #predict: ultralytics/engine/predictor.py
    for f in files:
        #results = model.predict("bus.jpg", save=True, imgsz=320, conf=0.5)
        results = model.predict(f, save=True, imgsz=imgss, conf=args.score_thr)
        print(results)

if __name__ == "__main__":
    args = parse_args()
    main(args)