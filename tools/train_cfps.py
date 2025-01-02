from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.pt")

# Train the model
train_results = model.train(
    data="b10cfps.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=512,  # training image size
    device="3",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=32,
    task="segment",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/wj/ai/mldata1/B10CFOD/example_input/BW.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
