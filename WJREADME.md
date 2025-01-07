# YOLO

## head

ultralytics.nn.modules.head

## build_dataset

ultralytics.models.yolo.detect.train get_dataloader-->build_dataset-->ultralytics.data.build build_yolo_dataset --> ultralytics.data.dataset YOLODataset --> ultralytics.data.base BaseDataset

## transforms

ultralytics.data.dataset::YOLODataset build_transforms  v8_transforms

## img2label_path

ultralytics.data.utils img2label_paths

## read label

ultralytics.data.utils.verify_image_label  

## input channel

模型配置文件中通过参数 ch 配置 (ultralytics.nn.tasks.DetectionModel:315)

## YAML to model

ultralytics.engine.model.Model yaml_model_load(加载配置) -> DetectionModel.__init__ parse_model --> ultralytics.nn.tasks::parse_model


