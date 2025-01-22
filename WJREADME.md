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

ultralytics.engine.model.Model yaml_model_load(加载配置) -> ultralytics.nn.tasks::DetectionModel.__init__ parse_model --> ultralytics.nn.tasks::parse_model

## train
训练时通过: 
下列步骤更新模型nc,ch等信息
   File "./tools/train.py", line 24, in <module>
    train_results = model.train(

   File "/home/wj/ai/work/ultralytics/ultralytics/engine/model.py", line 813, in train
    self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)

   File "/home/wj/ai/work/ultralytics/ultralytics/models/yolo/detect/train.py", line 88, in get_model
    model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)

   File "/home/wj/ai/work/ultralytics/ultralytics/nn/tasks.py", line 307, in __init__
    super().__init__()

## 记录graph日志

  File "./tools/train.py", line 24, in <module>
    train_results = model.train(

  File "/home/wj/ai/work/ultralytics/ultralytics/engine/model.py", line 817, in train
    self.trainer.train()

  File "/home/wj/ai/work/ultralytics/ultralytics/engine/trainer.py", line 207, in train
    self._do_train(world_size)

  File "/home/wj/ai/work/ultralytics/ultralytics/engine/trainer.py", line 330, in _do_train
    self.run_callbacks("on_train_start")

  File "/home/wj/ai/work/ultralytics/ultralytics/engine/trainer.py", line 168, in run_callbacks
    callback(self)

  File "/home/wj/ai/work/ultralytics/ultralytics/utils/callbacks/tensorboard.py", line 84, in on_train_start
    _log_tensorboard_graph(trainer)


