import torch
import sys
import numpy as np
import copy 
from ultralytics.models.yolo.detect.val import DetectionValidator 
from ultralytics.data.augment import RandomHSV
original_hsv = RandomHSV.__call__

def safe_hsv(self, labels):
    img = labels.get('img')
    if img is not None and len(img.shape) == 3 and img.shape[2] == 6:
        foggy = img[..., :3].copy()
        clean = img[..., 3:].copy()
        
        labels['img'] = foggy                 
        labels = original_hsv(self, labels)   
        labels['img'] = np.concatenate([labels['img'], clean], axis=-1) 
    else:
        labels = original_hsv(self, labels)
    return labels

RandomHSV.__call__ = safe_hsv

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics import YOLO
class FoggyValidator(DetectionValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if batch["img"].shape[1] == 6:
            batch["img"] = batch["img"][:, :3, :, :]
        return batch
    
class FoggyTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        self.phase1_epochs = overrides.pop('phase1_epochs', 50) 
        
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        
        teacher_weight = './ultralytics/weights/pre-tch.pt' 
        
        self.teacher = YOLO(teacher_weight).model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.teacher_to_device = False
        self.teacher_p5 = None  
        
        def intercept_features(module, args):
            self.teacher_p5 = args[0][2].detach()
        self.teacher.model[-1].register_forward_pre_hook(intercept_features)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if "text_anchors" not in batch:
            dataset = getattr(self.train_loader, 'dataset', None)
            if dataset is not None and hasattr(dataset, 'text_anchors'):
                batch["text_anchors"] = dataset.text_anchors

        batch["current_epoch"] = self.epoch 
        batch["phase1_epochs"] = self.phase1_epochs 
        
        if not self.teacher_to_device:
            self.teacher = self.teacher.to(self.device)
            self.teacher_to_device = True
        if batch["img"].shape[1] == 6:
            img_foggy = batch["img"][:, :3, :, :]  
            img_clean = batch["img"][:, 3:, :, :]   
            
            batch["img"] = img_foggy
            batch["img_clean"] = img_clean
        else:
            if "img_clean" not in batch:
                batch["img_clean"] = batch["img"].clone()

        with torch.no_grad():
            _ = self.teacher(batch["img_clean"])
            batch["feat_teacher_p5"] = self.teacher_p5
                
        return batch

    def save_model(self):
        super().save_model() 
        
        if self.epoch == self.phase1_epochs - 1:
            import shutil
            best_path = self.wdir / 'best.pt'
            last_path = self.wdir / 'last.pt'
            
            if best_path.exists():
                shutil.copy(best_path, self.wdir / 'phase1_best.pt')
            if last_path.exists():
                shutil.copy(last_path, self.wdir / 'phase1_last.pt')
    
    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return FoggyValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy.copy(self.args), 
            _callbacks=self.callbacks
        )


if __name__ == '__main__':
    model_yaml_path = "./ultralytics/cfg/models/26/yolo26-fogtrack.yaml"
    pre_model_name = './ultralytics/weights/pre-tch.pt'
    data_yaml_path = "./ultralytics/fogmvtpp-detection"

    overrides = {
        'model': model_yaml_path,
        'pretrained': pre_model_name, 
        'data': data_yaml_path,
        'optimizer': 'AdamW',
        'lr0': 0.001,   
        'lrf': 0.01,
        'warmup_epochs': 3.0, 
        'epochs': 200,           
        'phase1_epochs': 100,    
        'patience': 100,         
        'batch': 32,
        'imgsz': 640,
        'device': '2',
        'name': 'p12-test',
        'mosaic': 1.0,    
        'translate': 0.1,  
        'scale': 0.5,        
        'fliplr': 0.5,      
        'close_mosaic': 10,    
        'degrees': 0.0, 'shear': 0.0, 'perspective': 0.0,
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'bgr': 0.0,
    }
    trainer = FoggyTrainer(overrides=overrides)
    trainer.train()