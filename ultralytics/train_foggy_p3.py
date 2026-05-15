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
        img_current = img[..., :3].copy()
        img_prev = img[..., 3:].copy()
        labels['img'] = img_current                 
        labels = original_hsv(self, labels)   
        labels['img'] = np.concatenate([labels['img'], img_prev], axis=-1) 
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

class VideoFoggyTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.student_p5_prev = None
        self.hook_registered = False
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if "text_anchors" not in batch:
            dataset = getattr(self.train_loader, 'dataset', None)
            if dataset is not None and hasattr(dataset, 'text_anchors'):
                batch["text_anchors"] = dataset.text_anchors
        batch["is_phase3"] = True 
        if batch["img"].shape[1] == 6:
            img_current = batch["img"][:, :3, :, :]  
            img_prev = batch["img"][:, 3:, :, :]             
            batch["img"] = img_current
            batch["img_prev"] = img_prev

        if "img_prev" in batch:
            img_prev_norm = batch["img_prev"].to(self.device).float() / 255.0

            if not self.hook_registered and hasattr(self, 'model'):
                self.model.model[-1].register_forward_pre_hook(
                    lambda m, args: setattr(self, 'student_p5_prev', args[0][2].detach())
                )
                self.hook_registered = True          
            with torch.no_grad():
                _ = self.model(img_prev_norm)
                batch["feat_student_prev_p5"] = self.student_p5_prev
                
        return batch

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return FoggyValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy.copy(self.args), 
            _callbacks=self.callbacks
        )
if __name__ == '__main__':
    phase1_2_best_weight = "./ultralytics/runs/detect/p12-test/weights/best.pt"
    overrides = {
        'model': phase1_2_best_weight, 
        'data': "./ultralytics/fogmvtpp-detection.yaml",
        'epochs': 100,     
        'batch': 32,
        'imgsz': 640,
        'device': '3',
        'name': 'p3-test',
        'lr0': 0.0001,     
        'lrf': 0.1,       
        'warmup_epochs': 0.0, 
        'mosaic': 0.0, 'fliplr': 0.0, 'flipud': 0.0,
        'translate': 0.0, 'scale': 0.0,  
        'degrees': 0.0, 'shear': 0.0, 'perspective': 0.0,     
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'bgr': 0.0,
    }
    
    trainer = VideoFoggyTrainer(overrides=overrides)
    trainer.train()