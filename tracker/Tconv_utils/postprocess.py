from utils.general import scale_coords


def postprocess(out, img, ori_img):
    """
    将YOLO检测结果缩放到原图大小

    Args:
        out: YOLO模型输出
        img: 预处理后的输入图像 (经过resize和padding)
        ori_img: 原始图像

    Returns:
        resized_boxes: 缩放后的检测框 [x1, y1, x2, y2, conf, cls]
    """
    import torch

    out = out[0].boxes
    boxes = out.data  # [x1, y1, x2, y2, conf, cls]

    # 获取图像尺寸
    img_h, img_w = img.shape[2:]  # 预处理后的图像尺寸
    ori_h, ori_w = ori_img.shape[:2]  # 原始图像尺寸

    # 计算缩放比例和填充
    # 假设预处理时保持了宽高比进行resize
    scale = min(img_w / ori_w, img_h / ori_h)
    new_w = int(ori_w * scale)
    new_h = int(ori_h * scale)

    # 计算填充偏移量
    pad_x = (img_w - new_w) / 2
    pad_y = (img_h - new_h) / 2

    if len(boxes) > 0:
        # 复制boxes避免修改原始数据
        resized_boxes = boxes.clone()

        # 调整坐标：去除填充并缩放到原图尺寸
        # x坐标调整
        resized_boxes[:, [0, 2]] = resized_boxes[:, [0, 2]] - pad_x  # 去除x方向填充
        resized_boxes[:, [0, 2]] = resized_boxes[:, [0, 2]] / scale  # 缩放到原图尺寸

        # y坐标调整
        resized_boxes[:, [1, 3]] = resized_boxes[:, [1, 3]] - pad_y  # 去除y方向填充
        resized_boxes[:, [1, 3]] = resized_boxes[:, [1, 3]] / scale  # 缩放到原图尺寸

        # 确保坐标不超出图像边界
        resized_boxes[:, [0, 2]] = torch.clamp(resized_boxes[:, [0, 2]], 0, ori_w)
        resized_boxes[:, [1, 3]] = torch.clamp(resized_boxes[:, [1, 3]], 0, ori_h)

        return resized_boxes
    else:
        return boxes