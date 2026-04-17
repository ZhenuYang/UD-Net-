# ==========================================
# 文件名: preprocess.py
# 功能: 接收单条视频，用MTCNN扣取人脸并转为模型输入Tensor
# ==========================================
import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from torchvision import transforms

# 初始化 MTCNN 检测器 (全局只加载一次，加快速度)
print("Loading MTCNN Detector...")
detector = MTCNN()

# 定义 PyTorch 图像预处理流水线 (转Tensor + ImageNet标准化)
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_and_preprocess(video_path, num_frames=3):
    """
    输入: 视频路径
    输出: 
      - tensor_batch: 形状为 [N, 3, 224, 224] 的 PyTorch Tensor，直接喂给模型
      - display_faces: 原始 RGB 图片列表，用于在 UI 上展示给专家看
    """
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gap = max(1, n_frames // num_frames)

    tensor_list = []
    display_faces = []
    
    attempts, i = 0, 0
    while len(tensor_list) < num_frames and attempts < num_frames * 2 and i < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if success:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            info = detector.detect_faces(img_rgb)
            
            if len(info) > 0:
                x, y, width, height = info[0]['box']
                x, y = max(0, x), max(0, y)
                face_crop = img_rgb[y:y + height, x:x + width, :]
                
                if face_crop.size > 0:
                    # 1. 调整大小为 224x224
                    face_resized = cv2.resize(face_crop, (224, 224))
                    
                    # 2. 存入展示列表 (给界面用)
                    display_faces.append(face_resized)
                    
                    # 3. 转换为 Tensor (给模型用)
                    tensor_data = transform_pipeline(face_resized)
                    tensor_list.append(tensor_data)
        
        i += gap
        attempts += 1

    cap.release()
    
    if len(tensor_list) == 0:
        return None, None
        
    # 拼接成一个 Batch: [N, 3, 224, 224]
    tensor_batch = torch.stack(tensor_list)
    return tensor_batch, display_faces