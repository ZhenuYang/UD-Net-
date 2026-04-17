# ==========================================
# 文件名: app.py
# 功能: 软件前端UI与主流程入口（演示版：固定输出）
# ==========================================
import gradio as gr
import torch
import numpy as np

# 导入你自己的代码文件
from preprocess import extract_and_preprocess

print("⚠️ 当前为演示模式：模型推理已关闭，输出为固定值")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_video(video_path):
    if video_path is None:
        return "请先上传视频", None
    
    # 1. 仍然保留预处理（用于展示人脸检测效果）
    tensor_batch, display_faces = extract_and_preprocess(video_path, num_frames=3)
    
    if tensor_batch is None:
        return "警告：未能在视频中检测到清晰人脸，请更换光照良好的视频！", None
    
    # =====================================================
    # ❗❗❗ 核心修改：直接固定输出分数（跳过模型）
    final_score = 6.3
    # =====================================================
    
    # 3. 临床分级逻辑（照常使用）
    if final_score < 15:
        severity = "🟢 正常或轻微 (Mild)"
    elif final_score < 30:
        severity = "🟡 中度抑郁症状 (Moderate)"
    else:
        severity = "🔴 重度抑郁症状 (Severe)"
        
    score_text = (
        f"### 📊 综合评估得分：{final_score:.2f} / 63.0\n\n"
        f"### 🩺 临床辅助分级：{severity}\n\n"
        f"*(注：当前为演示模式，结果为固定输出)*"
    )
    
    return score_text, display_faces


# ==========================================
# 软件UI构建
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 UD-Net 视频抑郁症智能辅助评估系统")
    gr.Markdown("基于轻量级 ResNet-18 与 MTCNN 联合构建的临床端辅助推理软件。")
    gr.Markdown("⚠️ 当前为演示模式：模型未启用，结果为固定输出")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 临床数据录入")
            video_input = gr.Video(label="请上传患者访谈视频 (.mp4)")
            analyze_btn = gr.Button("🚀 启动自动化分析流水线", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. UD-Net 分析报告")
            score_output = gr.Markdown("### 等待数据录入与分析...")
            
    with gr.Row():
        gr.Markdown("### 3. 底层特征抓取监控 (MTCNN)")
        face_gallery = gr.Gallery(label="有效面部关键帧提取结果", columns=3, height="auto")

    analyze_btn.click(
        fn=analyze_video,
        inputs=video_input,
        outputs=[score_output, face_gallery]
    )

if __name__ == "__main__":
    app.launch()