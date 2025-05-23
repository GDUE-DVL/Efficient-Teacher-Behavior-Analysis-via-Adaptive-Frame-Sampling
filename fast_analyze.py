#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter, deque

# 导入检测模型和姿态估计函数
from models.yolo import Model
from utils.general import check_img_size, non_max_suppression_kpt, scale_coords_kpt

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    """加载YOLO姿态检测模型"""
    model = Model(model_path, int8=False, device=device)
    model.eval()
    return model

def process_frame(model, frame, confidence_threshold=0.2):
    """
    处理单帧图像，提取人体关键点
    
    Args:
        model: YOLO姿态检测模型
        frame: 输入图像帧
        confidence_threshold: 关键点置信度阈值
        
    Returns:
        keypoints: 关键点列表 [(x, y, confidence), ...]
        confidence: 平均置信度
    """
    # 预处理图像
    img = cv2.resize(frame, (640, 640))
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推理
    with torch.no_grad():
        outputs = model(img)
        outputs = non_max_suppression_kpt(outputs, 0.25, 0.65, nc=model.yaml['nc'], kpt_shape=(17, 3))

    if len(outputs[0]) == 0:
        return None, 0.0
    
    # 获取最佳人体检测结果（面积最大）
    best_person = max(outputs[0], key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    kpts = best_person[6:].detach().cpu().numpy().reshape(-1, 3)
    
    # 转换坐标到原始图像尺寸
    orig_h, orig_w = frame.shape[:2]
    scale_x, scale_y = orig_w / 640, orig_h / 640
    
    keypoints = []
    total_conf = 0
    valid_points = 0
    
    for kpt in kpts:
        x, y, conf = kpt
        x *= scale_x
        y *= scale_y
        
        if conf > confidence_threshold:
            keypoints.append((x, y, conf))
            total_conf += conf
            valid_points += 1
        else:
            keypoints.append(None)
    
    avg_conf = total_conf / valid_points if valid_points > 0 else 0
    
    return keypoints, avg_conf

def analyze_behavior(keypoints):
    """
    分析行为类型
    
    Args:
        keypoints: 关键点列表 [(x, y, confidence), ...]
        
    Returns:
        behavior: 行为类型字符串
    """
    if keypoints is None or len(keypoints) < 17:
        return "unknown"
    
    # 关键点索引
    nose = keypoints[0]
    left_ear = keypoints[3]
    right_ear = keypoints[4]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    # 计算写黑板得分
    writing_score = 0
    # 手臂抬高 - 肘部高于肩膀
    if (left_elbow and left_shoulder and left_elbow[1] < left_shoulder[1]) or \
       (right_elbow and right_shoulder and right_elbow[1] < right_shoulder[1]):
        writing_score += 1
    
    # 手臂伸展 - 手腕延伸
    if (left_wrist and left_elbow and left_shoulder and 
        left_wrist[0] < left_elbow[0] < left_shoulder[0]):
        writing_score += 1
    if (right_wrist and right_elbow and right_shoulder and 
        right_wrist[0] > right_elbow[0] > right_shoulder[0]):
        writing_score += 1
    
    # 侧对相机 - 一个耳朵可见性高于另一个
    if left_ear and right_ear and abs(left_ear[2] - right_ear[2]) > 0.3:
        writing_score += 0.5
    
    # 计算讲解得分
    explaining_score = 0
    # 面向相机 - 两个耳朵可见性接近
    if left_ear and right_ear and abs(left_ear[2] - right_ear[2]) < 0.2:
        explaining_score += 1
    
    # 手势 - 手臂抬起但不太高
    if (left_elbow and left_shoulder and left_hip and 
        left_shoulder[1] < left_elbow[1] < left_hip[1]):
        explaining_score += 0.5
    if (right_elbow and right_shoulder and right_hip and 
        right_shoulder[1] < right_elbow[1] < right_hip[1]):
        explaining_score += 0.5
    
    # 计算移动得分
    moving_score = 0
    
    # 最简单的基于得分的判断
    if writing_score >= 2:
        return "writing"
    elif explaining_score >= 1.5:
        return "explaining"
    elif moving_score >= 1:
        return "moving"
    else:
        return "standing"

def analyze_video(video_path, output_dir=None):
    """
    分析视频并生成报告
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录路径
    """
    print(f"分析视频: {video_path}")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model_path = "weights/yolo7-w6-pose.pt"
    if not os.path.exists(model_path):
        model_path = "weights/yolov7-w6-pose.pt"
    
    print(f"加载YOLO姿态检测模型: {model_path}")
    model = load_model(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {total_frames}, 时长: {duration/60:.2f}分钟")
    
    # 初始化变量
    behavior_history = []
    confidence_history = []
    behavior_buffer = deque(maxlen=5)  # 行为平滑缓冲区
    
    # 定义抽样间隔
    sample_interval = 10  # 每10帧处理一次
    
    # 处理视频
    pbar = tqdm(total=total_frames, desc="处理中")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 抽样处理
        if frame_idx % sample_interval == 0:
            # 处理帧
            keypoints, confidence = process_frame(model, frame)
            
            # 分析行为
            behavior = analyze_behavior(keypoints) if keypoints is not None else "unknown"
            
            # 更新历史记录
            confidence_history.append(confidence)
            
            # 平滑行为
            behavior_buffer.append(behavior)
            if len(behavior_buffer) > 0:
                behavior_counts = Counter(behavior_buffer)
                smoothed_behavior = behavior_counts.most_common(1)[0][0]
                behavior_history.append(smoothed_behavior)
            else:
                behavior_history.append(behavior)
            
            # 如果行为变化，输出信息
            if len(behavior_history) > 1 and behavior_history[-1] != behavior_history[-2]:
                print(f"行为变化: {behavior_history[-2]} -> {behavior_history[-1]}")
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # 计算行为统计
    behavior_counts = Counter(behavior_history)
    total_analyzed_frames = len(behavior_history)
    
    # 显示结果
    print("\n行为分析结果:")
    for behavior, count in behavior_counts.items():
        seconds = count * sample_interval / fps
        percentage = count / total_analyzed_frames * 100
        print(f"  {behavior}: {seconds:.2f}秒 ({percentage:.1f}%)")
    
    # 创建行为占比图
    plt.figure(figsize=(10, 6))
    
    # 只保留有数据的行为
    behaviors = []
    percentages = []
    for behavior, count in behavior_counts.items():
        if count > 0:
            behaviors.append(behavior)
            percentages.append(count / total_analyzed_frames * 100)
    
    # 绘制饼图
    plt.pie(percentages, labels=behaviors, autopct='%1.1f%%', startangle=90)
    plt.title(f'教师行为分布 - {os.path.basename(video_path)}')
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_behavior_chart.png")
    plt.savefig(chart_path)
    plt.close()
    
    print(f"\n分析完成! 图表已保存到: {chart_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python fast_analyze.py <视频文件路径>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"错误: 视频文件 {video_path} 不存在")
        sys.exit(1)
    
    analyze_video(video_path) 