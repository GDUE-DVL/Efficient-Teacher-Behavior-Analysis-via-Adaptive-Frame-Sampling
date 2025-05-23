import os
import sys
import time
import cv2
import json
import math
import numpy as np
from collections import defaultdict, deque, Counter
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw, ImageFont
import datetime
import pickle
import hashlib
import gc  # 添加gc模块导入，用于内存管理
from skimage import measure, draw

# 配置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm
import torch

# 添加YOLOv12路径
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
yolo_dir = os.path.join(parent_dir, 'yolov12-main')
if os.path.exists(yolo_dir):
    sys.path.append(yolo_dir)
    
# 字体文件
chinese_font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts/simhei.ttf")

# 检查系统平台
import platform
is_windows = platform.system() == "Windows"

# 使用PIL提供中文支持
def put_chinese_text(img, text, pos, font_size=30, color=(0, 0, 0)):
    """在图像上绘制中文文本"""
    # 将OpenCV图像(BGR)转换为PIL图像(RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 选择字体
    try:
        # 尝试使用系统中的中文字体
        fontpath = fm.findfont(fm.FontProperties(family=['SimHei', 'Microsoft YaHei', 'SimSun']))
        font = ImageFont.truetype(fontpath, font_size)
    except:
        # 如果找不到字体，使用默认字体
        font = ImageFont.load_default()
    
    # 在PIL图像上绘制文本
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color[::-1])  # PIL颜色顺序为RGB
    
    # 将PIL图像转回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class PoseSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.points_history = {}
        self.latest_length = 0  # 添加变量追踪最近的点列表长度
    
    def update(self, person_id, points):
        # 确保输入是有效的点列表
        if points is None or len(points) == 0:
            return None
        
        # 更新最新长度
        self.latest_length = max(self.latest_length, len(points))
        
        # 初始化历史记录
        if person_id not in self.points_history:
            self.points_history[person_id] = [deque(maxlen=self.window_size) for _ in range(self.latest_length)]
        
        # 确保历史记录数组足够大
        if len(self.points_history[person_id]) < len(points):
            # 扩展历史记录数组以适应新的点数量
            additional = len(points) - len(self.points_history[person_id])
            self.points_history[person_id].extend([deque(maxlen=self.window_size) for _ in range(additional)])
        
        # 安全地添加点到历史记录
        try:
            for i, point in enumerate(points):
                if i < len(self.points_history[person_id]):
                    self.points_history[person_id][i].append(point)
        except Exception as e:
            print(f"平滑器更新点时出错: {e}")
        
        return self.get_smoothed_points(person_id)
    
    def get_smoothed_points(self, person_id):
        if person_id not in self.points_history:
            return None
        
        # 安全检查
        if not self.points_history[person_id]:
            return None
        
        smoothed_points = []
        try:
            for point_history in self.points_history[person_id]:
                if len(point_history) == 0:
                    smoothed_points.append(None)
                    continue
                
                # 计算移动平均
                x_coords = [p[0] for p in point_history if p is not None]
                y_coords = [p[1] for p in point_history if p is not None]
                
                if len(x_coords) > 0 and len(y_coords) > 0:
                    avg_x = sum(x_coords) / len(x_coords)
                    avg_y = sum(y_coords) / len(y_coords)
                    smoothed_points.append((avg_x, avg_y))
                else:
                    smoothed_points.append(None)
        except Exception as e:
            print(f"平滑器计算平滑点时出错: {e}")
            return None
        
        return smoothed_points

class BehaviorStateMachine:
    def __init__(self):
        self.current_state = "unknown"
        self.state_duration = 0
        self.min_state_duration = 45  # 显著增加最小状态持续时间(30->45帧)
        self.transition_threshold = 0.75  # 提高状态转换阈值(0.7->0.75)
        self.hysteresis = 0.25  # 增大滞后效应(0.2->0.25)
        self.last_confidence = 0.0  # 记录上一次的置信度
        
        # 添加历史窗口滤波
        self.state_history = []  # 记录近期状态
        self.history_window = 20  # 增加历史窗口大小(15->20)
        self.confidence_history = []  # 记录置信度历史
        
        # 行为稳定性参数
        self.behavior_stability = {
            "moving": 0.2,       # 移动行为稳定性低(容易跳变)
            "explaining": 0.8,   # 讲解行为稳定性高
            "writing": 0.7,      # 板书行为稳定性较高
            "interacting": 0.6,  # 互动行为中等稳定性
            "standing": 0.9,     # 站立行为高稳定性
            "not_in_frame": 0.3  # 不在画面中低稳定性
        }
        
        # 行为转换惯性 - 从一种行为转到另一种的难度系数
        self.transition_inertia = {
            # 从站立转到其他行为
            ("standing", "explaining"): 0.1,  # 容易从站立转到讲解
            ("standing", "moving"): 0.3,      # 从站立到移动需要明显动作
            ("standing", "writing"): 0.5,     # 从站立到板书有较大变化
            ("standing", "interacting"): 0.2, # 从站立到互动相对容易
            
            # 从讲解转到其他行为
            ("explaining", "standing"): 0.1,  # 容易从讲解回到站立
            ("explaining", "moving"): 0.3,    # 从讲解到移动需要明显动作
            ("explaining", "writing"): 0.4,   # 从讲解到板书有较大变化
            ("explaining", "interacting"): 0.2, # 从讲解到互动相对容易
            
            # 从移动转到其他行为
            ("moving", "standing"): 0.2,      # 从移动到站立比较自然
            ("moving", "explaining"): 0.3,    # 从移动到讲解需要停下来
            ("moving", "writing"): 0.6,       # 从移动直接到板书需要大变化
            ("moving", "interacting"): 0.4,   # 从移动到互动需要先停下
            
            # 其他情况使用默认惯性
            ("default", "default"): 0.3
        }
    
    def update(self, new_behavior, confidence=1.0):
        """
        更新行为状态，使用增强的平滑和稳定算法
        :param new_behavior: 新的行为
        :param confidence: 新行为的置信度
        :return: 当前状态
        """
        # 更新历史记录
        self.state_history.append(new_behavior)
        self.confidence_history.append(confidence)
        
        # 保持历史窗口大小
        if len(self.state_history) > self.history_window:
            self.state_history.pop(0)
            self.confidence_history.pop(0)
        
        # 历史窗口投票机制 - 如果历史中大多数与新行为不同，可能是误判
        if len(self.state_history) >= 5:  # 至少需要几帧历史
            recent_history = self.state_history[-5:]  # 取最近5帧
            behavior_counts = {}
            
            # 计算各行为在最近历史中的出现次数
            for b in recent_history:
                behavior_counts[b] = behavior_counts.get(b, 0) + 1
            
            # 如果新行为和大多数历史行为不同，降低其置信度
            most_common = max(behavior_counts.items(), key=lambda x: x[1])
            if most_common[0] != new_behavior and most_common[1] >= 3:  # 超过一半的帧是同一行为
                # 根据历史行为的比例降低新行为的置信度
                confidence *= 0.7
                
                # 如果当前状态与历史主流一致，增加其稳定性
                if self.current_state == most_common[0]:
                    confidence *= 0.8
        
        # 应用行为稳定性 - 不同行为有不同的稳定性要求
        current_stability = self.behavior_stability.get(self.current_state, 0.5)
        new_stability = self.behavior_stability.get(new_behavior, 0.5)
        
        # 确定转换惯性 - 某些行为转换比其他更困难
        transition_key = (self.current_state, new_behavior)
        transition_cost = self.transition_inertia.get(
            transition_key, 
            self.transition_inertia.get(("default", "default"), 0.3)
        )
        
        # 对moving行为的特殊处理
        if new_behavior == "moving":
            # 提高对moving行为的检测阈值，减少误判
            if confidence < 0.85:  # 提高移动行为的门槛(0.8->0.85)
                # 如果当前不是moving，则不轻易转换为moving
                if self.current_state != "moving":
                    new_behavior = self.current_state if self.current_state != "unknown" else "explaining"
                    confidence = 0.6  # 提高保持当前状态的置信度(0.5->0.6)
        
        # 应用滞后效应
        if new_behavior != self.current_state:
            # 基础转换阈值由当前状态的稳定性和新状态的稳定性共同决定
            base_threshold = self.transition_threshold + transition_cost
            
            # 如果当前状态持续时间不够，需要更高的置信度才能转换
            if self.state_duration < self.min_state_duration:
                required_confidence = base_threshold + self.hysteresis
                # 随着持续时间增加，逐渐降低转换难度
                required_confidence -= min(self.hysteresis, (self.state_duration / self.min_state_duration) * 0.1)
            else:
                required_confidence = base_threshold
            
            # 特殊情况处理
            if self.current_state == "moving" and new_behavior != "moving":
                # 从移动转为其他状态需要稍低阈值，因为我们想快速结束移动状态
                required_confidence -= 0.15  # 降低移动->其他的阈值(0.1->0.15)
            elif self.current_state != "moving" and new_behavior == "moving":
                # 转为移动状态需要更高阈值，防止误判
                required_confidence += 0.25  # 提高其他->移动的阈值(0.2->0.25)
            
            # 应用平滑过滤 - 如果新行为的置信度不够高，保持当前状态
            if confidence < required_confidence:
                self.state_duration += 1
                return self.current_state
            
            # 如果新行为的置信度比上一次低很多，可能是误判
            confidence_drop = self.last_confidence - confidence
            if confidence_drop > 0.25:  # 增加置信度下降的容忍度(0.2->0.25)
                self.state_duration += 1
                return self.current_state
        
        # 更新状态
        if new_behavior == self.current_state:
            self.state_duration += 1
        else:
            # 记录状态变化
            old_state = self.current_state
            self.current_state = new_behavior
            self.state_duration = 1  # 重置为1而不是0，因为当前帧已经是新状态
            print(f"Behavior changed: {old_state} -> {new_behavior} (conf: {confidence:.2f})")
        
        self.last_confidence = confidence
        return self.current_state
    
    def get_state(self):
        return self.current_state
    
    def reset(self):
        self.current_state = "unknown"
        self.state_duration = 0
        self.last_confidence = 0.0
        self.state_history = []
        self.confidence_history = []

class TeacherEvaluator:
    def __init__(self, model_path=None, model_name=None, resolution=None):
        """初始化教师评估器"""
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"使用设备: {self.device}")
        
        # 设置默认模型路径
        if model_path is None:
            model_path = "models/pose_hrnet_w32_256x192.pth"
        
        # 设置默认模型名称
        if model_name is None:
            model_name = "HRNet"
            
        # 设置分辨率
        if resolution is None:
            resolution = [256, 192]  # 宽x高
            
        self.model_path = model_path
        self.model_name = model_name
        self.resolution = resolution
        self.model_type = "YOLO"  # 默认使用YOLO模型
        
        # 为关键点平滑和行为分析初始化变量
        self.keypoint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", 
                             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                             "left_wrist", "right_wrist", "left_hip", "right_hip", 
                             "left_knee", "right_knee", "left_ankle", "right_ankle"]
        
        # 初始化历史记录
        self.behavior_history = []
        self.last_position = None
        
        # 初始化缓存
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.enable_cache = True  # 默认启用缓存
        
        # 尝试加载YOLOv12模型
        try:
            parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            yolo_dir = os.path.join(parent_dir, 'yolov12-main')
            
            # 直接使用指定的模型路径
            specific_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "yolo11x-pose.pt")
            if os.path.exists(specific_model_path):
                print(f"已添加YOLOv12路径: {yolo_dir}")
                sys.path.append(yolo_dir)
                import ultralytics
                from ultralytics import YOLO
                
                self.yolo_model = YOLO(specific_model_path)
                self.model_type = "YOLOv11X"
                print(f"已加载指定的YOLO模型: {specific_model_path}")
            else:
                print(f"指定的模型文件不存在: {specific_model_path}")
                
        except ImportError as e:
            print(f"无法导入YOLO: {e}")
        except Exception as e:
            print(f"加载YOLO模型时出错: {e}")
            
        # 中文字体设置
        self.font_path = "./simhei.ttf"
        if not os.path.exists(self.font_path):
            print(f"中文字体文件 {self.font_path} 不存在，将使用默认字体")
            self.font_path = None
            
        print("\n增强型教师评估系统初始化完成\n")
        # 输出模型配置
        print("模型配置:")
        print(f"- 设备: {self.device}")
        print(f"- 模型类型: {self.model_type}")
        print(f"- YOLO置信度阈值: 0.1")
        print(f"- YOLO IOU阈值: 0.45")
        print(f"- 关键点置信度阈值: 0.05")
        print(f"- 最小有效关键点数量: 3")
        print()
        
        # 设置检测参数
        self.keypoint_conf_threshold = 0.25
        self.min_valid_keypoints = 3
        
        # 初始化行为状态机
        self.behavior_state_machine = {
            "standing": 0.0,  # 站立状态
            "sitting": 0.0,   # 坐姿状态
            "explaining": 0.0,  # 讲解状态
            "writing": 0.0,   # 板书状态
            "moving": 0.0,    # 移动状态
            "interacting": 0.0,  # 互动状态
            "not_in_frame": 0.0,  # 不在画面中
        }
        
        # 初始化姿态平滑器
        self.pose_smoother = PoseSmoother()
        
        # 初始化行为计数字典
        self.behavior_counts = {
            "standing": 0,
            "sitting": 0,      # 添加坐姿行为计数
            "explaining": 0,
            "writing": 0,
            "moving": 0,
            "interacting": 0,
            "not_in_frame": 0,  # 增加"不在画面中"的计数
        }
        
        # 记录历史
        self.pose_history = []
        self.position_history = []
        self.behavior_history = []
        self.behavior_change_count = 0
        self.last_position = None
        
        # 设置置信度阈值 - 降低以提高检测率
        self.yolo_model.conf = 0.1  # 从0.2降低到0.1
        self.yolo_model.iou = 0.45   # IOU阈值
        
        # 添加YOLO置信度阈值
        self.yolo_conf_threshold = 0.1  # 从0.2降低到0.1
        
        # 关键点置信度阈值，大幅降低以检测更多关键点
        self.keypoint_conf_threshold = 0.05  # 从0.15降低到0.05
        
        # 最小有效关键点数量，降低以增加检测成功率
        self.min_valid_keypoints = 3  # 从4降低到3
        
        # 定义关键点索引及其名称 (COCO格式，与YOLOv8-pose一致)
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 关键点分类（按重要性）
        self.key_joints = {
            "primary": [5, 6, 9, 10],  # 肩膀和手腕（用于手势识别）
            "secondary": [0, 11, 12],  # 头部和臀部（用于姿势识别）
            "tertiary": [7, 8, 13, 14, 15, 16]  # 肘部和腿部（辅助判断）
        }
        
        # 定义连接关系用于绘制骨架
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], 
            [12, 13], [6, 12], [7, 13], [6, 7], 
            [6, 8], [7, 9], [8, 10], [9, 11], 
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5]
        ]
        
        # 行为检测相关
        self.last_position = None
        self.behavior_history = []
        self.behavior_change_count = 0
        self.last_behavior = "unknown"
        
        # 行为状态机
        self.behavior_state_machine = BehaviorStateMachine()
        
        # 行为计数器
        self.behavior_counts = {
            "explaining": 0,
            "writing": 0, 
            "moving": 0,
            "unknown": 0
        }
        
        # 存储历史姿态数据
        self.pose_history = []
        self.position_history = deque(maxlen=300)  # 存储最近300帧的位置
        self.gesture_history = deque(maxlen=300)   # 存储最近300帧的手势
        
        # 姿态平滑器
        self.pose_smoother = PoseSmoother(window_size=5)
        
        # 评估指标
        self.metrics = {
            "teaching_activity": 0.0,      # 教学活跃度
            "teaching_engagement": 0.0,    # 教学参与度
            "teaching_rhythm": 0.0,        # 教学节奏
            "teaching_diversity": 0.0,     # 教学多样性
            "overall_score": 0.0           # 综合得分
        }

    def clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def analyze_posture(self, keypoints):
        """分析教师姿态"""
        # 确保关键点是numpy数组
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        
        keypoints_data = {}
        
        # 头部姿势分析
        if 0 in keypoints and 1 in keypoints and 2 in keypoints:
            # 鼻子和眼睛位置
            nose = keypoints[0]
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            keypoints_data["head_pose"] = {
                "nose": nose,
                "left_eye": left_eye,
                "right_eye": right_eye
            }
        
        # 肩膀分析
        if 5 in keypoints and 6 in keypoints:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            shoulder_angle = np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ) * 180 / np.pi
            keypoints_data["shoulder_angle"] = shoulder_angle
        
        # 上肢分析
        if 5 in keypoints and 7 in keypoints and 9 in keypoints:
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            
            # 计算左臂角度
            vec1 = left_elbow - left_shoulder
            vec2 = left_wrist - left_elbow
            
            cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            left_arm_angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
            keypoints_data["left_arm_angle"] = left_arm_angle
        
        # 右臂分析
        if 6 in keypoints and 8 in keypoints and 10 in keypoints:
            right_shoulder = keypoints[6]
            right_elbow = keypoints[8]
            right_wrist = keypoints[10]
            
            # 计算右臂角度
            vec1 = right_elbow - right_shoulder
            vec2 = right_wrist - right_elbow
            
            cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            right_arm_angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
            keypoints_data["right_arm_angle"] = right_arm_angle
        
        # 身体姿势分析
        if 5 in keypoints and 6 in keypoints and 11 in keypoints and 12 in keypoints:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            # 计算上半身倾斜度
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            
            body_angle = np.arctan2(
                shoulder_center[0] - hip_center[0],
                shoulder_center[1] - hip_center[1]
            ) * 180 / np.pi
            keypoints_data["body_angle"] = body_angle
        
        # 下肢分析
        if 11 in keypoints and 13 in keypoints and 15 in keypoints:
            left_hip = keypoints[11]
            left_knee = keypoints[13]
            left_ankle = keypoints[15]
            
            # 计算左腿角度
            vec1 = left_knee - left_hip
            vec2 = left_ankle - left_knee
            
            cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            left_leg_angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
            keypoints_data["left_leg_angle"] = left_leg_angle
        
        # 站姿分析
        if 11 in keypoints and 12 in keypoints and 15 in keypoints and 16 in keypoints:
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            
            hip_center = (left_hip + right_hip) / 2
            ankle_center = (left_ankle + right_ankle) / 2
            
            # 计算站姿垂直度
            vertical_angle = np.arctan2(
                hip_center[0] - ankle_center[0],
                hip_center[1] - ankle_center[1]
            ) * 180 / np.pi
            keypoints_data["vertical_angle"] = vertical_angle
        
        # 计算中心位置
        if 5 in keypoints and 6 in keypoints and 11 in keypoints and 12 in keypoints:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            # 计算身体中心点
            center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            keypoints_data["center"] = center
        
        # 总体姿态评分计算
        posture_score = 0.0
        score_count = 0
        
        # 肩膀水平评分 (0分表示完全不水平，10分表示完全水平)
        if "shoulder_angle" in keypoints_data:
            shoulder_horizontality = 10 - min(10, abs(keypoints_data["shoulder_angle"]) / 9)
            posture_score += shoulder_horizontality
            score_count += 1
        
        # 身体垂直度评分
        if "body_angle" in keypoints_data:
            body_verticality = 10 - min(10, abs(keypoints_data["body_angle"]) / 9)
            posture_score += body_verticality
            score_count += 1
        
        # 站姿垂直度评分
        if "vertical_angle" in keypoints_data:
            stance_verticality = 10 - min(10, abs(keypoints_data["vertical_angle"]) / 9)
            posture_score += stance_verticality
            score_count += 1
        
        # 计算平均分
        if score_count > 0:
            posture_score /= score_count
        
        keypoints_data["posture_score"] = posture_score
        
        # 将当前帧姿态添加到历史记录
        self.pose_history.append(keypoints_data)
        
        # 如果有中心点信息，追踪位置
        if "center" in keypoints_data:
            self.position_history.append(keypoints_data["center"])
        
        # 识别教学行为
        behavior = self.identify_teaching_behavior(keypoints_data)
        
        return keypoints_data, posture_score, behavior
    
    def identify_teaching_behavior(self, keypoints_data):
        """基于关键点数据识别教师行为"""
        # 初始化行为分数
        behavior_scores = {
            "writing": 0.1,      # 写黑板 - 低基础分，因为老师没有写黑板字
            "explaining": 0.6,    # 讲解 - 较高基础分，因为主要是站立讲解
            "moving": 0.0,        # 移动 - 无基础分
            "interacting": 0.1,   # 互动 - 较低基础分
            "standing": 0.5,      # 站立 - 较高基础分，因为老师确实主要在站立
            "not_in_frame": 0.0   # 不在镜头中
        }
        
        # 1. 首先判断老师是否在画面中
        if not keypoints_data or len(keypoints_data) < 5:
            return "not_in_frame"
        
        # 2. 基于可见度判断
        face_visible = keypoints_data.get("face_visibility", 0)
        body_visible = keypoints_data.get("body_visibility", 0)
        
        if face_visible < 0.2 and body_visible < 0.3:
            return "not_in_frame"
            
        # 3. 移动行为判断
        if "movement" in keypoints_data:
            movement = keypoints_data["movement"]
            continuous_movement = keypoints_data.get("continuous_movement", False)
            
            # 大幅度移动才被视为移动行为
            if movement > 60:
                # 只有连续移动才给高分，但也降低最高分数
                if continuous_movement:
                    movement_score = min(0.8, movement / 200)
                    behavior_scores["moving"] = movement_score
                else:
                    # 不连续移动给极低分数
                    behavior_scores["moving"] = 0.1
            
        # 4. 讲解行为判断 - 优化检测手势
        if face_visible > 0.3 and body_visible > 0.3:
            # 检查手势活动
            has_hand_gesture = False
            left_arm_raised = keypoints_data.get("left_arm_raised", False)
            right_arm_raised = keypoints_data.get("right_arm_raised", False)
            left_arm_angle = keypoints_data.get("left_arm_angle", 0)
            right_arm_angle = keypoints_data.get("right_arm_angle", 0)
            
            # 手臂举起或呈讲解姿势
            if left_arm_raised or right_arm_raised:
                has_hand_gesture = True
                behavior_scores["explaining"] += 0.2
                
            # 讲解手势 - 手臂在身体前方且角度适中
            if (20 < left_arm_angle < 100) or (20 < right_arm_angle < 100):
                has_hand_gesture = True
                behavior_scores["explaining"] += 0.2
            
            # 检查姿态是否符合讲解
            if "vertical_angle" in keypoints_data:
                vertical_angle = abs(keypoints_data["vertical_angle"])
                if vertical_angle < 20:  # 身体基本垂直，符合讲解状态
                    behavior_scores["explaining"] += 0.1
            
        # 5. 站立行为判断 - 针对站立讲解场景优化
        if face_visible > 0.3 and body_visible > 0.3:
            # 如果身体姿态垂直，增加站立得分
            if "vertical_angle" in keypoints_data:
                vertical_angle = abs(keypoints_data["vertical_angle"])
                if vertical_angle < 15:  # 身体非常垂直
                    behavior_scores["standing"] += 0.2
            
            # 检查最近几帧的移动情况，如果移动很少，更可能是站立
            movement = keypoints_data.get("movement", 0)
            if movement < 20:  # 非常少的移动
                behavior_scores["standing"] += 0.2
            elif movement < 40:  # 轻微移动
                behavior_scores["standing"] += 0.1
        
        # 6. 综合考虑手势和站立
        # 如果有手势动作，减少站立得分，增加讲解得分
        if "has_hand_gesture" in keypoints_data and keypoints_data["has_hand_gesture"]:
            behavior_scores["explaining"] += 0.1
            behavior_scores["standing"] -= 0.1
        
        # 7. 互动行为判断
        if "face_orientation" in keypoints_data:
            face_orientation = keypoints_data["face_orientation"]
            if abs(face_orientation) > 30:  # 面部明显朝向一侧
                behavior_scores["interacting"] += 0.3
        
        # 选择得分最高的行为
        best_behavior = max(behavior_scores.items(), key=lambda x: x[1])
        behavior = best_behavior[0]
        confidence = best_behavior[1]
        
        # 如果所有行为得分都很低且不是not_in_frame，依然保持为explaining
        if confidence < 0.3 and behavior != "not_in_frame":
            behavior = "explaining"
            confidence = 0.3
        
        # 更新行为计数
        self.behavior_counts[behavior] = self.behavior_counts.get(behavior, 0) + 1
        
        # 使用行为状态机更新状态
        current_behavior = self.behavior_state_machine.update(behavior, confidence)
        
        return current_behavior
    
    def calculate_movement_metrics(self):
        """计算移动相关指标"""
        if len(self.position_history) < 2:
            return 0, 0
        
        # 计算移动距离
        distances = []
        positions = list(self.position_history)
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            distances.append(dist)
        
        # 计算移动范围 (x方向和y方向的最大范围)
        pos_array = np.array(positions)
        x_range = np.max(pos_array[:, 0]) - np.min(pos_array[:, 0])
        y_range = np.max(pos_array[:, 1]) - np.min(pos_array[:, 1])
        
        movement_range = np.sqrt(x_range**2 + y_range**2)
        avg_movement = np.mean(distances) if distances else 0
        
        return movement_range, avg_movement
    
    def process_frame(self, frame):
        """处理单帧并检测教师行为"""
        # 清理GPU内存
        self.clear_gpu_memory()
        
        try:
            # 图像增强
            frame = self.enhance_image(frame)
            
            # 调整图像大小以提高检测效果
            original_height, original_width = frame.shape[:2]
            target_size = 1024  # 增大到1024以提高精度
            scale = min(target_size / original_width, target_size / original_height)
            width = int(original_width * scale)
            height = int(original_height * scale)
            resized_frame = cv2.resize(frame, (width, height))
            
            # YOLO检测参数优化
            try:
                # YOLOv8的检测方式
                results = self.yolo_model(
                    resized_frame, 
                    verbose=False, 
                    conf=self.yolo_conf_threshold
                )
            except Exception as e:
                print(f"YOLO检测错误: {e}")
                import traceback
                traceback.print_exc()
                return None, 0.0, None
                
            # 提取人物关键点
            if results and len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                # 获取所有人物的关键点
                all_keypoints = results[0].keypoints
                if len(all_keypoints) == 0:
                    print("未检测到任何人物关键点")
                    return None, 0.0, None
                
                # 选择最可能的教师
                # 先获取所有边界框
                persons = []
                for i, box in enumerate(results[0].boxes):
                    # 确保是人类目标
                    try:
                        if int(box.cls[0]) == 0:  # 类别0是人
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # 过滤太小的检测
                            if (x2 - x1) < 15 or (y2 - y1) < 40:  # 降低最小尺寸要求
                                continue
                                
                            area = (x2 - x1) * (y2 - y1)
                            confidence = float(box.conf[0])
                            box_center_x = (x1 + x2) / 2
                            
                            # 计算位置分数 - 偏好画面中心区域和较大的人物
                            center_score = 1.0 - abs(box_center_x - width/2) / (width/2)
                            size_score = min(1.0, area / 40000)  # 根据面积赋予额外分数
                            # 综合评分 = 面积 * 置信度 * 中心偏好 * 大小分数
                            combined_score = area * confidence * center_score * (1 + size_score)
                            
                            persons.append({
                                "index": i, 
                                "area": area, 
                                "confidence": confidence,
                                "combined_score": combined_score,
                                "box": (x1, y1, x2, y2)
                            })
                    except Exception as e:
                        print(f"处理边界框时出错: {e}")
                        continue
                
                if not persons:
                    print("未检测到合适的人物")
                    return None, 0.0, None
                
                # 使用教师识别算法选择最可能的教师
                # 存储上一帧的教师信息用于连续性跟踪
                previous_teacher = getattr(self, 'previous_teacher', None)
                teacher = self.identify_teacher(persons, previous_teacher)
                # 更新上一帧的教师信息
                self.previous_teacher = teacher
                
                person_idx = teacher["index"]
                
                try:
                    # 获取关键点和置信度 - YOLOv8格式
                    kpts = all_keypoints[person_idx].data[0].cpu().numpy()  # [17, 3] shape
                
                    # 创建关键点字典
                    keypoints_dict = {}
                    confidence_sum = 0
                    valid_keypoints = 0
                
                    for i in range(len(kpts)):
                        # 每个关键点包含x, y, confidence
                        x, y, conf = kpts[i]
                        if conf > self.keypoint_conf_threshold and i < len(self.keypoint_names):
                            keypoints_dict[self.keypoint_names[i]] = (float(x), float(y))
                            confidence_sum += conf
                            valid_keypoints += 1
                
                    # 计算平均置信度
                    avg_confidence = confidence_sum / valid_keypoints if valid_keypoints > 0 else 0
                
                    # 检查是否有足够的关键点
                    if valid_keypoints < self.min_valid_keypoints:
                        print(f"有效关键点数量不足: {valid_keypoints}/{len(self.keypoint_names)}")
                        # 尝试从历史中补充关键点
                        if len(self.pose_history) > 0:
                            print("尝试从历史中恢复关键点...")
                            last_valid_keypoints = None
                            for hist in reversed(self.pose_history[-15:]):  # 从15帧历史中寻找
                                if len(hist) >= self.min_valid_keypoints:
                                    last_valid_keypoints = hist
                                    break
                            
                            if last_valid_keypoints:
                                print(f"从历史中恢复了 {len(last_valid_keypoints)} 个关键点")
                                keypoints_dict = last_valid_keypoints.copy()
                                avg_confidence = 0.5  # 设置一个默认置信度
                                valid_keypoints = len(keypoints_dict)
                            else:
                                return None, 0.0, None
                        else:
                            return None, 0.0, None
                
                    # 特别关注重要关键点的存在
                    important_points = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
                    missing_important = [p for p in important_points if p not in keypoints_dict]
                    if missing_important:
                        print(f"缺少重要关键点: {missing_important}")
                        # 尝试使用平滑历史数据填充缺失的关键点
                        if len(self.pose_history) > 0:
                            for point in missing_important:
                                for hist in reversed(self.pose_history[-10:]):  # 从10帧历史中寻找，而不是5帧
                                    if point in hist:
                                        keypoints_dict[point] = hist[point]
                                        print(f"从历史数据中填充关键点: {point}")
                                        break
                    
                    # 调试输出关键点信息
                    print(f"检测到 {valid_keypoints} 个有效关键点，平均置信度: {avg_confidence:.3f}")
                    
                    # 应用姿态平滑，使用更大的平滑窗口
                    self.pose_smoother.window_size = 7  # 增加平滑窗口大小
                    try:
                        # 确保keypoints_dict非空并包含有效值
                        if keypoints_dict and any(keypoints_dict.values()):
                            # 安全地获取平滑后的关键点
                            smoothed_keypoints = self.pose_smoother.update(0, list(keypoints_dict.values()))
                            
                            if smoothed_keypoints:
                                # 更新关键点字典
                                for i, point in enumerate(smoothed_keypoints):
                                    if point is not None and i < len(self.keypoint_names):
                                        keypoints_dict[self.keypoint_names[i]] = point
                        else:
                            print("跳过平滑：关键点字典为空或仅包含无效值")
                    except Exception as e:
                        print(f"姿态平滑处理时出错: {e}")
                        # 出错时继续使用原始关键点
                        import traceback
                        traceback.print_exc()
                    
                    # 插值计算缺失的关键点
                    keypoints_dict = self.interpolate_missing_keypoints(keypoints_dict)
                
                    # 分析行为
                    behavior = self.analyze_behavior(keypoints_dict)
                
                    # 更新行为历史
                    self.behavior_history.append(behavior)
                    if len(self.behavior_history) > 300:  # 保持最近300帧的历史
                        self.behavior_history.pop(0)
                    
                    return keypoints_dict, avg_confidence, behavior
                    
                except Exception as e:
                    print(f"处理关键点时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, 0.0, None
            else:
                print("YOLO未检测到任何目标或关键点")
                return None, 0.0, None
                
        except Exception as e:
            print(f"处理帧时出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, 0.0, None

    def evaluate_video(self, video_path, batch_size=10, skip_frames=1, start_frame=0, max_frames=None, use_cache=True):
        """
        评估视频中的教师行为
        
        Args:
            video_path (str): 视频文件路径
            batch_size (int): 批处理大小
            skip_frames (int): 每隔多少帧处理一次
            start_frame (int): 从第几帧开始处理
            max_frames (int): 最多处理多少帧
            use_cache (bool): 是否使用缓存加速
            
        Returns:
            dict: 分析结果
        """
        # 重置状态
        self.behavior_history = []
        self.last_position = None
        
        results = {
            "status": "initialization",
            "message": "正在初始化"
        }
        
        # 检查是否使用缓存
        if use_cache and self.enable_cache:
            # 生成缓存文件名
            cache_key = f"{os.path.basename(video_path)}_{start_frame}_{max_frames}_{skip_frames}_{batch_size}"
            cache_file = os.path.join(self.cache_dir, f"{hash(cache_key)}.pkl")
            
            # 尝试从缓存加载
            if os.path.exists(cache_file):
                try:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        cached_results = pickle.load(f)
                    
                    print(f"从缓存加载分析结果: {cache_file}")
                    
                    # 恢复行为历史
                    if "frames_data" in cached_results:
                        self.behavior_history = [data.get("behavior") if data else None for data in cached_results["frames_data"]]
                    
                    # 记录缓存命中
                    cached_results["cache_info"] = {
                        "cache_hit": True,
                        "cache_file": cache_file,
                        "original_time": cached_results.get("performance", {}).get("total_time", 0),
                        "load_time": 0.01  # 近似值
                    }
                    
                    return cached_results
                except Exception as e:
                    print(f"加载缓存时出错: {e}, 将重新分析视频")
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "status": "error",
                    "message": f"无法打开视频: {video_path}"
                }
                
            # 获取视频基本信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 如果指定了起始帧，跳到指定位置
            if start_frame > 0:
                if start_frame >= frame_count:
                    return {
                        "status": "error",
                        "message": f"起始帧 {start_frame} 超出视频总帧数 {frame_count}"
                    }
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 计算实际要处理的帧数
            if max_frames is not None:
                actual_frame_count = min(frame_count - start_frame, max_frames)
            else:
                actual_frame_count = frame_count - start_frame
            
            # 计算实际需要处理的帧数（考虑跳帧）
            frames_to_process = actual_frame_count // skip_frames
            if actual_frame_count % skip_frames > 0:
                frames_to_process += 1
                
            print(f"处理视频从第 {start_frame} 帧开始，总帧数: {actual_frame_count}，跳帧: {skip_frames}，实际处理: {frames_to_process} 帧")
            
            # 开始全局计时
            total_start_time = time.time()
            
            # 初始化进度条
            from tqdm import tqdm
            progress_bar = tqdm(total=frames_to_process)
            
            # 初始化结果
            frames_data = []
            behavior_counts = {}
            processed_frames = 0
            valid_frames = 0
            total_confidence = 0.0
            total_posture_score = 0.0
            batch_times = []
            gpu_usages = []
            
            # 初始化关键点平滑器和行为状态机
            pose_smoother = PoseSmoother(window_size=3)
            behavior_state_machine = BehaviorStateMachine()
            
            # 读取视频帧 - 优化版跳帧处理
            batch_frames = []
            frame_idx = start_frame
            processed_count = 0

            # 使用快速读取方式
            while cap.isOpened() and (max_frames is None or processed_count < max_frames):
                # 快速跳过不需要处理的帧
                if frame_idx > start_frame and skip_frames > 1:
                    # 计算需要跳过的帧数
                    for _ in range(skip_frames - 1):
                        if not cap.grab():  # grab()只获取帧不解码，比read()更快
                            break
                
                # 读取需要处理的帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 批量处理
                batch_frames.append((frame_idx, frame))
                processed_count += 1
                
                # 如果帧数已经足够一个批次，处理这个批次
                if len(batch_frames) >= batch_size:
                    # 处理批次前记录GPU状态
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 确保之前的GPU操作完成
                        gpu_usage = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                        gpu_usages.append(gpu_usage)
                    
                    # 开始批处理时间计时
                    batch_start_time = time.time()
                    
                    # 处理这个批次
                    batch_results = self._process_batch(batch_frames)
                    frames_data.extend(batch_results)
                    
                    # 更新行为计数和统计信息
                    for result in batch_results:
                        if result and result.get("status") == "success":
                            behavior = result.get("behavior")
                            if behavior:
                                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                                behavior_state_machine.update(behavior)
                                valid_frames += 1
                            
                            # 更新总置信度和姿态分数
                            confidence = result.get("confidence", 0.0)
                            posture_score = result.get("posture_score", 0.0)
                            total_confidence += confidence
                            total_posture_score += posture_score
                    
                    # 计算批处理时间
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)
                    
                    # 更新进度条
                    progress_bar.update(len(batch_frames))
                    processed_frames += len(batch_frames)
                    
                    # 清空批次
                    batch_frames = []
                
                # 更新帧索引
                frame_idx += skip_frames
            
            # 处理最后一个批次
            if batch_frames:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_usage = torch.cuda.memory_allocated() / (1024 ** 3)
                    gpu_usages.append(gpu_usage)
                
                batch_start_time = time.time()
                batch_results = self._process_batch(batch_frames)
                frames_data.extend(batch_results)
                
                for result in batch_results:
                    if result and result.get("status") == "success":
                        behavior = result.get("behavior")
                        if behavior:
                            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                            behavior_state_machine.update(behavior)
                            valid_frames += 1
                        
                        confidence = result.get("confidence", 0.0)
                        posture_score = result.get("posture_score", 0.0)
                        total_confidence += confidence
                        total_posture_score += posture_score
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                
                progress_bar.update(len(batch_frames))
                processed_frames += len(batch_frames)
            
            # 关闭进度条和视频
            progress_bar.close()
            cap.release()
            
            # 计算总时间
            total_end_time = time.time()
            total_elapsed_time = total_end_time - total_start_time
            
            # 计算性能指标
            total_time = sum(batch_times)
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            processing_fps = processed_frames / total_time if total_time > 0 else 0
            avg_confidence = total_confidence / valid_frames if valid_frames > 0 else 0
            avg_posture_score = total_posture_score / valid_frames if valid_frames > 0 else 0
            
            # 计算跳帧前后的估计节省时间
            if skip_frames > 1:
                estimated_full_time = total_time * skip_frames
                time_saved = estimated_full_time - total_time
                print(f"\n使用跳帧分析（跳{skip_frames}帧）节省了约 {time_saved:.2f} 秒 (约 {time_saved/60:.2f} 分钟)")
                print(f"全帧分析估计需要 {estimated_full_time:.2f} 秒，实际用时 {total_time:.2f} 秒")
                print(f"处理速度: {processing_fps:.2f} 帧/秒")
            
            # GPU使用情况
            avg_gpu_usage = 0
            max_gpu_usage = 0
            if gpu_usages:
                avg_gpu_usage = sum(gpu_usages) / len(gpu_usages)
                max_gpu_usage = max(gpu_usages)
                print(f"GPU内存: 平均 {avg_gpu_usage:.2f} GB, 最大 {max_gpu_usage:.2f} GB")
            
            # 返回最终结果
            results = {
                "status": "success",
                "behavior_counts": behavior_counts,
                "total_frames": frame_count,
                "start_frame": start_frame,
                "processed_frames": processed_frames,
                "valid_frames": valid_frames,
                "fps": fps,
                "avg_confidence": avg_confidence,
                "avg_posture_score": avg_posture_score,
                "frames_data": frames_data,
                "performance": {
                    "total_time": total_time,
                    "total_elapsed_time": total_elapsed_time,
                    "processing_fps": processing_fps,
                    "avg_batch_time": avg_batch_time,
                    "avg_gpu_usage": avg_gpu_usage,
                    "max_gpu_usage": max_gpu_usage,
                    "skip_frames": skip_frames,
                    "time_saved_seconds": time_saved if skip_frames > 1 else 0
                },
                "cache_info": {
                    "cache_hit": False,
                    "cache_saved": True if use_cache and self.enable_cache else False
                }
            }
            
            # 设置最终状态
            self.behavior_history = [data.get("behavior") if data else None for data in frames_data]
            
            # 如果启用了缓存，保存结果
            if use_cache and self.enable_cache:
                try:
                    import pickle
                    with open(cache_file, 'wb') as f:
                        pickle.dump(results, f)
                    print(f"分析结果已缓存: {cache_file}")
                    results["cache_info"]["cache_file"] = cache_file
                except Exception as e:
                    print(f"缓存结果时出错: {e}")
            
            return results
            
        except Exception as e:
            print(f"视频评估出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": f"处理异常: {str(e)}"
            }

    def _process_batch(self, batch_frames):
        """
        批量处理帧

        Args:
            batch_frames: 帧元组列表，每个元组包含 (frame_idx, frame)

        Returns:
            list: 每一帧的处理结果
        """
        results = []
        
        try:
            # 准备批处理所需的帧
            frames = [frame for _, frame in batch_frames]
            frame_indices = [idx for idx, _ in batch_frames]
            
            # 使用GPU加速时，尝试批量预测（如果模型支持）
            if hasattr(self, 'model') and torch.cuda.is_available():
                try:
                    # 预先转换所有帧为RGB（批量操作更快）
                    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
                    
                    # 批量预测
                    with torch.no_grad():  # 使用no_grad减少内存使用
                        batch_predictions = []
                        for rgb_frame in rgb_frames:
                            # 这里可能会根据模型类型不同而变化
                            prediction = self.process_frame(rgb_frame)
                            batch_predictions.append(prediction)
                    
                    # 处理预测结果
                    for i, (prediction, frame_idx) in enumerate(zip(batch_predictions, frame_indices)):
                        try:
                            keypoints, confidence, behavior = prediction
                            
                            # 创建单帧处理结果
                            frame_result = {
                                'status': 'success',
                                'keypoints': keypoints,
                                'behavior': behavior,
                                'confidence': confidence,
                                'frame_idx': frame_idx
                            }
                            
                            # 添加姿态详情（如果有）
                            if hasattr(self, 'posture_details'):
                                frame_result['posture_details'] = self.posture_details
                                
                            results.append(frame_result)
                        except Exception as e:
                            print(f"处理批中的帧 {frame_idx} 时出错: {e}")
                            results.append({
                                'status': 'error',
                                'message': f'处理异常: {str(e)}',
                                'keypoints': None,
                                'behavior': 'unknown',
                                'frame_idx': frame_idx
                            })
                except Exception as e:
                    print(f"批处理预测时出错: {e}，回退到逐帧处理")
                    # 如果批处理失败，回退到逐帧处理
                    for frame_idx, frame in batch_frames:
                        try:
                            # 创建单帧处理结果
                            frame_result = self._process_frame(frame)
                            if frame_result:
                                # 添加帧索引
                                frame_result['frame_idx'] = frame_idx
                            results.append(frame_result)
                        except Exception as e:
                            print(f"处理帧 {frame_idx} 时出错: {e}")
                            # 添加错误结果
                            results.append({
                                'status': 'error',
                                'message': f'处理异常: {str(e)}',
                                'keypoints': None,
                                'behavior': 'unknown',
                                'frame_idx': frame_idx
                            })
            else:
                # 常规逐帧处理
                for frame_idx, frame in batch_frames:
                    try:
                        # 创建单帧处理结果
                        frame_result = self._process_frame(frame)
                        if frame_result:
                            # 添加帧索引
                            frame_result['frame_idx'] = frame_idx
                        results.append(frame_result)
                    except Exception as e:
                        print(f"处理帧 {frame_idx} 时出错: {e}")
                        # 添加错误结果
                        results.append({
                            'status': 'error',
                            'message': f'处理异常: {str(e)}',
                            'keypoints': None,
                            'behavior': 'unknown',
                            'frame_idx': frame_idx
                        })
        except Exception as e:
            print(f"批处理预测时出错: {e}")
            
        return results

    def generate_comprehensive_report(self, video_path, fps=None, selected_behavior_types=None):
        """生成教师教学行为的综合分析报告"""
        # 提取行为序列
        behavior_sequence, valid_frames, posture_details_sequence = self._extract_behavior_sequence(video_path, fps)
        
        if not behavior_sequence:
            return {"error": "无法提取行为序列"}
            
        # 计算视频时长
        total_duration = len(behavior_sequence) / fps if fps else 0
        
        # 计算行为时间占比
        behavior_frames = {}
        for behavior in behavior_sequence:
            if behavior:
                behavior_frames[behavior] = behavior_frames.get(behavior, 0) + 1
                
        # 包含所有可能的行为类型
        all_behaviors = ["explaining", "writing", "moving", "interacting", 
                         "standing", "sitting", "pointing", "raising_hand", "unknown", "not_in_frame"]
                         
        # 如果指定了选择的行为类型，则使用它们
        if selected_behavior_types:
            all_behaviors = selected_behavior_types
        
        behavior_time_proportions = {}
        total_valid_frames = sum(1 for frame in behavior_sequence if frame)
        
        for behavior in all_behaviors:
            count = behavior_frames.get(behavior, 0)
            proportion = count / total_valid_frames if total_valid_frames > 0 else 0
            behavior_time_proportions[behavior] = proportion
            
        # 计算行为变化频率（每分钟变化次数）
        changes = sum(1 for i in range(1, len(behavior_sequence)) 
                     if behavior_sequence[i] and behavior_sequence[i-1] and 
                     behavior_sequence[i] != behavior_sequence[i-1])
        change_frequency = changes / (total_duration / 60) if total_duration > 0 else 0
        
        # 确定主要行为（占比超过10%的行为）
        major_behaviors = {b: p for b, p in behavior_time_proportions.items() if p >= 0.1}
        
        # 分析上下身姿态统计
        posture_stats = self._analyze_posture_details(posture_details_sequence)
        
        # 调用现有方法生成教学风格指标
        metrics = self.calculate_teaching_metrics(behavior_frames, len(behavior_sequence), fps, 
                                              sum(valid_frames.values()))
        
        # 生成完整报告
        report = {
            "total_duration": total_duration,
            "behavior_time_proportions": behavior_time_proportions,
            "behavior_change_frequency": change_frequency,
            "major_behaviors": major_behaviors,
            "teaching_style_metrics": metrics,
            "posture_analysis": posture_stats,  # 添加姿态分析结果
            "behavior_sequence": behavior_sequence[:100]  # 仅包含前100个行为，避免返回过大的数据
        }
        
        return report
        
    def _analyze_posture_details(self, posture_details_sequence):
        """分析上下身姿态得分序列，生成统计结果"""
        if not posture_details_sequence or all(pd is None for pd in posture_details_sequence):
            return {"error": "无有效姿态数据"}
        
        # 初始化统计变量
        stats = {
            "upper_body_standing": {"avg": 0, "max": 0, "min": 1},
            "lower_body_standing": {"avg": 0, "max": 0, "min": 1},
            "overall_standing": {"avg": 0, "max": 0, "min": 1},
            "sitting": {"avg": 0, "max": 0, "min": 1},
            "upper_lower_consistency": 0  # 上下身一致性
        }
        
        # 有效数据计数
        valid_count = 0
        consistency_sum = 0
        
        # 累计数据
        for details in posture_details_sequence:
            if details:
                valid_count += 1
                
                # 更新上半身站立得分统计
                if "upper_body_standing" in details:
                    score = details["upper_body_standing"]
                    stats["upper_body_standing"]["avg"] += score
                    stats["upper_body_standing"]["max"] = max(stats["upper_body_standing"]["max"], score)
                    stats["upper_body_standing"]["min"] = min(stats["upper_body_standing"]["min"], score)
                
                # 更新下半身站立得分统计
                if "lower_body_standing" in details:
                    score = details["lower_body_standing"]
                    stats["lower_body_standing"]["avg"] += score
                    stats["lower_body_standing"]["max"] = max(stats["lower_body_standing"]["max"], score)
                    stats["lower_body_standing"]["min"] = min(stats["lower_body_standing"]["min"], score)
                
                # 更新整体站立得分统计
                if "overall_standing" in details:
                    score = details["overall_standing"]
                    stats["overall_standing"]["avg"] += score
                    stats["overall_standing"]["max"] = max(stats["overall_standing"]["max"], score)
                    stats["overall_standing"]["min"] = min(stats["overall_standing"]["min"], score)
                
                # 更新坐姿得分统计
                if "sitting" in details:
                    score = details["sitting"]
                    stats["sitting"]["avg"] += score
                    stats["sitting"]["max"] = max(stats["sitting"]["max"], score)
                    stats["sitting"]["min"] = min(stats["sitting"]["min"], score)
                
                # 计算上下身一致性
                if "upper_body_standing" in details and "lower_body_standing" in details:
                    upper = details["upper_body_standing"]
                    lower = details["lower_body_standing"]
                    # 相似度计算：1 - |上半身得分 - 下半身得分|
                    consistency = 1 - abs(upper - lower)
                    consistency_sum += consistency
        
        # 计算平均值
        if valid_count > 0:
            stats["upper_body_standing"]["avg"] /= valid_count
            stats["lower_body_standing"]["avg"] /= valid_count
            stats["overall_standing"]["avg"] /= valid_count
            stats["sitting"]["avg"] /= valid_count
            stats["upper_lower_consistency"] = consistency_sum / valid_count
        
        return stats

    def generate_enhanced_visualization_charts(self, report_data, video_path):
        """生成增强的可视化图表"""
        # 创建图表
        plt.figure(figsize=(12, 10))

        # 1. 教学行为占比饼图
        plt.subplot(2, 2, 1)
        behaviors = ['writing', 'explaining', 'interacting', 'moving', 'standing']
        percentages = [
            report_data["teaching_behaviors"]["writing_percentage"],
            report_data["teaching_behaviors"]["explaining_percentage"],
            report_data["teaching_behaviors"]["interacting_percentage"],
            report_data["teaching_behaviors"]["moving_percentage"],
            report_data["teaching_behaviors"]["standing_percentage"]
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        
        # 过滤掉零值
        non_zero_behaviors = []
        non_zero_percentages = []
        non_zero_colors = []
        for i, p in enumerate(percentages):
            if p > 0:
                non_zero_behaviors.append(behaviors[i])
                non_zero_percentages.append(p)
                non_zero_colors.append(colors[i])
        
        plt.pie(non_zero_percentages, labels=non_zero_behaviors, colors=non_zero_colors, 
                autopct='%1.1f%%', startangle=90)
        plt.title('Teaching Behaviors Proportion')
        
        # 2. 教学指标雷达图
        plt.subplot(2, 2, 2)
        metrics = [
            'Teaching Activity',
            'Teaching Rhythm',
            'Teaching Diversity',
            'Teaching Engagement',
            'Overall Score'
        ]
        metrics_values = [
            report_data["teaching_metrics"]["teaching_activity"],
            report_data["teaching_metrics"]["teaching_rhythm"],
            report_data["teaching_metrics"]["teaching_diversity"],
            report_data["teaching_metrics"]["teaching_engagement"],
            report_data["teaching_metrics"]["overall_score"]
        ]
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        # 闭合雷达图
        metrics_values.append(metrics_values[0])
        angles.append(angles[0])
        
        ax = plt.subplot(2, 2, 2, polar=True)
        ax.fill(angles, metrics_values, 'b', alpha=0.1)
        ax.plot(angles, metrics_values, 'o-', linewidth=2)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 10)
        plt.title('Teaching Metrics Analysis')
        
        # 3. 教学指标柱状图
        plt.subplot(2, 2, 3)
        metrics_names = [
            'Teaching Activity',
            'Teaching Rhythm',  
            'Teaching Diversity',
            'Teaching Engagement',
            'Overall Score'
        ]
        metrics_scores = [
            report_data["teaching_metrics"]["teaching_activity"],
            report_data["teaching_metrics"]["teaching_rhythm"],
            report_data["teaching_metrics"]["teaching_diversity"],
            report_data["teaching_metrics"]["teaching_engagement"],
            report_data["teaching_metrics"]["overall_score"]
        ]
        
        y_pos = np.arange(len(metrics_names))
        bars = plt.barh(y_pos, metrics_scores, align='center', alpha=0.7)
        plt.yticks(y_pos, metrics_names)
        plt.xlabel('Score')
        plt.title('Teaching Metrics Scores')
        
        # 添加数值标签
        for i, v in enumerate(metrics_scores):
            plt.text(v + 0.1, i, f"{v:.2f}", va='center')
        
        # 4. 教学时长
        plt.subplot(2, 2, 4)
        # 添加文本摘要
        summary_text = (
            f"Teacher Evaluation Report\n\n"
            f"Video: {os.path.basename(video_path)}\n"
            f"Analysis Time: {report_data['teacher_info']['analysis_time']}\n"
            f"Video Duration: {report_data['teacher_info']['duration_minutes']:.2f} minutes\n\n"
            f"Valid Frames Percentage: {report_data['teacher_info']['valid_frames_percentage']:.1f}%\n\n"
            f"Behavior Changes: {report_data['teaching_patterns']['behavior_changes']}\n"
            f"Average Behavior Duration: {report_data['teaching_patterns']['average_behavior_duration_seconds']:.2f} seconds\n"
            f"Dominant Behavior: {report_data['teaching_patterns']['dominant_behavior']}\n\n"
            f"Writing Duration: {report_data['behavior_durations']['writing_minutes']:.2f} minutes\n"
            f"Explaining Duration: {report_data['behavior_durations']['explaining_minutes']:.2f} minutes\n"
            f"Interacting Duration: {report_data['behavior_durations']['interacting_minutes']:.2f} minutes\n"
            f"Moving Duration: {report_data['behavior_durations']['moving_minutes']:.2f} minutes\n"
            f"Standing Duration: {report_data['behavior_durations']['standing_minutes']:.2f} minutes\n\n"
            f"Overall Score: {report_data['teaching_metrics']['overall_score']:.2f}/10\n\n"
            f"Improvement Suggestions:\n{report_data['improvement_suggestions']}"
        )
        plt.text(0, 1, summary_text, fontsize=10, va='top')
        plt.axis('off')
        
        # 保存图表
        plt.tight_layout()
        chart_filename = f"teacher_evaluation_chart_{os.path.splitext(os.path.basename(video_path))[0]}.png"
        plt.savefig(chart_filename, dpi=300)
        plt.close()
        
        print(f"Visualization chart saved as: {chart_filename}")

    def generate_time_based_behavior_analysis(self, behavior_sequence, fps, video_path, output_dir=None):
        """
        生成基于时间的行为分析可视化 - 增强美化版
        
        Args:
            behavior_sequence: 行为序列
            fps: 视频帧率
            video_path: 视频路径
            output_dir: 输出目录
            
        Returns:
            str: 生成的图表文件路径
        """
        if not behavior_sequence or not fps:
            print("无法生成时间分析：缺少行为序列或帧率信息")
            return None
            
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(video_path))
        
        # 获取视频文件名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 设置图表保存路径
        chart_path = os.path.join(output_dir, f"{video_name}_behavior_analysis.png")
        
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.colors import LinearSegmentedColormap, to_rgba
            from datetime import datetime, timedelta
            from matplotlib.ticker import MultipleLocator, MaxNLocator
            import matplotlib as mpl
            from scipy.ndimage import gaussian_filter1d
            
            # 设置全局风格
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 设置字体 - 使用简单无衬线字体
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 18
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            
            # 现代配色方案 - 专业的配色
            behavior_colors = {
                "explaining": "#3498db",  # 蓝色
                "writing": "#2ecc71",     # 绿色
                "moving": "#e74c3c",      # 红色
                "interacting": "#9b59b6", # 紫色
                "standing": "#f39c12",    # 橙色
                "not_in_frame": "#95a5a6", # 灰色
                "unknown": "#7f8c8d"      # 深灰色
            }
            
            # 英文标签
            behavior_labels = {
                "explaining": "Explaining",
                "writing": "Writing",
                "moving": "Moving",
                "interacting": "Interacting",
                "standing": "Standing",
                "not_in_frame": "Not in Frame",
                "unknown": "Unknown"
            }
            
            # 准备数据
            frame_indices = sorted(behavior_sequence.keys())
            behaviors = [behavior_sequence[idx] for idx in frame_indices]
            
            # 应用行为平滑
            smoothed_behaviors = self._smooth_behavior_sequence(behaviors, window_size=25)  # 增加窗口大小从15到25
            
            # 转换为时间
            start_time = datetime(2023, 1, 1, 0, 0, 0)  # 使用虚拟起始时间
            times = [start_time + timedelta(seconds=idx/fps) for idx in frame_indices]
            
            # 计算每种行为的占比
            behavior_counts = {}
            for behavior in behaviors:
                if behavior:
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                    
            total_frames = len(behaviors)
            behavior_percentages = {b: count/total_frames*100 for b, count in behavior_counts.items() if b}
            
            # 创建图表 - 更大尺寸，更高分辨率
            fig = plt.figure(figsize=(16, 10), dpi=200, facecolor='white')
            
            # 添加主标题
            plt.suptitle(f"{video_name} - Teacher Behavior Analysis", 
                       fontsize=22, fontweight='bold', y=0.98)
            
            # 创建主子图和额外的子图(用于饼图)
            gs = mpl.gridspec.GridSpec(2, 3, height_ratios=[4, 1], width_ratios=[1, 1, 1])
            ax_main = plt.subplot(gs[0, :])
            ax_pie = plt.subplot(gs[1, 0])
            ax_bars = plt.subplot(gs[1, 1:])
            
            # 设置行为码值
            behavior_values = {}
            unique_behaviors = sorted(set(behaviors))
            for i, behavior in enumerate(unique_behaviors):
                behavior_values[behavior] = i + 1
                
            # 创建行为数值序列
            behavior_nums = [behavior_values.get(b, 0) for b in smoothed_behaviors]
            
            # 增加数据点密度，使图表更平滑
            times_dense = []
            behavior_nums_dense = []
            
            # 使用插值增加数据点密度
            from scipy.interpolate import interp1d
            
            # 创建时间戳数组（秒为单位）
            time_stamps = [(t - start_time).total_seconds() for t in times]
            
            if len(time_stamps) > 3:  # 至少需要几个点才能插值
                # 创建插值函数
                interp_func = interp1d(time_stamps, behavior_nums, kind='linear')
                
                # 创建更密集的时间戳
                dense_time_stamps = np.linspace(min(time_stamps), max(time_stamps), len(time_stamps) * 10)  # 增加密度倍数从5倍到10倍
                
                # 应用插值
                dense_behavior_nums = interp_func(dense_time_stamps)
                
                # 应用额外平滑 - 平滑窗口
                dense_behavior_nums = gaussian_filter1d(dense_behavior_nums, sigma=8)  # 增加sigma从5到8
                
                # 转换回时间对象
                times_dense = [start_time + timedelta(seconds=t) for t in dense_time_stamps]
                behavior_nums_dense = dense_behavior_nums
            else:
                times_dense = times
                behavior_nums_dense = behavior_nums
            
            # 设置主图的美化样式
            ax_main.set_facecolor('#f8f9fa')
            ax_main.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
            
            # 绘制平滑的行为曲线 - 使用深色渐变线条
            ax_main.plot(times_dense, behavior_nums_dense, linewidth=2.5, 
                     color='#2c3e50', alpha=0.7, zorder=1)
            
            # 美化主图边框
            for spine in ax_main.spines.values():
                spine.set_visible(True)
                spine.set_color('#cccccc')
                spine.set_linewidth(0.8)
            
            # 渐变背景颜色
            cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                                                  ['#ffffff', '#f0f0f0'])
            
            # 为每种行为添加填充颜色 - 使用半透明渐变色
            for i, behavior in enumerate(unique_behaviors):
                if behavior in behavior_colors:
                    # 创建该行为的掩码
                    mask = np.array(behavior_nums_dense) == behavior_values[behavior]
                    if np.any(mask):  # 确保有数据点
                        # 获取颜色并调整透明度
                        base_color = to_rgba(behavior_colors[behavior])
                        fill_color = (base_color[0], base_color[1], base_color[2], 0.6)
                        edge_color = (base_color[0], base_color[1], base_color[2], 0.9)
                        
                        # 计算底部y轴值
                        bottom_y = behavior_values[behavior] - 0.4
                        # 给每个行为绘制带状区域 - 美化边缘和填充
                        ax_main.fill_between(
                            times_dense, 
                            [bottom_y] * len(times_dense),
                            [behavior_values[behavior] + 0.4] * len(times_dense),
                            where=mask, 
                            color=fill_color,
                            edgecolor=edge_color,
                            linewidth=1.0,
                            alpha=0.8,
                            zorder=2,
                            label=behavior_labels.get(behavior, behavior)
                        )
            
            # 设置y轴刻度为行为名称 - 美化标签
            ax_main.set_yticks([behavior_values[b] for b in unique_behaviors])
            ax_main.set_yticklabels(
                [behavior_labels.get(b, b) for b in unique_behaviors],
                fontsize=12, fontweight='medium'
            )
            
            # 设置图表标题和标签 - 使用更明显的样式
            ax_main.set_title("Temporal Behavior Analysis", fontsize=18, pad=15)
            ax_main.set_xlabel("Time (MM:SS)", fontsize=14, labelpad=10)
            ax_main.set_ylabel("Behavior Category", fontsize=14, labelpad=10)
            
            # 格式化x轴为时间格式
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
            
            # 自动调整x轴标签数量 - 更好的间隔
            ax_main.xaxis.set_major_locator(MaxNLocator(10))
            
            # 旋转x轴标签以避免重叠
            plt.setp(ax_main.get_xticklabels(), rotation=30, ha='right')
            
            # 自定义图例 - 放在图表下方中央位置
            handles, labels = ax_main.get_legend_handles_labels()
            ax_main.legend(
                handles, labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, -0.15), 
                ncol=min(5, len(unique_behaviors)),
                frameon=True, 
                facecolor='white', 
                edgecolor='#cccccc',
                fontsize=11,
                title="Behavior Types",
                title_fontsize=13
            )
            
            # 绘制饼图 - 行为分布
            if behavior_percentages:
                wedges, texts, autotexts = ax_pie.pie(
                    [behavior_percentages.get(b, 0) for b in unique_behaviors],
                    labels=None,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=[behavior_colors.get(b, '#999999') for b in unique_behaviors],
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
                    textprops={'fontsize': 10, 'color': 'white', 'fontweight': 'bold'},
                    explode=[0.05] * len(unique_behaviors),
                    shadow=False
                )
                
                # 美化饼图文本
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                
                # 设置饼图标题
                ax_pie.set_title('Behavior Distribution', fontsize=14, pad=10)
                
                # 自定义饼图图例
                ax_pie.legend(
                    wedges,
                    [behavior_labels.get(b, b) for b in unique_behaviors],
                    loc='center left',
                    bbox_to_anchor=(1, 0.5),
                    frameon=False,
                    fontsize=10
                )
            
            # 绘制条形图 - 行为时长统计
            if behavior_percentages:
                # 排序行为，从高到低
                sorted_behaviors = sorted(
                    behavior_percentages.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # 提取数据
                bar_labels = [behavior_labels.get(b[0], b[0]) for b in sorted_behaviors]
                bar_values = [b[1] for b in sorted_behaviors]
                bar_colors = [behavior_colors.get(b[0], '#999999') for b in sorted_behaviors]
                
                # 创建水平条形图
                bars = ax_bars.barh(
                    bar_labels, 
                    bar_values,
                    color=bar_colors,
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=1
                )
                
                # 添加数值标签
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax_bars.text(
                        width + 1, 
                        bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%',
                        va='center',
                        fontsize=9,
                        fontweight='bold',
                        color='#444444'
                    )
                
                # 美化条形图
                ax_bars.set_title('Time Allocation', fontsize=14, pad=10)
                ax_bars.set_xlabel('Percentage (%)', fontsize=12, labelpad=8)
                ax_bars.set_xlim(0, max(bar_values) * 1.15)  # 留出空间显示标签
                ax_bars.grid(axis='x', linestyle='--', alpha=0.6)
                ax_bars.spines['top'].set_visible(False)
                ax_bars.spines['right'].set_visible(False)
            
            # 调整子图之间的间距
            plt.subplots_adjust(
                top=0.9,
                bottom=0.15,
                left=0.1,
                right=0.95,
                hspace=0.4,
                wspace=0.3
            )
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
            
            # 自动调整x轴标签数量
            ax.xaxis.set_major_locator(MaxNLocator(10))
            
            # 添加网格
            plt.grid(True, alpha=0.3)
            
            # 添加图例
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(unique_behaviors))
            
            # 添加行为占比信息
            stats_text = "Behavior Statistics:\n"
            for behavior in sorted(behavior_percentages.keys(), key=lambda x: behavior_percentages[x], reverse=True):
                if behavior in behavior_labels:
                    stats_text += f"{behavior_labels[behavior]}: {behavior_percentages[behavior]:.1f}%\n"
            
            plt.annotate(
                stats_text, 
                xy=(0.02, 0.98), 
                xycoords='axes fraction',
                verticalalignment='top',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
            )
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(chart_path, dpi=200)
            plt.close()
            
            print(f"行为时间分析图表已保存为: {chart_path}")
            return chart_path
            
        except Exception as e:
            print(f"生成时间分析图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _smooth_behavior_sequence(self, behaviors, window_size=25):
        """
        平滑行为序列，减少行为跳变
        
        Args:
            behaviors: 行为序列列表
            window_size: 平滑窗口大小
            
        Returns:
            list: 平滑后的行为序列
        """
        if not behaviors or window_size <= 1:
            return behaviors
            
        smoothed = []
        half_window = window_size // 2
        
        # 先对行为序列进行初步平滑
        for i in range(len(behaviors)):
            # 获取窗口范围
            start = max(0, i - half_window)
            end = min(len(behaviors), i + half_window + 1)
            window = behaviors[start:end]
            
            # 统计窗口内各行为出现次数
            behavior_counts = {}
            for j, b in enumerate(window):
                if b:
                    # 距离当前位置越近，权重越高
                    distance = abs(start + j - i)
                    weight = 1.0
                    
                    # 当前位置权重非常高
                    if distance == 0:
                        weight = 5.0  # 大幅增加当前位置权重
                    # 邻近位置权重也较高
                    elif distance <= 2:
                        weight = 3.0
                    elif distance <= 5:
                        weight = 2.0
                    elif distance <= 10:
                        weight = 1.5
                    
                    behavior_counts[b] = behavior_counts.get(b, 0) + weight
            
            # 如果窗口内有行为，选择出现最多的行为
            if behavior_counts:
                # 选择最常见的行为
                most_common = max(behavior_counts.items(), key=lambda x: x[1])
                smoothed.append(most_common[0])
            else:
                # 如果窗口内没有有效行为，保留原行为
                smoothed.append(behaviors[i])
        
        # 二次平滑处理，消除残余跳变
        double_smoothed = []
        for i in range(len(smoothed)):
            # 小窗口二次平滑
            small_window_size = 11  # 较小的窗口用于二次平滑
            small_half = small_window_size // 2
            start = max(0, i - small_half)
            end = min(len(smoothed), i + small_half + 1)
            window = smoothed[start:end]
            
            # 统计窗口内各行为出现次数
            behavior_counts = {}
            for b in window:
                if b:
                    behavior_counts[b] = behavior_counts.get(b, 0) + 1
            
            # 如果窗口内有行为，选择出现最多的行为
            if behavior_counts:
                # 给当前行为额外权重，但小于第一次平滑
                current_behavior = smoothed[i]
                if current_behavior in behavior_counts:
                    behavior_counts[current_behavior] += 1.5
                
                most_common = max(behavior_counts.items(), key=lambda x: x[1])
                
                # 如果当前位置与前后位置行为不同，检查是否应当平滑
                if i > 0 and i < len(smoothed) - 1:
                    prev_behavior = smoothed[i-1]
                    next_behavior = smoothed[i+1]
                    
                    # 如果前后行为相同但当前不同，可能是孤立的跳变
                    if prev_behavior == next_behavior and current_behavior != prev_behavior:
                        # 如果孤立行为段长度小于等于3帧，视为噪声
                        isolated_length = 1
                        # 向前查找相同行为
                        for j in range(i-1, max(0, i-5), -1):
                            if smoothed[j] == current_behavior:
                                isolated_length += 1
                            else:
                                break
                        # 向后查找相同行为
                        for j in range(i+1, min(len(smoothed), i+5)):
                            if smoothed[j] == current_behavior:
                                isolated_length += 1
                            else:
                                break
                                
                        # 如果是短暂行为，替换为前一个行为
                        if isolated_length <= 3:
                            double_smoothed.append(prev_behavior)
                            continue
                
                double_smoothed.append(most_common[0])
            else:
                double_smoothed.append(smoothed[i])
                
        return double_smoothed

    def enhance_image(self, image):
        """增强图像以提高关键点检测质量"""
        try:
            # 转换为RGB以便于处理
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 已经是彩色图像
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 灰度图像转RGB
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 计算图像亮度
            brightness = np.mean(img_rgb)
            
            # 自适应参数调整
            if brightness < 100:  # 暗图像
                alpha = 1.7  # 增加对比度
                beta = 30    # 增加亮度
            elif brightness > 200:  # 亮图像
                alpha = 0.8  # 降低对比度
                beta = -10   # 降低亮度
            else:  # 正常亮度
                alpha = 1.3
                beta = 15
            
            # 应用亮度和对比度调整
            enhanced = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
            
            # 应用CLAHE增强对比度
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # 使用更强的CLAHE参数
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # 合并通道
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # 锐化处理以增强边缘
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced_sharp = cv2.filter2D(enhanced_rgb, -1, kernel)
            
            # 应用高斯模糊减少噪声
            enhanced_filtered = cv2.GaussianBlur(enhanced_sharp, (3, 3), 0)
            
            # 转回BGR用于OpenCV处理
            enhanced_bgr = cv2.cvtColor(enhanced_filtered, cv2.COLOR_RGB2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            print(f"图像增强出错: {str(e)}")
            return image  # 出错时返回原始图像

    def interpolate_missing_keypoints(self, keypoints_dict):
        """插值计算缺失的关键点"""
        if not keypoints_dict:
            # 如果没有任何关键点，创建一个默认的姿势（居中站立姿势）
            if len(self.pose_history) > 0 and any(len(hist) > 0 for hist in self.pose_history[-10:]):
                # 使用历史数据创建默认姿势
                for hist in reversed(self.pose_history):
                    if len(hist) > 3:  # 至少有几个关键点
                        return hist  # 直接使用最近的有效姿势
            return keypoints_dict
        
        new_keypoints = keypoints_dict.copy()
        
        # 1. 如果有左右肩膀，但没有鼻子，可以估计鼻子位置
        if "left_shoulder" in keypoints_dict and "right_shoulder" in keypoints_dict and "nose" not in keypoints_dict:
            left_shoulder = keypoints_dict["left_shoulder"]
            right_shoulder = keypoints_dict["right_shoulder"]
            
            # 计算肩膀中心
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2)
            
            # 假设鼻子在肩膀中心上方约1/3的身高
            estimated_nose = (shoulder_center[0], shoulder_center[1] - 70)
            new_keypoints["nose"] = estimated_nose
            print("估计鼻子位置")
        
        # 2. 如果有左右肩膀，但没有髋部，可以估计髋部位置
        if "left_shoulder" in keypoints_dict and "right_shoulder" in keypoints_dict:
            left_shoulder = keypoints_dict["left_shoulder"]
            right_shoulder = keypoints_dict["right_shoulder"]
            
            if "left_hip" not in keypoints_dict:
                # 假设左髋部在左肩下方
                estimated_left_hip = (left_shoulder[0], left_shoulder[1] + 150)
                new_keypoints["left_hip"] = estimated_left_hip
                print("估计左髋部位置")
            
            if "right_hip" not in keypoints_dict:
                # 假设右髋部在右肩下方
                estimated_right_hip = (right_shoulder[0], right_shoulder[1] + 150)
                new_keypoints["right_hip"] = estimated_right_hip
                print("估计右髋部位置")
        
        # 3. 如果有肩膀但没有肘部，可以估计肘部位置
        if "left_shoulder" in keypoints_dict and "left_elbow" not in keypoints_dict:
            left_shoulder = keypoints_dict["left_shoulder"]
            # 估计左肘位置
            estimated_left_elbow = (left_shoulder[0] - 50, left_shoulder[1] + 50)
            new_keypoints["left_elbow"] = estimated_left_elbow
            print("估计左肘位置")
        
        if "right_shoulder" in keypoints_dict and "right_elbow" not in keypoints_dict:
            right_shoulder = keypoints_dict["right_shoulder"]
            # 估计右肘位置
            estimated_right_elbow = (right_shoulder[0] + 50, right_shoulder[1] + 50)
            new_keypoints["right_elbow"] = estimated_right_elbow
            print("估计右肘位置")
        
        # 4. 如果有肘部但没有手腕，可以估计手腕位置
        if "left_elbow" in new_keypoints and "left_wrist" not in keypoints_dict:
            left_elbow = new_keypoints["left_elbow"]
            # 估计左手腕位置
            estimated_left_wrist = (left_elbow[0] - 50, left_elbow[1] + 20)
            new_keypoints["left_wrist"] = estimated_left_wrist
            print("估计左手腕位置")
        
        if "right_elbow" in new_keypoints and "right_wrist" not in keypoints_dict:
            right_elbow = new_keypoints["right_elbow"]
            # 估计右手腕位置
            estimated_right_wrist = (right_elbow[0] + 50, right_elbow[1] + 20)
            new_keypoints["right_wrist"] = estimated_right_wrist
            print("估计右手腕位置")
        
        # 5. 如果有髋部但没有膝盖，可以估计膝盖位置
        if "left_hip" in new_keypoints and "left_knee" not in keypoints_dict:
            left_hip = new_keypoints["left_hip"]
            # 估计左膝盖位置
            estimated_left_knee = (left_hip[0], left_hip[1] + 120)
            new_keypoints["left_knee"] = estimated_left_knee
            print("估计左膝盖位置")
        
        if "right_hip" in new_keypoints and "right_knee" not in keypoints_dict:
            right_hip = new_keypoints["right_hip"]
            # 估计右膝盖位置
            estimated_right_knee = (right_hip[0], right_hip[1] + 120)
            new_keypoints["right_knee"] = estimated_right_knee
            print("估计右膝盖位置")
        
        # 计算重要的角度
        self.calculate_additional_angles(new_keypoints)
        
        return new_keypoints

    def calculate_additional_angles(self, keypoints_dict):
        """计算额外的角度和特征"""
        # 计算肩膀角度
        if "left_shoulder" in keypoints_dict and "right_shoulder" in keypoints_dict:
            left_shoulder = keypoints_dict["left_shoulder"]
            right_shoulder = keypoints_dict["right_shoulder"]
            
            # 肩膀角度（水平线与肩膀连线的夹角）
            shoulder_angle = np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ) * 180 / np.pi
            keypoints_dict["shoulder_angle"] = shoulder_angle
        
        # 计算身体垂直角度
        if "left_shoulder" in keypoints_dict and "right_shoulder" in keypoints_dict and \
           "left_hip" in keypoints_dict and "right_hip" in keypoints_dict:
            left_shoulder = keypoints_dict["left_shoulder"]
            right_shoulder = keypoints_dict["right_shoulder"]
            left_hip = keypoints_dict["left_hip"]
            right_hip = keypoints_dict["right_hip"]
            
            # 计算肩膀中心和髋部中心
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                              (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2)
            
            # 计算身体角度
            body_angle = np.arctan2(
                shoulder_center[0] - hip_center[0],
                shoulder_center[1] - hip_center[1]
            ) * 180 / np.pi
            keypoints_dict["body_angle"] = body_angle
        
        # 计算手臂角度
        if "left_shoulder" in keypoints_dict and "left_elbow" in keypoints_dict and "left_wrist" in keypoints_dict:
            left_shoulder = keypoints_dict["left_shoulder"]
            left_elbow = keypoints_dict["left_elbow"]
            left_wrist = keypoints_dict["left_wrist"]
            
            # 计算左臂角度
            vec1 = np.array(left_elbow) - np.array(left_shoulder)
            vec2 = np.array(left_wrist) - np.array(left_elbow)
            
            cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            left_arm_angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
            keypoints_dict["left_arm_angle"] = left_arm_angle
        
        if "right_shoulder" in keypoints_dict and "right_elbow" in keypoints_dict and "right_wrist" in keypoints_dict:
            right_shoulder = keypoints_dict["right_shoulder"]
            right_elbow = keypoints_dict["right_elbow"]
            right_wrist = keypoints_dict["right_wrist"]
            
            # 计算右臂角度
            vec1 = np.array(right_elbow) - np.array(right_shoulder)
            vec2 = np.array(right_wrist) - np.array(right_elbow)
            
            cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            right_arm_angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
            keypoints_dict["right_arm_angle"] = right_arm_angle
        
        return keypoints_dict

    def calculate_behavior_confidence(self, keypoints_dict, behavior):
        """计算行为判断的置信度"""
        # 获取最近的行为历史
        recent_behaviors = self.behavior_history[-10:] if len(self.behavior_history) >= 10 else self.behavior_history
        
        # 计算行为稳定性得分
        stability_score = 0.0
        if recent_behaviors:
            same_behavior_count = sum(1 for b in recent_behaviors if b == behavior)
            stability_score = same_behavior_count / len(recent_behaviors)
        
        # 计算关键点质量得分
        quality_score = 0.0
        required_points = {
            "writing": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
            "explaining": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
            "interacting": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
            "moving": ["nose", "left_hip", "right_hip"],
            "standing": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"]
        }
        
        if behavior in required_points:
            available_points = sum(1 for point in required_points[behavior] if point in keypoints_dict)
            quality_score = available_points / len(required_points[behavior])
        
        # 计算姿态一致性得分
        consistency_score = 0.0
        if behavior == "writing":
            consistency_score = self.calculate_writing_score(keypoints_dict)
        elif behavior == "explaining":
            consistency_score = self.calculate_explaining_score(keypoints_dict)
        elif behavior == "interacting":
            consistency_score = self.calculate_interacting_score(keypoints_dict)
        elif behavior == "moving":
            consistency_score = self.calculate_moving_score(keypoints_dict)
        elif behavior == "standing":
            consistency_score = self.calculate_standing_score(keypoints_dict)
        
        # 综合计算置信度
        confidence = (stability_score * 0.3 + quality_score * 0.3 + consistency_score * 0.4)
        return confidence

    def analyze_behavior(self, keypoints_dict):
        """基于关键点分析教师的行为"""
        # 如果关键点极少，判断为不在镜头中
        if not keypoints_dict or len(keypoints_dict) < 3:
            return "not_in_frame"  # 不在镜头中
        
        # 获取关键点
        nose = keypoints_dict.get("nose")
        left_eye = keypoints_dict.get("left_eye")
        right_eye = keypoints_dict.get("right_eye")
        left_shoulder = keypoints_dict.get("left_shoulder")
        right_shoulder = keypoints_dict.get("right_shoulder")
        left_wrist = keypoints_dict.get("left_wrist")
        right_wrist = keypoints_dict.get("right_wrist")
        left_hip = keypoints_dict.get("left_hip")
        right_hip = keypoints_dict.get("right_hip")
        
        # 检查是否背对镜头
        if (left_shoulder and right_shoulder and 
            not nose and not left_eye and not right_eye):
            return "back_to_camera"
            
        # 计算各种行为的得分
        moving_score = self.calculate_moving_score(keypoints_dict)
        pointing_score = self.calculate_pointing_score(keypoints_dict)
        explaining_score = self.calculate_explaining_score(keypoints_dict)
        writing_score = self.calculate_writing_score(keypoints_dict)
        
        # 上下身姿态得分分离计算
        upper_body_standing_score = self.calculate_upper_body_standing_score(keypoints_dict) * 1.2
        lower_body_standing_score = self.calculate_lower_body_standing_score(keypoints_dict) * 1.2
        
        # 整体站立得分取上下身得分的加权平均
        standing_score = upper_body_standing_score * 0.5 + lower_body_standing_score * 0.5
        # 只有在明确检测到坐姿特征时才计算坐姿得分，且降低权重
        sitting_score = self.calculate_sitting_score(keypoints_dict) * 0.5
        
        # 添加举手得分
        raising_hand_score = self.calculate_raising_hand_score(keypoints_dict)
        
        # 存储上下身得分信息（可用于可视化和报告）
        self.posture_details = {
            "upper_body_standing": upper_body_standing_score,
            "lower_body_standing": lower_body_standing_score,
            "overall_standing": standing_score,
            "sitting": sitting_score
        }
        
        # 基于肢体姿态调整手部行为得分
        # 站立状态下，手部行为更重要
        if standing_score > 0.5:
            explaining_score *= 1.2
            pointing_score *= 1.2
            writing_score *= 1.2
            raising_hand_score *= 1.3  # 站立时举手动作明显
        
        # 解决坐立冲突 - 默认假设为站立，只有在坐姿得分明显高于站立得分时才改为坐姿
        if sitting_score > standing_score + 0.3:  # 设置一个阈值差，确保坐姿得分明显高
            standing_score = 0
        else:
            sitting_score = 0
            # 如果站立得分不高但也没有明显的坐姿，默认为站立
            if standing_score < 0.4:  # 降低站立判定阈值
                standing_score = 0.4  # 设置一个最小站立得分
        
        # 计算手部动作得分总和
        hand_movement_scores = {
            "explaining": explaining_score,
            "pointing": pointing_score,
            "writing": writing_score,
            "raising_hand": raising_hand_score
        }
        
        # 找出得分最高的手部动作
        max_hand_movement = max(hand_movement_scores.items(), key=lambda x: x[1])
        max_hand_movement_type, max_hand_movement_score = max_hand_movement
        
        # 创建行为得分字典
        behavior_scores = {
            "moving": moving_score,
            max_hand_movement_type: max_hand_movement_score,  # 只保留得分最高的手部动作
            "standing": standing_score,
            "sitting": sitting_score
        }
        
        # 返回得分最高的行为
        max_behavior = max(behavior_scores.items(), key=lambda x: x[1])
        behavior_type, behavior_score = max_behavior
        
        # 如果移动得分超过阈值，优先判断为移动
        if moving_score > 0.6:
            return "moving"
            
        # 如果手部动作得分显著高于姿态得分，返回手部动作
        if max_hand_movement_score > 0.6 and max_hand_movement_score > max(standing_score, sitting_score):
            return max_hand_movement_type
            
        # 如果坐立姿态得分明显，返回相应姿态
        if standing_score > 0.4:  # 降低站立判定阈值
            return "standing"
        if sitting_score > 0.6:  # 提高坐姿判定阈值，更严格
            return "sitting"
            
        # 如果所有得分都很低，默认返回站立
        return "standing"
    
    def calculate_writing_score(self, keypoints_dict):
        """Calculate the score for writing on the board behavior"""
        score = 0.0
        
        # Get keypoints
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_elbow = keypoints_dict.get('left_elbow')
        right_elbow = keypoints_dict.get('right_elbow')
        left_wrist = keypoints_dict.get('left_wrist')
        right_wrist = keypoints_dict.get('right_wrist')
        
        if all([left_shoulder, right_shoulder]) and (left_wrist or right_wrist):
            # Lower the difficulty of writing detection
            
            # Check if hands are raised - only one hand needs to be raised
            hands_raised = False
            if left_wrist and left_wrist[1] < left_shoulder[1] - 20:  # Left hand raised
                hands_raised = True
                score += 0.4
            elif right_wrist and right_wrist[1] < right_shoulder[1] - 20:  # Right hand raised
                hands_raised = True
                score += 0.4
                
            # There's a certain probability of writing on the board if hands are raised
            if hands_raised:
                score += 0.3
            
            # Check arm angles
            if left_elbow and left_wrist:
                left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                if 60 < left_arm_angle < 140:  # Wider angle range
                    score += 0.2
            
            if right_elbow and right_wrist:
                right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                if 60 < right_arm_angle < 140:  # Wider angle range
                    score += 0.2
        
        return score
    
    def calculate_explaining_score(self, keypoints_dict):
        """Calculate the score for explaining behavior"""
        score = 0.2  # Adjusted base score
        
        # Get keypoints
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_elbow = keypoints_dict.get('left_elbow')
        right_elbow = keypoints_dict.get('right_elbow')
        left_wrist = keypoints_dict.get('left_wrist')
        right_wrist = keypoints_dict.get('right_wrist')
        
        # Check arm poses
        has_left_arm = left_shoulder and left_elbow and left_wrist
        has_right_arm = right_shoulder and right_elbow and right_wrist
        
        # Check gestures - at least one side arm visible
        if has_left_arm or has_right_arm:
            score += 0.1
            
            # Check left arm raised
            if has_left_arm:
                # Wrist above shoulder - typical explaining gesture
                if left_wrist[1] < left_shoulder[1]:
                    score += 0.3
                # Arm bent (by elbow angle)
                elif left_elbow[1] < left_shoulder[1]:
                    score += 0.2
            
            # Check right arm raised
            if has_right_arm:
                # Wrist above shoulder
                if right_wrist[1] < right_shoulder[1]:
                    score += 0.3
                # Arm bent
                elif right_elbow[1] < right_shoulder[1]:
                    score += 0.2
        
        return score
    
    def calculate_interacting_score(self, keypoints_dict):
        """Calculate the score for interacting behavior"""
        score = 0.0
        
        # Get keypoints
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_elbow = keypoints_dict.get('left_elbow')
        right_elbow = keypoints_dict.get('right_elbow')
        left_wrist = keypoints_dict.get('left_wrist')
        right_wrist = keypoints_dict.get('right_wrist')
        
        if all([left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]):
            # Calculate arm angles
            left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Check arm movement amplitude
            if abs(left_wrist[0] - left_shoulder[0]) > 50 or abs(right_wrist[0] - right_shoulder[0]) > 50:
                score += 0.4
            
            # Check arm angle changes
            if 20 < left_arm_angle < 160 and 20 < right_arm_angle < 160:
                score += 0.3
            
            # Check if arms are on both sides of the body
            if (left_wrist[0] < left_shoulder[0] - 30 or left_wrist[0] > left_shoulder[0] + 30) and \
               (right_wrist[0] < right_shoulder[0] - 30 or right_wrist[0] > right_shoulder[0] + 30):
                score += 0.3
        
        return score
    
    def calculate_moving_score(self, keypoints_dict):
        """Calculate the score for moving behavior"""
        score = 0.0
        
        # Check if there are reference points
        if 'nose' in keypoints_dict or 'left_shoulder' in keypoints_dict or 'right_shoulder' in keypoints_dict:
            current_position = None
            
            # Use the nose or midpoint of shoulders as position reference
            if 'nose' in keypoints_dict:
                current_position = keypoints_dict['nose']
            else:
                left_shoulder = keypoints_dict.get('left_shoulder')
                right_shoulder = keypoints_dict.get('right_shoulder')
                
                if left_shoulder and right_shoulder:
                    current_position = [(left_shoulder[0] + right_shoulder[0]) / 2,
                                        (left_shoulder[1] + right_shoulder[1]) / 2]
                elif left_shoulder:
                    current_position = left_shoulder
                elif right_shoulder:
                    current_position = right_shoulder
            
            # If there's a previous frame position record, calculate movement distance
            if self.last_position and current_position:
                distance = np.sqrt((current_position[0] - self.last_position[0])**2 + 
                                  (current_position[1] - self.last_position[1])**2)
                
                # Adjust movement thresholds
                if distance > 80:  # Large movement
                    score = 0.9
                elif distance > 40:  # Medium movement
                    score = 0.6
                elif distance > 20:  # Small movement
                    score = 0.3
                
            # Update previous frame position
            if current_position:
                self.last_position = current_position
        
        return score
    
    def calculate_standing_score(self, keypoints_dict):
        """Calculate the score for standing behavior"""
        score = 0.3  # Reduced standing base score
        
        # Get keypoints
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_hip = keypoints_dict.get('left_hip')
        right_hip = keypoints_dict.get('right_hip')
        left_knee = keypoints_dict.get('left_knee')
        right_knee = keypoints_dict.get('right_knee')
        left_ankle = keypoints_dict.get('left_ankle')
        right_ankle = keypoints_dict.get('right_ankle')
        
        # If there are not enough keypoints, cannot determine standing
        if not (left_shoulder or right_shoulder) or not (left_hip or right_hip):
            return 0.0
            
        # If shoulders and hips are visible, check vertical alignment
        if (left_shoulder and left_hip) or (right_shoulder and right_hip):
            score += 0.2
            
            # Check vertical standing pose - shoulders above hips
            if left_shoulder and left_hip and left_shoulder[1] < left_hip[1]:
                score += 0.1
            if right_shoulder and right_hip and right_shoulder[1] < right_hip[1]:
                score += 0.1
                
            # Check knees below hips (standing feature)
            if left_hip and left_knee and left_hip[1] < left_knee[1]:
                score += 0.1
            if right_hip and right_knee and right_hip[1] < right_knee[1]:
                score += 0.1
                
            # Check knees straight (standing feature)
            if left_knee and left_ankle and abs(left_knee[0] - left_ankle[0]) < 30:
                score += 0.1
            if right_knee and right_ankle and abs(right_knee[0] - right_ankle[0]) < 30:
                score += 0.1
                
            # Check for sitting pose features (if present, reduce standing score)
            if (left_hip and left_knee and left_knee[1] < left_hip[1] + 20) or \
               (right_hip and right_knee and right_knee[1] < right_hip[1] + 20):
                score -= 0.3  # Knees near or above hips are sitting features, reduce standing score
        
        return min(1.0, max(0.0, score))
    
    def calculate_upper_body_standing_score(self, keypoints_dict):
        """Calculate the score for upper body standing pose"""
        score = 0.5  # Initial score, assuming standing pose
        
        # Get keypoints
        nose = keypoints_dict.get('nose')
        left_eye = keypoints_dict.get('left_eye')
        right_eye = keypoints_dict.get('right_eye')
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_hip = keypoints_dict.get('left_hip')
        right_hip = keypoints_dict.get('right_hip')
        
        # Check if there are enough keypoints to evaluate upper body pose
        if not (left_shoulder or right_shoulder):
            return score  # Return default score
        
        # Shoulder vertical position evaluation
        if (left_shoulder and left_hip) or (right_shoulder and right_hip):
            # Check if shoulders are above hips (standing feature)
            if left_shoulder and left_hip and left_shoulder[1] < left_hip[1]:
                score += 0.15
            if right_shoulder and right_hip and right_shoulder[1] < right_hip[1]:
                score += 0.15
            
            # Check vertical distance between shoulders and hips (usually larger for standing)
            if left_shoulder and left_hip:
                vertical_distance = abs(left_shoulder[1] - left_hip[1])
                if vertical_distance > 50:  # Adjust threshold as needed
                    score += 0.1
            if right_shoulder and right_hip:
                vertical_distance = abs(right_shoulder[1] - right_hip[1])
                if vertical_distance > 50:
                    score += 0.1
        
        # Head position evaluation (if visible)
        if nose and (left_shoulder or right_shoulder):
            # Head above shoulders
            shoulder_y = min(left_shoulder[1] if left_shoulder else float('inf'), 
                           right_shoulder[1] if right_shoulder else float('inf'))
            if nose[1] < shoulder_y:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def calculate_lower_body_standing_score(self, keypoints_dict):
        """Calculate the score for lower body standing pose"""
        score = 0.5  # Initial score, assuming standing pose
        
        # Get keypoints
        left_hip = keypoints_dict.get('left_hip')
        right_hip = keypoints_dict.get('right_hip')
        left_knee = keypoints_dict.get('left_knee')
        right_knee = keypoints_dict.get('right_knee')
        left_ankle = keypoints_dict.get('left_ankle')
        right_ankle = keypoints_dict.get('right_ankle')
        
        # Check if there are enough keypoints to evaluate lower body pose
        if not (left_hip or right_hip) or not (left_knee or right_knee):
            return score  # Return default score
        
        # Vertical relationship between hips and knees
        if (left_hip and left_knee) or (right_hip and right_knee):
            # Check if knees are below hips (standing feature)
            if left_hip and left_knee and left_hip[1] < left_knee[1]:
                score += 0.15
            if right_hip and right_knee and right_hip[1] < right_knee[1]:
                score += 0.15
            
            # Check vertical distance between hips and knees (usually larger for standing)
            if left_hip and left_knee:
                vertical_distance = abs(left_hip[1] - left_knee[1])
                if vertical_distance > 40:  # Adjust threshold as needed
                    score += 0.1
            if right_hip and right_knee:
                vertical_distance = abs(right_hip[1] - right_knee[1])
                if vertical_distance > 40:
                    score += 0.1
        
        # Vertical relationship between knees and ankles
        if (left_knee and left_ankle) or (right_knee and right_ankle):
            # Check if ankles are below knees (standing feature)
            if left_knee and left_ankle and left_knee[1] < left_ankle[1]:
                score += 0.1
            if right_knee and right_ankle and right_knee[1] < right_ankle[1]:
                score += 0.1
            
            # Check if knees are straight (standing feature)
            if left_knee and left_ankle and abs(left_knee[0] - left_ankle[0]) < 30:
                score += 0.1
            if right_knee and right_ankle and abs(right_knee[0] - right_ankle[0]) < 30:
                score += 0.1
        
        # Check for sitting pose features (if present, reduce score)
        if (left_hip and left_knee and left_knee[1] < left_hip[1] + 10) or \
           (right_hip and right_knee and right_knee[1] < right_hip[1] + 10):
            score -= 0.4  # Knees near or above hips are strong sitting features
        
        return min(1.0, max(0.0, score))
        
    def calculate_sitting_score(self, keypoints_dict):
        """Calculate the score for sitting behavior"""
        score = 0.0  # Initial score is 0
        
        # Get keypoints
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_hip = keypoints_dict.get('left_hip')
        right_hip = keypoints_dict.get('right_hip')
        left_knee = keypoints_dict.get('left_knee')
        right_knee = keypoints_dict.get('right_knee')
        left_ankle = keypoints_dict.get('left_ankle')
        right_ankle = keypoints_dict.get('right_ankle')
        
        # 如果没有足够的关键点，无法判断坐姿
        if not (left_hip or right_hip) or not (left_knee or right_knee):
            return 0.0
            
        # 坐姿的主要特征是膝盖与髋部高度接近或膝盖高于髋部
        if (left_hip and left_knee):
            if left_knee[1] <= left_hip[1] + 20:  # 膝盖接近或高于髋部
                score += 0.3
            # 坐姿时膝盖通常在髋部前方
            if left_knee[0] > left_hip[0] + 15:
                score += 0.2
                
        if (right_hip and right_knee):
            if right_knee[1] <= right_hip[1] + 20:  # 膝盖接近或高于髋部
                score += 0.3
            # 坐姿时膝盖通常在髋部前方
            if right_knee[0] < right_hip[0] - 15:  # 右膝在右髋前方（坐标系中x值更小）
                score += 0.2
                
        # 检查膝盖弯曲（坐姿特征）
        if left_knee and left_ankle and left_hip:
            knee_ankle_dist = np.sqrt((left_knee[0] - left_ankle[0])**2 + (left_knee[1] - left_ankle[1])**2)
            hip_knee_dist = np.sqrt((left_hip[0] - left_knee[0])**2 + (left_hip[1] - left_knee[1])**2)
            if knee_ankle_dist < hip_knee_dist * 0.7:  # 膝盖到脚踝距离明显小于髋部到膝盖距离
                score += 0.2
                
        if right_knee and right_ankle and right_hip:
            knee_ankle_dist = np.sqrt((right_knee[0] - right_ankle[0])**2 + (right_knee[1] - right_ankle[1])**2)
            hip_knee_dist = np.sqrt((right_hip[0] - right_knee[0])**2 + (right_hip[1] - right_knee[1])**2)
            if knee_ankle_dist < hip_knee_dist * 0.7:
                score += 0.2
                
        # 降低明显站立姿势的坐姿得分
        if (left_shoulder and left_hip and left_knee and 
            left_shoulder[1] < left_hip[1] and left_hip[1] < left_knee[1] and
            abs(left_shoulder[0] - left_hip[0]) < 20):  # 垂直对齐的站姿
            score -= 0.3
            
        if (right_shoulder and right_hip and right_knee and 
            right_shoulder[1] < right_hip[1] and right_hip[1] < right_knee[1] and
            abs(right_shoulder[0] - right_hip[0]) < 20):  # 垂直对齐的站姿
            score -= 0.3
            
        return min(1.0, max(0.0, score))
    
    def calculate_angle(self, p1, p2, p3):
        """计算三个点形成的角度"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # 计算向量的模
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        
        return angle

    def calculate_posture_score(self, keypoints):
        """计算姿态评分 (0-10分)"""
        if not keypoints or len(keypoints) < 17:
            return 0.0
            
        try:
            # 转换为字典以便访问
            keypoints_dict = {}
            if isinstance(keypoints, dict):
                keypoints_dict = keypoints
            elif isinstance(keypoints, (list, np.ndarray)) and len(keypoints) >= 17:
                # 假设是17关键点的COCO格式
                mapping = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                          "left_wrist", "right_wrist", "left_hip", "right_hip",
                          "left_knee", "right_knee", "left_ankle", "right_ankle"]
                for i, point in enumerate(keypoints):
                    if i < len(mapping):
                        keypoints_dict[mapping[i]] = point
            else:
                return 0.0
                
            # 所需的关键点
            required_points = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", 
                             "left_knee", "right_knee", "nose"]
            
            # 检查是否所有所需的关键点都存在
            if not all(point in keypoints_dict for point in required_points):
                return 0.0
                
            # 分数计算组件
            score = 0.0
            
            # 1. 躯干垂直度评分 (0-4分)
            left_shoulder = keypoints_dict["left_shoulder"]
            right_shoulder = keypoints_dict["right_shoulder"]
            left_hip = keypoints_dict["left_hip"]
            right_hip = keypoints_dict["right_hip"]
            
            # 计算躯干中线斜率
            torso_center_top = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                               (left_shoulder[1] + right_shoulder[1]) / 2)
            torso_center_bottom = ((left_hip[0] + right_hip[0]) / 2, 
                                  (left_hip[1] + right_hip[1]) / 2)
            
            # 计算躯干角度与垂直线的偏差
            torso_angle = abs(90 - abs(np.arctan2(torso_center_bottom[1] - torso_center_top[1],
                                               torso_center_bottom[0] - torso_center_top[0]) * 180 / np.pi))
            
            # 垂直度分数 (偏差越小，分数越高)
            verticality_score = max(0, 4 - (torso_angle / 15))
            score += verticality_score
            
            # 2. 肩膀平衡度评分 (0-2分)
            shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_balance_score = max(0, 2 - (shoulder_height_diff / 20))
            score += shoulder_balance_score
            
            # 3. 站姿稳定性评分 (0-2分)
            if "left_knee" in keypoints_dict and "right_knee" in keypoints_dict:
                left_knee = keypoints_dict["left_knee"]
                right_knee = keypoints_dict["right_knee"]
                
                # 计算髋部与膝盖的位置关系
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                knee_center = ((left_knee[0] + right_knee[0]) / 2, (left_knee[1] + right_knee[1]) / 2)
                
                # 计算膝盖与髋部的垂直对齐度
                alignment_diff = abs(hip_center[0] - knee_center[0])
                stability_score = max(0, 2 - (alignment_diff / 30))
                score += stability_score
            
            # 4. 头部姿势评分 (0-2分)
            if "nose" in keypoints_dict:
                nose = keypoints_dict["nose"]
                
                # 计算头部与躯干中线的水平偏差
                head_alignment_diff = abs(nose[0] - torso_center_top[0])
                head_posture_score = max(0, 2 - (head_alignment_diff / 25))
                score += head_posture_score
            
            return score
            
        except Exception as e:
            print(f"计算姿态评分时出错: {str(e)}")
            return 0.0

    def calculate_teaching_metrics(self, behavior_counts, total_frames, fps, valid_frames):
        """
        计算教学评价指标
        
        Args:
            behavior_counts: 行为计数字典
            total_frames: 总帧数
            fps: 帧率
            valid_frames: 有效帧数
            
        Returns:
            dict: 包含各种教学指标的字典
        """
        # 计算每种行为的帧数占比
        behavior_proportions = {}
        for behavior, count in behavior_counts.items():
            behavior_proportions[behavior] = count / valid_frames if valid_frames > 0 else 0
            
        # 确保所有行为类型都有值
        for behavior in ["explaining", "writing", "moving", "interacting", "standing", "unknown", "not_in_frame"]:
            if behavior not in behavior_proportions:
                behavior_proportions[behavior] = 0
        
        # 教学活跃度 - 移动、互动和讲解的加权和
        teaching_activity = (
            behavior_proportions.get("moving", 0) * 1.0 + 
            behavior_proportions.get("interacting", 0) * 1.5 + 
            behavior_proportions.get("explaining", 0) * 0.8
        ) * 10
        
        # 教学节奏 - 基于行为变化频率
        # 理想值约为每20秒变化一次行为
        behavior_changes = self.behavior_change_count
        total_duration = total_frames / fps if fps > 0 else 0
        behavior_change_frequency = behavior_changes / total_duration if total_duration > 0 else 0
        
        # 节奏评分，基于变化频率与理想频率的接近程度
        ideal_frequency = 1/20  # 每20秒变化一次
        max_frequency = 1/5     # 每5秒变化一次已经太频繁
        min_frequency = 1/120   # 每120秒变化一次过于单调
        
        if behavior_change_frequency > max_frequency:
            # 太频繁，超过每5秒一次变化，分数降低
            rhythm_score = 7 - (behavior_change_frequency - max_frequency) * 50
        elif behavior_change_frequency < min_frequency:
            # 太少变化，低于每120秒一次，分数降低
            rhythm_score = 7 - (min_frequency - behavior_change_frequency) * 300
        else:
            # 在理想范围内，接近理想频率得分最高
            proximity = abs(behavior_change_frequency - ideal_frequency) / ideal_frequency
            rhythm_score = 10 - proximity * 4
        
        # 确保分数在0-10范围内
        teaching_rhythm = max(0, min(10, rhythm_score))
        
        # 教学多样性 - 主要行为的分布均衡程度
        teaching_behaviors = ["explaining", "writing", "moving", "interacting", "standing"]
        behavior_values = [behavior_proportions.get(b, 0) for b in teaching_behaviors]
        non_zero_behaviors = sum(1 for v in behavior_values if v > 0.05)
        
        # 计算教学多样性指标
        if non_zero_behaviors <= 1:
            teaching_diversity = 2  # 太单一
        else:
            # 计算香农熵作为多样性指标
            def shannon_entropy(props):
                return -sum(p * math.log2(p) if p > 0 else 0 for p in props)
            
            max_entropy = math.log2(len(teaching_behaviors))  # 最大可能熵
            actual_entropy = shannon_entropy(behavior_values)
            
            # 将熵转换为0-10分制
            teaching_diversity = (actual_entropy / max_entropy) * 10 if max_entropy > 0 else 0
        
        # 教学投入度 - 不在画面中的时间越少越好，站立和未知行为越少越好
        engagement_score = 10 - (
            behavior_proportions.get("not_in_frame", 0) * 8 +  # 不在画面中严重影响投入度
            behavior_proportions.get("unknown", 0) * 5 +       # 未知行为中等影响
            behavior_proportions.get("standing", 0) * 2        # 站立较小影响
        ) * 10
        
        teaching_engagement = max(0, min(10, engagement_score))
        
        # 教学效果预估 - 基于各项指标的加权平均
        weights = {
            "activity": 0.25,
            "rhythm": 0.25,
            "diversity": 0.2,
            "engagement": 0.3
        }
        
        overall_score = (
            teaching_activity * weights["activity"] +
            teaching_rhythm * weights["rhythm"] +
            teaching_diversity * weights["diversity"] +
            teaching_engagement * weights["engagement"]
        )
        
        # 教学模式识别
        # 基于行为占比识别教学风格
        teaching_style = "未识别"
        style_confidence = 0.0
        
        # 演讲型教师
        if (behavior_proportions["explaining"] > 0.6 and 
            behavior_proportions["writing"] < 0.2):
            teaching_style = "演讲型"
            style_confidence = behavior_proportions["explaining"] * 1.2
            
        # 板书型教师
        elif (behavior_proportions["writing"] > 0.4 and 
              behavior_proportions["explaining"] < 0.3):
            teaching_style = "板书型"
            style_confidence = behavior_proportions["writing"] * 1.2
            
        # 互动型教师
        elif (behavior_proportions["interacting"] > 0.3 and 
              behavior_proportions["moving"] > 0.2):
            teaching_style = "互动型"
            style_confidence = behavior_proportions["interacting"] * 1.5
            
        # 平衡型教师
        elif (behavior_proportions["explaining"] > 0.2 and 
              behavior_proportions["writing"] > 0.2 and
              behavior_proportions["interacting"] > 0.1 and
              non_zero_behaviors >= 3):
            teaching_style = "平衡型"
            style_confidence = 0.7 + 0.3 * (non_zero_behaviors / 5)
            
        # 活跃型教师
        elif behavior_proportions["moving"] > 0.4:
            teaching_style = "活跃型"
            style_confidence = behavior_proportions["moving"] * 1.3
        
        # 构建行为时间序列
        # 计算行为变化点和持续时间
        behavior_transitions = []
        behavior_durations = {}
        
        if len(self.behavior_history) > 0:
            current_behavior = self.behavior_history[0]
            start_frame = 0
            
            for i in range(1, len(self.behavior_history)):
                if self.behavior_history[i] != current_behavior:
                    # 记录转换
                    duration = (i - start_frame) / fps if fps > 0 else 0
                    behavior_transitions.append({
                        "from": current_behavior,
                        "to": self.behavior_history[i],
                        "start_time": start_frame / fps if fps > 0 else 0,
                        "duration": duration
                    })
                    
                    # 累计该行为的总时长
                    if current_behavior not in behavior_durations:
                        behavior_durations[current_behavior] = 0
                    behavior_durations[current_behavior] += duration
                    
                    # 更新当前状态
                    current_behavior = self.behavior_history[i]
                    start_frame = i
            
            # 处理最后一个行为
            duration = (len(self.behavior_history) - start_frame) / fps if fps > 0 else 0
            if current_behavior not in behavior_durations:
                behavior_durations[current_behavior] = 0
            behavior_durations[current_behavior] += duration
        
        # 获取最常见的行为转换模式
        transition_patterns = {}
        if len(behavior_transitions) > 0:
            for i in range(len(behavior_transitions)-1):
                pattern = f"{behavior_transitions[i]['from']}->{behavior_transitions[i]['to']}->{behavior_transitions[i+1]['to']}"
                transition_patterns[pattern] = transition_patterns.get(pattern, 0) + 1
        
        # 排序并提取前3个模式
        common_patterns = sorted(transition_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 返回完整指标
        metrics = {
            "teaching_activity": teaching_activity,
            "teaching_rhythm": teaching_rhythm,
            "teaching_diversity": teaching_diversity,
            "teaching_engagement": teaching_engagement,
            "overall_score": overall_score,
            "behavior_proportions": behavior_proportions,
            "behavior_changes": behavior_changes,
            "behavior_change_frequency": behavior_change_frequency,
            "teaching_style": teaching_style,
            "style_confidence": min(style_confidence, 1.0),
            "behavior_durations": behavior_durations,
            "common_transition_patterns": common_patterns,
            "total_valid_frames": valid_frames,
            "total_frames": total_frames,
            "fps": fps
        }
        
        return metrics

    def generate_teaching_suggestions(self, metrics, behavior_counts):
        """
        基于教学指标生成个性化教学建议
        
        Args:
            metrics: 教学评估指标
            behavior_counts: 行为计数
            
        Returns:
            list: 教学建议列表
        """
        suggestions = []
        
        # 提取关键指标
        activity = metrics.get("teaching_activity", 0)
        rhythm = metrics.get("teaching_rhythm", 0)
        diversity = metrics.get("teaching_diversity", 0)
        engagement = metrics.get("teaching_engagement", 0)
        behavior_proportions = metrics.get("behavior_proportions", {})
        teaching_style = metrics.get("teaching_style", "未识别")
        style_confidence = metrics.get("style_confidence", 0)
        
        # 1. 教学活跃度建议
        if activity < 4:
            suggestions.append("教学活跃度较低。建议增加课堂互动环节，适当走动以吸引学生注意力，用手势和表情增强表达力。")
        elif activity < 7:
            suggestions.append("教学活跃度中等。建议在关键概念讲解时增加互动和走动，可以考虑引入小组讨论或提问环节。")
        elif activity > 9:
            suggestions.append("教学活跃度非常高。注意过度活跃可能分散学生注意力，建议在讲解重要概念时适当放缓节奏，给学生思考时间。")
            
        # 2. 教学节奏建议
        if rhythm < 4:
            suggestions.append("教学节奏不够流畅。建议避免长时间停留在单一行为上，适时转换教学活动，例如从讲解切换到提问或板书。")
        elif rhythm < 7:
            suggestions.append("教学节奏基本合理，但可以进一步优化。注意观察学生反应，及时调整行为转换的频率。")
        elif rhythm > 9:
            suggestions.append("教学节奏把握极佳。请保持这种平衡的教学节奏，它有助于保持学生注意力并促进有效学习。")
            
        # 3. 教学多样性建议
        if diversity < 4:
            suggestions.append("教学行为较为单一。建议丰富教学手段，尝试结合讲解、板书、提问、小组活动等多种形式，使课堂更加生动。")
        elif diversity < 7:
            suggestions.append("教学多样性尚可，但仍有提升空间。可尝试引入更多元的教学活动，如演示实验、案例分析或角色扮演等。")
            
        # 4. 教学投入度建议
        if engagement < 5:
            suggestions.append("教学投入度有待提高。请避免过多的停顿或离开镜头，增加与学生的眼神交流和互动频率。")
        elif engagement > 8:
            suggestions.append("教学投入度很高，这对学生参与度有积极影响。建议继续保持这种高度专注的教学状态。")
            
        # 5. 基于行为占比的针对性建议
        explaining_pct = behavior_proportions.get("explaining", 0) * 100
        writing_pct = behavior_proportions.get("writing", 0) * 100
        interacting_pct = behavior_proportions.get("interacting", 0) * 100
        moving_pct = behavior_proportions.get("moving", 0) * 100
        standing_pct = behavior_proportions.get("standing", 0) * 100
        
        # 讲解行为建议
        if explaining_pct > 70:
            suggestions.append(f"讲解占比过高({explaining_pct:.1f}%)。建议减少单向讲解时间，增加学生参与环节，例如设计思考题或小组讨论。")
        elif explaining_pct < 20 and not (teaching_style == "板书型" and style_confidence > 0.7):
            suggestions.append(f"讲解占比偏低({explaining_pct:.1f}%)。口头讲解是传递知识的重要手段，建议适当增加清晰、有条理的讲解。")
            
        # 板书行为建议
        if writing_pct > 60:
            suggestions.append(f"板书占比过高({writing_pct:.1f}%)。过多板书可能导致学生被动接受，建议结合口头讲解和互动，提高板书效率。")
        elif writing_pct < 10 and not (teaching_style == "互动型" and style_confidence > 0.7):
            suggestions.append(f"板书占比偏低({writing_pct:.1f}%)。适当板书有助于知识结构化和重点突出，建议增加关键概念和框架的板书。")
            
        # 互动行为建议
        if interacting_pct < 5:
            suggestions.append(f"互动环节较少({interacting_pct:.1f}%)。建议增加师生互动，例如提问、讨论、点评等，促进学生主动思考。")
        elif interacting_pct > 40:
            suggestions.append(f"互动环节占比较高({interacting_pct:.1f}%)。高互动度很好，但确保互动高效并关注到所有学生，避免只关注小部分积极学生。")
            
        # 移动行为建议
        if moving_pct < 5:
            suggestions.append(f"课堂移动较少({moving_pct:.1f}%)。适当在教室内走动可以拉近与学生距离，建议增加走动频率，关注到每个角落。")
        elif moving_pct > 50:
            suggestions.append(f"移动占比过高({moving_pct:.1f}%)。过多走动可能分散注意力，建议在讲解重点内容时保持相对稳定，步伐放缓。")
            
        # 站立行为建议
        if standing_pct > 30:
            suggestions.append(f"静态站立时间较长({standing_pct:.1f}%)。长时间站立可能显得呆板，建议增加手势、表情变化，加入适当走动。")
            
        # 6. 基于教学风格的建议
        if teaching_style == "演讲型" and style_confidence > 0.6:
            suggestions.append("您的教学风格偏向演讲型。这种风格适合知识传授，但建议增加互动和视觉辅助，确保信息有效传达和接收。")
        elif teaching_style == "板书型" and style_confidence > 0.6:
            suggestions.append("您的教学风格偏向板书型。板书有助于知识梳理，建议关注书写效率和结构清晰度，加强口头解释以确保学生理解。")
        elif teaching_style == "互动型" and style_confidence > 0.6:
            suggestions.append("您的教学风格偏向互动型。互动有助于学生参与，建议确保互动的目的性和高效性，注意掌控课堂节奏。")
        elif teaching_style == "平衡型" and style_confidence > 0.6:
            suggestions.append("您展现出平衡型教学风格，能够综合运用多种教学手段。这是理想的教学模式，建议继续保持各教学环节的合理配比。")
        elif teaching_style == "活跃型" and style_confidence > 0.6:
            suggestions.append("您的教学风格偏向活跃型。活跃的课堂氛围有利于学生兴趣培养，但需确保活动与教学目标紧密相关，避免流于形式。")
            
        # 7. 个性化建议 - 基于整体评分
        overall_score = metrics.get("overall_score", 0)
        if overall_score < 5:
            suggestions.append(f"整体教学评分为{overall_score:.1f}分，有较大提升空间。建议关注教学多样性和节奏感，参考优秀教学案例，逐步调整教学方式。")
        elif overall_score < 7:
            suggestions.append(f"整体教学评分为{overall_score:.1f}分，处于中等水平。建议有针对性地改进薄弱环节，例如增加课堂互动或优化讲解结构。")
        elif overall_score < 9:
            suggestions.append(f"整体教学评分为{overall_score:.1f}分，教学效果良好。建议在现有基础上精益求精，关注细节如语言表达、板书布局、时间分配等。")
        else:
            suggestions.append(f"整体教学评分为{overall_score:.1f}分，教学效果优秀。建议尝试更多创新教学方法，进一步提升教学专业性。")
        
        return suggestions

    def create_evaluation_video(self, input_video_path, behavior_sequence, valid_frames, metrics=None, output_path=None):
        """
        Create evaluation video with behavior assessment markers
        
        Args:
            input_video_path: Input video path
            behavior_sequence: Behavior sequence dictionary, key is frame index, value is behavior type
            valid_frames: List of valid frames
            metrics: Teaching metrics dictionary
            output_path: Output video path
            
        Returns:
            bool: Whether successfully created
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video {input_video_path}")
                return False
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Debug info
            print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
            print(f"Behavior sequence contains {len(behavior_sequence)} entries")

            # Create info panel size - add right side info panel
            panel_width = 350
            new_width = width + panel_width
            
            # Create output video file
            if output_path is None:
                output_path = f"evaluated_{os.path.basename(input_video_path)}"
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, height))
            
            # Define behavior color mapping (BGR format)
            behavior_colors = {
                "explaining": (0, 255, 0),     # Green - Explaining
                "writing": (255, 0, 0),        # Blue - Writing
                "moving": (0, 0, 255),         # Red - Moving
                "interacting": (255, 255, 0),  # Cyan - Interacting
                "standing": (0, 255, 255),     # Yellow - Standing
                "not_in_frame": (128, 0, 128), # Purple - Not in frame
                "unknown": (128, 128, 128)     # Gray - Unknown
            }
            
            # Behavior type English names
            behavior_names = {
                "explaining": "Explaining",
                "writing": "Writing",
                "moving": "Moving",
                "interacting": "Interacting",
                "standing": "Standing",
                "not_in_frame": "Not in Frame",
                "unknown": "Unknown"
            }
            
            # Generate metrics text - compatible with old and new formats
            metrics_text = []
            metrics_values = {}
            
            if metrics:
                if isinstance(metrics, dict):
                    # Try to process different metric formats
                    if "teaching_activity" in metrics:
                        # Old format
                        metrics_text = [
                            ("Teaching Activity", metrics.get('teaching_activity', 0)),
                            ("Teaching Engagement", metrics.get('teaching_engagement', 0)),
                            ("Teaching Rhythm", metrics.get('teaching_rhythm', 0)),
                        ]
                        metrics_values = {
                            "Teaching Activity": metrics.get('teaching_activity', 0),
                            "Teaching Engagement": metrics.get('teaching_engagement', 0),
                            "Teaching Rhythm": metrics.get('teaching_rhythm', 0),
                        }
                    elif "verbal_orientation" in metrics:
                        # New format
                        metrics_text = [
                            ("Verbal Expression", metrics.get('verbal_orientation', 0)),
                            ("Blackboard Usage", metrics.get('board_usage', 0)),
                            ("Classroom Activity", metrics.get('classroom_mobility', 0)),
                            ("Student Interaction", metrics.get('student_interaction', 0)),
                        ]
                        metrics_values = {
                            "Verbal Expression": metrics.get('verbal_orientation', 0),
                            "Blackboard Usage": metrics.get('board_usage', 0),
                            "Classroom Activity": metrics.get('classroom_mobility', 0),
                            "Student Interaction": metrics.get('student_interaction', 0),
                        }
            
            # Collect behavior statistics directly from behavior_sequence
            behavior_stats = {}
            for _, behavior in behavior_sequence.items():
                if behavior:
                    behavior_stats[behavior] = behavior_stats.get(behavior, 0) + 1
            
            # Calculate percentages based on the number of frames with behavior data
            total_behavior_frames = sum(behavior_stats.values())
            if total_behavior_frames > 0:
                behavior_percentages = {b: (count/total_behavior_frames*100) for b, count in behavior_stats.items()}
            else:
                behavior_percentages = {}
                
            # Debug behavior statistics
            print("Behavior statistics from create_evaluation_video:")
            for behavior, count in behavior_stats.items():
                percentage = behavior_percentages.get(behavior, 0)
                print(f"  {behavior}: {count} frames ({percentage:.1f}%)")
            
            # For creating timeline
            duration_sec = frame_count / fps
            
            # Process video frames
            frame_idx = 0
            pbar = tqdm(total=frame_count, desc="Generating evaluation video")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Create canvas with info panel
                canvas = np.zeros((height, new_width, 3), dtype=np.uint8)
                # Left side for video frame
                canvas[:, :width] = frame
                # Right side for info panel with dark background
                canvas[:, width:] = (40, 40, 40)  # Dark gray background
                
                # Calculate time
                current_time = frame_idx / fps
                time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}.{int((current_time%1)*100):02d}"
                
                # Draw color legend for each behavior
                legend_y = 50
                cv2.putText(canvas, "Behavior Types:", (width + 10, legend_y-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                          
                # Only show behaviors that appear in the data
                sorted_behaviors = sorted(behavior_percentages.items(), key=lambda x: x[1], reverse=True)
                for i, (behavior, percentage) in enumerate(sorted_behaviors):
                    if behavior in behavior_colors:
                        y_pos = legend_y + i * 25
                        # Draw color square
                        color = behavior_colors.get(behavior, (128, 128, 128))
                        cv2.rectangle(canvas, (width + 10, y_pos), (width + 30, y_pos + 20), color, -1)
                        # Show behavior name and percentage
                        behavior_name = behavior_names.get(behavior, behavior)
                        cv2.putText(canvas, f"{behavior_name}: {percentage:.1f}%", 
                                    (width + 40, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (255, 255, 255), 1)
                
                # Draw current behavior
                current_behavior = behavior_sequence.get(frame_idx, "unknown")
                current_color = behavior_colors.get(current_behavior, (128, 128, 128))
                behavior_y = legend_y + (len(sorted_behaviors) or len(behavior_colors)) * 25 + 40
                
                cv2.putText(canvas, "Current Behavior:", (width + 10, behavior_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Use larger label to highlight current behavior
                cv2.rectangle(canvas, (width + 10, behavior_y + 10), (width + panel_width - 20, behavior_y + 50), 
                             current_color, -1)
                behavior_name = behavior_names.get(current_behavior, current_behavior)
                cv2.putText(canvas, behavior_name, (width + 20, behavior_y + 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Show time information
                time_y = behavior_y + 80
                cv2.putText(canvas, f"Time: {time_str}", (width + 10, time_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(canvas, f"Frame: {frame_idx}/{frame_count}", (width + 10, time_y + 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw timeline
                timeline_y = time_y + 60
                timeline_width = panel_width - 40
                cv2.rectangle(canvas, (width + 20, timeline_y), (width + 20 + timeline_width, timeline_y + 10), 
                             (100, 100, 100), -1)
                
                # Current position indicator
                position_x = width + 20 + int(timeline_width * (frame_idx / frame_count))
                cv2.rectangle(canvas, (position_x - 5, timeline_y - 5), 
                             (position_x + 5, timeline_y + 15), (0, 255, 255), -1)
                
                # Draw metrics
                metrics_y = timeline_y + 50
                cv2.putText(canvas, "Performance Metrics:", (width + 10, metrics_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                for i, (metric_name, metric_value) in enumerate(metrics_text):
                    y_pos = metrics_y + 30 + i * 25
                    # Draw score bar
                    bar_width = int(metric_value * (panel_width - 40) / 5)  # Assuming max score is 5
                    # Generate color gradient based on score (low=red -> high=green)
                    bar_color = (
                        int(max(0, 255 * (5 - metric_value) / 5)),  # B
                        int(max(0, 255 * metric_value / 5)),        # G
                        0                                           # R
                    )
                    cv2.rectangle(canvas, (width + 20, y_pos), (width + 20 + bar_width, y_pos + 15), 
                                 bar_color, -1)
                    cv2.rectangle(canvas, (width + 20, y_pos), (width + 20 + panel_width - 40, y_pos + 15), 
                                 (100, 100, 100), 1)
                    # Show metric name and value
                    cv2.putText(canvas, f"{metric_name}: {metric_value:.2f}", 
                              (width + 20, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 255, 255), 1)
                
                # If valid frame, green border
                if frame_idx in valid_frames:
                    # Add green border around original video
                    cv2.rectangle(canvas, (0, 0), (width-1, height-1), (0, 255, 0), 2)
                    status_text = "Analyzed Frame"
                    status_color = (0, 255, 0)  # Green
                else:
                    # Otherwise, orange border for skipped frames
                    cv2.rectangle(canvas, (0, 0), (width-1, height-1), (0, 165, 255), 2)
                    status_text = "Skipped Frame"
                    status_color = (0, 165, 255)  # Orange
                
                # Show frame status in right panel
                status_y = metrics_y + 30 + len(metrics_text) * 25 + 30
                cv2.putText(canvas, f"Status: {status_text}", (width + 10, status_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Add video title and analysis info
                title_y = 30
                video_name = os.path.basename(input_video_path).split('.')[0]
                cv2.putText(canvas, f"Video Analysis: {video_name}", (width + 10, title_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write to output video
                out.write(canvas)
                
                frame_idx += 1
                pbar.update(1)
            
            pbar.close()
            cap.release()
            out.release()
            
            print(f"\nEnhanced evaluation video saved as: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating evaluation video: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_behavior_sequence(self, video_path, fps=None):
        """从视频中提取行为序列"""
        behavior_sequence = []
        valid_frames = {}
        posture_details_sequence = []  # 保存上下身姿态得分序列
        
        # 检查是否已经有behavior_history
        if hasattr(self, 'behavior_history') and self.behavior_history:
            print(f"使用已有的{len(self.behavior_history)}个行为记录")
            behavior_sequence = self.behavior_history
            
            # 创建有效帧映射
            for i, behavior in enumerate(behavior_sequence):
                valid_frames[i] = behavior is not None
                
            # 检查是否有姿态详情记录
            if hasattr(self, 'posture_details_history') and self.posture_details_history:
                posture_details_sequence = self.posture_details_history
            else:
                # 无姿态详情历史，创建空记录
                posture_details_sequence = [None] * len(behavior_sequence)
                
            return behavior_sequence, valid_frames, posture_details_sequence
            
        # 如果没有现成的行为历史，进行视频分析
        print("没有现成行为历史，正在分析视频...")
        
        # 进行视频分析，获取帧数据
        basic_results = self.evaluate_video(video_path, batch_size=15)
        
        if not basic_results or basic_results.get("status") != "success":
            return None, None, None
            
        # 提取行为序列
        frames_data = basic_results.get("frames_data", [])
        
        # 初始化姿态详情历史记录
        self.posture_details_history = []
        
        for frame_idx, frame_data in enumerate(frames_data):
            if frame_data and frame_data.get("behavior"):
                behavior = frame_data.get("behavior")
                keypoints = frame_data.get("keypoints")
                behavior_sequence.append(behavior)
                valid_frames[frame_idx] = True
                
                # 保存上下身姿态得分数据（如果有）
                if "posture_details" in frame_data:
                    posture_details = frame_data["posture_details"]
                    posture_details_sequence.append(posture_details)
                    self.posture_details_history.append(posture_details)
                else:
                    posture_details_sequence.append(None)
                    self.posture_details_history.append(None)
            else:
                behavior_sequence.append(None)
                valid_frames[frame_idx] = False
                posture_details_sequence.append(None)
                self.posture_details_history.append(None)
        
        return behavior_sequence, valid_frames, posture_details_sequence

    def _process_frame(self, frame, prev_keypoints=None):
        """
        处理单帧并返回结果

        Args:
            frame: 输入帧
            prev_keypoints: 上一帧的关键点（可选）

        Returns:
            dict: 处理结果
        """
        try:
            # 处理帧
            keypoints, confidence, behavior = self.process_frame(frame)
            
            if keypoints is not None:
                # 获取姿态得分
                posture_score = self.calculate_posture_score(keypoints)
                
                # 返回结果
                return {
                    'status': 'success',
                    'keypoints': keypoints,
                    'behavior': behavior,
                    'confidence': confidence,
                    'posture_score': posture_score,
                    'posture_details': self.posture_details if hasattr(self, 'posture_details') else None
                }
            else:
                return {
                    'status': 'no_detection',
                    'keypoints': None,
                    'behavior': 'unknown',
                    'confidence': 0.0,
                    'posture_score': 0.0
                }
                
        except Exception as e:
            print(f"处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e),
                'keypoints': None,
                'behavior': 'unknown',
                'confidence': 0.0,
                'posture_score': 0.0
            }

    def identify_teacher(self, detections, previous_teacher=None):
        """
        从多个人物检测中确定哪一个是教师
        
        Args:
            detections: 检测到的所有人物信息列表，每个元素包含索引、区域、置信度和综合得分
            previous_teacher: 上一帧中确定的教师信息（可选）
            
        Returns:
            dict: 教师的检测信息
        """
        if not detections:
            return None
            
        # 如果只有一个人，直接返回
        if len(detections) == 1:
            return detections[0]
        
        # 初始化教师候选人评分
        teacher_scores = []
        
        for i, person in enumerate(detections):
            # 基础分数 - 使用之前计算的综合得分
            base_score = person["combined_score"]
            
            # 连续性分数 - 如果有上一帧的教师信息，计算位置变化
            continuity_score = 0
            if previous_teacher is not None:
                # 获取当前和上一帧的边界框信息
                try:
                    prev_idx = previous_teacher["index"]
                    # 如果索引相同，给予高连续性分数
                    if prev_idx == person["index"]:
                        continuity_score = 10000  # 非常高的连续性得分
                    # 否则计算面积和位置变化
                    else:
                        # 面积变化率
                        area_ratio = person["area"] / previous_teacher["area"] if previous_teacher["area"] > 0 else 0
                        # 面积变化惩罚 - 变化越大惩罚越多
                        area_penalty = abs(1 - area_ratio) * 5000
                        
                        # 计算最终连续性得分
                        continuity_score = 5000 - area_penalty
                except:
                    continuity_score = 0
            
            # 中心位置偏好 - 更偏向画面中心的人物
            # 这个分数已经在combined_score中考虑了
            
            # 大小偏好 - 更偏向画面中较大的人物
            # 这个分数已经在combined_score中考虑了
            
            # 计算最终得分
            final_score = base_score + continuity_score
            
            teacher_scores.append({
                "index": i,
                "person": person,
                "score": final_score
            })
        
        # 选择得分最高的人作为教师
        if teacher_scores:
            best_teacher = max(teacher_scores, key=lambda x: x["score"])
            return best_teacher["person"]
        
        # 如果无法确定，返回面积最大的人
        return max(detections, key=lambda x: x["area"])

    def calculate_pointing_score(self, keypoints_dict):
        """计算指点行为的得分"""
        score = 0.0
        
        # 获取关键点
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_elbow = keypoints_dict.get('left_elbow')
        right_elbow = keypoints_dict.get('right_elbow')
        left_wrist = keypoints_dict.get('left_wrist')
        right_wrist = keypoints_dict.get('right_wrist')
        
        # 检查是否有足够的关键点
        if (left_shoulder and left_elbow and left_wrist) or (right_shoulder and right_elbow and right_wrist):
            # 检查左臂是否在指点姿势
            if left_shoulder and left_elbow and left_wrist:
                # 手臂伸直 - 指点特征
                left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                if 140 < left_arm_angle <= 180:  # 手臂接近伸直
                    score += 0.4
                    
                # 手臂水平伸出 - 指点特征
                wrist_shoulder_angle = abs(left_wrist[1] - left_shoulder[1])
                if wrist_shoulder_angle < 30:  # 手与肩膀高度接近
                    score += 0.3
            
            # 检查右臂是否在指点姿势
            if right_shoulder and right_elbow and right_wrist:
                # 手臂伸直 - 指点特征
                right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                if 140 < right_arm_angle <= 180:  # 手臂接近伸直
                    score += 0.4
                    
                # 手臂水平伸出 - 指点特征
                wrist_shoulder_angle = abs(right_wrist[1] - right_shoulder[1])
                if wrist_shoulder_angle < 30:  # 手与肩膀高度接近
                    score += 0.3
        
        return min(1.0, score)  # 确保得分不超过1.0
    
    def calculate_raising_hand_score(self, keypoints_dict):
        """计算举手行为的得分"""
        score = 0.0
        
        # 获取关键点
        left_shoulder = keypoints_dict.get('left_shoulder')
        right_shoulder = keypoints_dict.get('right_shoulder')
        left_elbow = keypoints_dict.get('left_elbow')
        right_elbow = keypoints_dict.get('right_elbow')
        left_wrist = keypoints_dict.get('left_wrist')
        right_wrist = keypoints_dict.get('right_wrist')
        
        # 检查关键点存在性
        if (left_shoulder and left_elbow and left_wrist) or (right_shoulder and right_elbow and right_wrist):
            # 判断手是否明显高于肩膀 - 典型的举手特征
            if left_wrist and left_shoulder and left_wrist[1] < left_shoulder[1] - 50:
                score += 0.6  # 左手举高
            if right_wrist and right_shoulder and right_wrist[1] < right_shoulder[1] - 50:
                score += 0.6  # 右手举高
            
            # 判断手臂是否伸直 - 举手通常手臂较直
            if left_elbow and left_shoulder and left_wrist:
                left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                if 150 < left_arm_angle <= 180:  # 手臂接近伸直
                    score += 0.3
            
            if right_elbow and right_shoulder and right_wrist:
                right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                if 150 < right_arm_angle <= 180:  # 手臂接近伸直
                    score += 0.3
        
        return min(1.0, score)  # 确保得分不超过1.0

    def evaluate_video_adaptive(self, video_path, batch_size=10, initial_skip_frames=5, min_skip_frames=1, max_skip_frames=10, start_frame=0, max_frames=None, use_cache=True):
        """
        使用自适应跳帧策略评估视频中的教师行为
        
        Args:
            video_path (str): 视频文件路径
            batch_size (int): 批处理大小
            initial_skip_frames (int): 初始跳帧率
            min_skip_frames (int): 最小跳帧率
            max_skip_frames (int): 最大跳帧率
            start_frame (int): 从第几帧开始处理
            max_frames (int): 最多处理多少帧
            use_cache (bool): 是否使用缓存加速
            
        Returns:
            dict: 分析结果
        """
        # 重置状态
        self.behavior_history = []
        self.last_position = None
        
        results = {
            "status": "initialization",
            "message": "正在初始化自适应分析"
        }
        
        # 检查是否使用缓存 (对于自适应模式，生成特殊的缓存标识)
        if use_cache and self.enable_cache:
            # 生成缓存文件名
            cache_key = f"{os.path.basename(video_path)}_adaptive_{start_frame}_{max_frames}_{min_skip_frames}_{max_skip_frames}"
            cache_file = os.path.join(self.cache_dir, f"{hash(cache_key)}.pkl")
            
            # 尝试从缓存加载
            if os.path.exists(cache_file):
                try:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        cached_results = pickle.load(f)
                    
                    print(f"从缓存加载自适应分析结果: {cache_file}")
                    
                    # 恢复行为历史
                    if "frames_data" in cached_results:
                        self.behavior_history = [data.get("behavior") if data else None for data in cached_results["frames_data"]]
                    
                    # 记录缓存命中
                    cached_results["cache_info"] = {
                        "cache_hit": True,
                        "cache_file": cache_file,
                        "original_time": cached_results.get("performance", {}).get("total_time", 0),
                        "load_time": 0.01  # 近似值
                    }
                    
                    return cached_results
                except Exception as e:
                    print(f"加载缓存时出错: {e}, 将重新分析视频")
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "status": "error",
                    "message": f"无法打开视频: {video_path}"
                }
                
            # 获取视频基本信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 如果指定了起始帧，跳到指定位置
            if start_frame > 0:
                if start_frame >= frame_count:
                    return {
                        "status": "error",
                        "message": f"起始帧 {start_frame} 超出视频总帧数 {frame_count}"
                    }
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 计算实际要处理的帧数
            if max_frames is not None:
                actual_frame_count = min(frame_count - start_frame, max_frames)
            else:
                actual_frame_count = frame_count - start_frame
                
            print(f"自适应处理视频从第 {start_frame} 帧开始，总帧数: {actual_frame_count}，初始跳帧: {initial_skip_frames}")
            
            # 开始全局计时
            total_start_time = time.time()
            
            # 初始化进度条
            from tqdm import tqdm
            progress_bar = tqdm(total=actual_frame_count)
            
            # 初始化结果
            frames_data = []
            behavior_counts = {}
            processed_frames = 0
            valid_frames = 0
            total_confidence = 0.0
            total_posture_score = 0.0
            batch_times = []
            gpu_usages = []
            
            # 初始化行为状态机用于自适应调整
            behavior_state = None
            consecutive_same_behavior = 0
            behavior_change_frames = []
            
            # 初始化当前跳帧率
            current_skip_frames = initial_skip_frames
            
            # 读取视频帧 - 自适应跳帧处理
            frame_idx = start_frame
            processed_count = 0
            
            # 实际处理帧和每帧对应的原始帧索引的列表
            actual_processed_frames = []
            adaptive_frame_indices = []
            
            # 记录采样信息
            sampling_info = {
                "skip_frames_history": [],
                "behavior_changes": [],
                "stable_periods": []
            }

            while cap.isOpened() and (max_frames is None or processed_count < max_frames):
                # 读取当前帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每个自适应周期是否需要处理这一帧
                process_this_frame = (frame_idx % current_skip_frames == 0)
                
                if process_this_frame:
                    # 保存帧索引
                    adaptive_frame_indices.append(frame_idx)
                    
                    # 记录采样率历史
                    sampling_info["skip_frames_history"].append({
                        "frame": frame_idx,
                        "skip_rate": current_skip_frames
                    })
                    
                    # 处理帧
                    keypoints, confidence, behavior = self.process_frame(frame)
                    
                    # 创建帧数据
                    frame_data = {
                        'status': 'success' if keypoints else 'no_detection',
                        'keypoints': keypoints,
                        'behavior': behavior,
                        'confidence': confidence if keypoints else 0.0,
                        'posture_score': self.calculate_posture_score(keypoints) if keypoints else 0.0,
                        'frame_idx': frame_idx,
                        'skip_rate': current_skip_frames
                    }
                    
                    # 添加帧数据
                    frames_data.append(frame_data)
                    
                    # 更新统计信息
                    if keypoints:
                        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                        valid_frames += 1
                        total_confidence += confidence
                        total_posture_score += frame_data['posture_score']
                    
                    # 更新行为状态和连续计数
                    if behavior_state is None:
                        behavior_state = behavior
                    elif behavior != behavior_state:
                        # 记录行为变化
                        behavior_change_frames.append(frame_idx)
                        sampling_info["behavior_changes"].append({
                            "frame": frame_idx,
                            "from": behavior_state,
                            "to": behavior
                        })
                        
                        # 重置连续计数
                        consecutive_same_behavior = 0
                        
                        # 行为变化时降低跳帧率以捕获更多细节
                        new_skip_frames = max(min_skip_frames, current_skip_frames // 2)
                        print(f"行为从 {behavior_state} 变为 {behavior}，降低跳帧率: {current_skip_frames} -> {new_skip_frames}")
                        current_skip_frames = new_skip_frames
                        
                        # 更新行为状态
                        behavior_state = behavior
                    else:
                        consecutive_same_behavior += 1
                        
                        # 如果连续30帧相同行为，逐步增加跳帧率
                        if consecutive_same_behavior >= 30 and consecutive_same_behavior % 10 == 0:
                            if current_skip_frames < max_skip_frames:
                                old_skip_frames = current_skip_frames
                                current_skip_frames = min(max_skip_frames, current_skip_frames + 1)
                                if old_skip_frames != current_skip_frames:
                                    print(f"连续 {consecutive_same_behavior} 帧行为保持 {behavior}，增加跳帧率: {old_skip_frames} -> {current_skip_frames}")
                                    
                                    # 记录稳定期
                                    sampling_info["stable_periods"].append({
                                        "frame": frame_idx,
                                        "behavior": behavior,
                                        "duration_frames": consecutive_same_behavior,
                                        "new_skip_rate": current_skip_frames
                                    })
                    
                    # 更新处理计数
                    processed_count += 1
                    actual_processed_frames.append(frame)
                    
                # 更新进度条（每一帧都更新，无论是否处理）
                progress_bar.update(1)
                
                # 更新帧索引
                frame_idx += 1
                
                # 检查是否达到最大帧数
                if max_frames is not None and frame_idx >= start_frame + max_frames:
                    break
            
            # 关闭进度条和视频
            progress_bar.close()
            cap.release()
            
            # 计算总时间
            total_end_time = time.time()
            total_elapsed_time = total_end_time - total_start_time
            
            # 计算性能指标
            avg_confidence = total_confidence / valid_frames if valid_frames > 0 else 0
            avg_posture_score = total_posture_score / valid_frames if valid_frames > 0 else 0
            
            # 计算跳帧前后的估计节省时间 - 固定跳帧
            if initial_skip_frames > 1:
                estimated_full_frames = actual_frame_count
                actual_processed = len(actual_processed_frames)
                frames_saved = estimated_full_frames - actual_processed
                time_per_frame = total_elapsed_time / actual_processed if actual_processed > 0 else 0
                estimated_time_saved = frames_saved * time_per_frame
                
                print(f"\n使用自适应跳帧分析节省了约 {estimated_time_saved:.2f} 秒 (约 {estimated_time_saved/60:.2f} 分钟)")
                print(f"总共处理了 {actual_processed} 帧，跳过了 {frames_saved} 帧")
                print(f"平均跳帧率: {actual_frame_count / actual_processed:.2f}")
            
            # 返回最终结果
            results = {
                "status": "success",
                "behavior_counts": behavior_counts,
                "total_frames": frame_count,
                "start_frame": start_frame,
                "processed_frames": len(actual_processed_frames),
                "valid_frames": valid_frames,
                "adaptive_frame_indices": adaptive_frame_indices,
                "fps": fps,
                "avg_confidence": avg_confidence,
                "avg_posture_score": avg_posture_score,
                "frames_data": frames_data,
                "performance": {
                    "total_elapsed_time": total_elapsed_time,
                    "adaptive_sampling": {
                        "initial_skip_frames": initial_skip_frames,
                        "min_skip_frames": min_skip_frames,
                        "max_skip_frames": max_skip_frames,
                        "behavior_change_points": behavior_change_frames,
                        "adaptive_info": sampling_info
                    }
                },
                "cache_info": {
                    "cache_hit": False,
                    "cache_saved": True if use_cache and self.enable_cache else False
                }
            }
            
            # 设置最终状态
            self.behavior_history = [data.get("behavior") if data else None for data in frames_data]
            
            # 如果启用了缓存，保存结果
            if use_cache and self.enable_cache:
                try:
                    import pickle
                    with open(cache_file, 'wb') as f:
                        pickle.dump(results, f)
                    print(f"自适应分析结果已缓存: {cache_file}")
                    results["cache_info"]["cache_file"] = cache_file
                except Exception as e:
                    print(f"缓存结果时出错: {e}")
            
            return results
            
        except Exception as e:
            print(f"自适应视频评估出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": f"处理异常: {str(e)}"
            }