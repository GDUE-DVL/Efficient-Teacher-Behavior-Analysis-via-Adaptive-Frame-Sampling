#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import time
from datetime import datetime

# 配置matplotlib使用英文
import matplotlib
matplotlib.use('Agg')  # 设置后端
import matplotlib.pyplot as plt

# 使用英文作为绘图标签，避免中文乱码问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

from teacher_evaluation import TeacherEvaluator


def main():
    """主函数 - 使用自适应抽帧分析教师行为"""
    parser = argparse.ArgumentParser(description="自适应抽帧教师行为分析工具")
    
    # 必需参数
    parser.add_argument("video_path", help="输入视频文件路径")
    parser.add_argument("output_dir", nargs="?", default=None, help="可选的输出目录路径")
    
    # 可选参数
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小，同时处理的帧数（默认：10）")
    parser.add_argument("--initial_skip_frames", type=int, default=5, help="初始跳帧率（默认：5）")
    parser.add_argument("--min_skip_frames", type=int, default=1, help="最小跳帧率（默认：1）")
    parser.add_argument("--max_skip_frames", type=int, default=10, help="最大跳帧率（默认：10）")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID（默认：0）")
    parser.add_argument("--create_video", action="store_true", help="生成评价视频（默认：否）")
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧位置（默认：0）")
    parser.add_argument("--max_frames", type=int, default=None, help="最大处理帧数（默认：处理所有帧）")
    parser.add_argument("--time_analysis", action="store_true", help="生成时间分析图表（默认：是）")

    args = parser.parse_args()
    
    # 获取命令行参数
    video_path = args.video_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    initial_skip_frames = args.initial_skip_frames
    min_skip_frames = args.min_skip_frames
    max_skip_frames = args.max_skip_frames
    
    # 设置GPU设备
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"使用GPU: CUDA设备 {args.gpu_id}")
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件 {video_path} 不存在")
        sys.exit(1)
    
    # 如果没有指定输出目录，使用视频所在目录
    if output_dir is None:
        output_dir = os.path.dirname(video_path) or "."
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备分析
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    start_time = time.time()
    
    print(f"\n== 自适应抽帧分析视频 ==")
    print(f"视频: {video_path}")
    print(f"输出目录: {output_dir}")
    print(f"批处理大小: {batch_size}")
    print(f"初始跳帧率: {initial_skip_frames}")
    print(f"最小跳帧率: {min_skip_frames}")
    print(f"最大跳帧率: {max_skip_frames}")
    
    if args.start_frame > 0:
        print(f"起始帧: {args.start_frame}")
    if args.max_frames:
        print(f"最大处理帧数: {args.max_frames}")
    
    # 获取视频基本信息
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            sys.exit(1)
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {frame_count}, 时长: {duration/60:.2f}分钟")
        cap.release()
    except Exception as e:
        print(f"获取视频信息时出错: {e}")
        sys.exit(1)
    
    # 创建评估器实例
    evaluator = TeacherEvaluator()
    
    # 第一阶段: 行为分析 (使用自适应抽帧)
    print("\n第一阶段: 自适应抽帧行为分析...")
    
    # 准备自适应分析参数
    adaptive_params = {
        "batch_size": batch_size,
        "initial_skip_frames": initial_skip_frames,
        "min_skip_frames": min_skip_frames,
        "max_skip_frames": max_skip_frames,
        "use_cache": True  # 启用缓存加速
    }
    
    # 添加可选参数
    if args.start_frame > 0:
        adaptive_params["start_frame"] = args.start_frame
    
    if args.max_frames:
        adaptive_params["max_frames"] = args.max_frames
    
    # 执行自适应分析
    basic_results = evaluator.evaluate_video_adaptive(video_path, **adaptive_params)
    
    if not basic_results or basic_results.get("status") != "success":
        error_msg = basic_results.get("message", "未知错误") if basic_results else "分析失败"
        print(f"错误: {error_msg}")
        sys.exit(1)
    
    # 打印自适应抽帧统计信息
    if "performance" in basic_results and "adaptive_sampling" in basic_results["performance"]:
        adaptive_info = basic_results["performance"]["adaptive_sampling"]
        print("\n自适应抽帧统计:")
        print(f"- 初始跳帧率: {adaptive_info['initial_skip_frames']}")
        print(f"- 行为变化点数量: {len(adaptive_info.get('behavior_change_points', []))}")
        
        behavior_changes = adaptive_info.get("adaptive_info", {}).get("behavior_changes", [])
        if behavior_changes:
            print(f"- 行为变化详情 (前5个):")
            for i, change in enumerate(behavior_changes[:5]):
                print(f"  * 帧 {change['frame']}: {change['from']} -> {change['to']}")
    
    # 第二阶段: 生成综合报告
    print("\n第二阶段: 生成综合报告...")
    report_data = evaluator.generate_comprehensive_report(video_path, basic_results.get("fps"))
    
    # 保存报告JSON
    report_path = os.path.join(output_dir, f"{video_name}_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_info": {
                    "name": video_name,
                    "path": video_path,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration_seconds": duration
                },
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "basic_results": basic_results,
                "comprehensive_report": report_data,
                "adaptive_analysis": True
            }, f, ensure_ascii=False, indent=2)
        print(f"报告已保存到: {report_path}")
    except Exception as e:
        print(f"保存报告时出错: {e}")
    
    # 显示行为占比
    print("\n行为时间占比:")
    for behavior, proportion in report_data.get("behavior_time_proportions", {}).items():
        if proportion > 0.01:  # 只显示占比超过1%的行为
            minutes = proportion * duration / 60
            print(f"  {behavior}: {minutes:.2f}分钟 ({proportion*100:.1f}%)")
    
    # 显示姿态分析结果（如果有）
    if "posture_analysis" in report_data and isinstance(report_data["posture_analysis"], dict):
        posture_stats = report_data["posture_analysis"]
        if "error" not in posture_stats:
            print("\n姿态分析结果:")
            # 显示上下身一致性
            if "upper_lower_consistency" in posture_stats:
                consistency = posture_stats["upper_lower_consistency"]
                print(f"  上下身姿态一致性: {consistency*100:.1f}%")
            
            # 显示上半身站立得分
            if "upper_body_standing" in posture_stats:
                upper_stats = posture_stats["upper_body_standing"]
                print(f"  上半身站立得分: {upper_stats['avg']:.2f} (最小: {upper_stats['min']:.2f}, 最大: {upper_stats['max']:.2f})")
            
            # 显示下半身站立得分
            if "lower_body_standing" in posture_stats:
                lower_stats = posture_stats["lower_body_standing"]
                print(f"  下半身站立得分: {lower_stats['avg']:.2f} (最小: {lower_stats['min']:.2f}, 最大: {lower_stats['max']:.2f})")
            
            # 显示整体站立得分
            if "overall_standing" in posture_stats:
                overall_stats = posture_stats["overall_standing"]
                print(f"  整体站立得分: {overall_stats['avg']:.2f} (最小: {overall_stats['min']:.2f}, 最大: {overall_stats['max']:.2f})")
            
            # 显示坐姿得分
            if "sitting" in posture_stats:
                sitting_stats = posture_stats["sitting"]
                print(f"  坐姿得分: {sitting_stats['avg']:.2f} (最小: {sitting_stats['min']:.2f}, 最大: {sitting_stats['max']:.2f})")
    
    # 第三阶段: 生成时间分析图表
    if args.time_analysis:
        print("\nPhase 3: Generating time-based behavior analysis chart...")
        behavior_sequence = basic_results.get("frames_data", [])
        if behavior_sequence:
            # 提取行为序列
            behaviors = [data.get("behavior") for data in behavior_sequence if data]
            print(f"Found {len(behaviors)} valid behavior records for time analysis")
            if behaviors:
                try:
                    # 确保输出目录存在
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Output directory: {output_dir}")
                    
                    # 生成时间分析图表
                    time_chart_path = evaluator.generate_time_based_behavior_analysis(
                        behaviors, fps, video_path, output_dir
                    )
                    
                    if time_chart_path and os.path.exists(time_chart_path):
                        print(f"Time analysis chart generated: {time_chart_path}")
                    else:
                        print(f"Warning: Time analysis chart generation failed or file does not exist")
                except Exception as e:
                    print(f"Error generating time analysis chart: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Warning: No valid behavior records found, cannot generate time analysis chart")
        else:
            print("Warning: No frame data found, cannot generate time analysis chart")
    
    # 第四阶段: 创建评价视频（如果指定了）
    if args.create_video:
        print("\nPhase 4: Creating evaluation video with info overlay...")
        frames_data = basic_results.get("frames_data", [])
        behavior_data_dict = {f_data["frame_index"]: f_data for f_data in frames_data if f_data}
        
        video_output_path = os.path.join(output_dir, f"info_evaluated_{video_name}.mp4")
        
        # 调用新的视频创建函数
        create_info_video(video_path, video_output_path, behavior_data_dict, fps, width, height)
    
    # 完成
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n总运行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    return 0


# --- New Simplified Video Creation Function --- 
def create_info_video(input_video_path, output_path, behavior_data, fps, width, height):
    """Creates a video with an information overlay."""
    try:
        import cv2
        from tqdm import tqdm
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_video_path}")
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Panel settings
        panel_height = 60
        panel_color = (30, 30, 30)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1

        # Behavior mapping and colors
        behavior_en_map = {
            "explaining": "Explaining", "writing": "Writing", "moving": "Moving",
            "interacting": "Interacting", "standing": "Standing", "sitting": "Sitting",
            "pointing": "Pointing", "raising_hand": "Raising Hand", "unknown": "Unknown",
            "not_in_frame": "Not in Frame", None: "Processing..."
        }
        color_map = {
            "explaining": (0, 165, 255), "writing": (255, 0, 0), "moving": (0, 0, 255),
            "interacting": (0, 255, 255), "standing": (0, 255, 0), "sitting": (128, 0, 128),
            "pointing": (255, 255, 0), "raising_hand": (255, 192, 203), "unknown": (128, 128, 128),
            "not_in_frame": (64, 64, 64), None: (200, 200, 200)
        }

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Creating info video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw panel background
            cv2.rectangle(frame, (0, height - panel_height), (width, height), panel_color, -1)

            # Get behavior info
            b_info = behavior_data.get(frame_idx, {})
            behavior = b_info.get("behavior")
            confidence = b_info.get("confidence", 0.0)
            behavior_text = behavior_en_map.get(behavior, "N/A")
            behavior_color = color_map.get(behavior, (128, 128, 128))

            # Time
            current_time_sec = frame_idx / fps
            time_str = f"Time: {int(current_time_sec // 60):02d}:{int(current_time_sec % 60):02d}"
            cv2.putText(frame, time_str, (10, height - panel_height + 20), font, font_scale, text_color, font_thickness)

            # Behavior
            cv2.putText(frame, f"Behavior: ", (10, height - panel_height + 40), font, font_scale, text_color, font_thickness)
            cv2.putText(frame, behavior_text, (100, height - panel_height + 40), font, font_scale, behavior_color, font_thickness+1)

            # Confidence
            conf_str = f"Conf: {confidence:.2f}"
            cv2.putText(frame, conf_str, (width - 150, height - panel_height + 20), font, font_scale, text_color, font_thickness)
            
            # Frame number
            frame_str = f"Frame: {frame_idx}"
            cv2.putText(frame, frame_str, (width - 150, height - panel_height + 40), font, font_scale, text_color, font_thickness)


            out.write(frame)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()
        print(f"Info video saved to: {output_path}")

    except Exception as e:
        print(f"Error in create_info_video: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main()) 