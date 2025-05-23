#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
from teacher_evaluation import TeacherEvaluator
import cv2
import json
from datetime import datetime

def analyze_video(evaluator, video_path, output_dir=None):
    """分析单个视频文件并生成报告"""
    print(f"\n开始分析视频: {video_path}")
    
    # 确保输出目录存在
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), "analysis_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 视频基本信息
    start_time = time.time()
    
    try:
        # 获取视频时长
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return {"status": "error", "message": "无法打开视频文件"}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {frame_count}, 时长: {duration/60:.2f}分钟")
    except Exception as e:
        print(f"获取视频信息时出错: {e}")
        return {"status": "error", "message": f"获取视频信息失败: {e}"}
    
    try:
        # 第一阶段：基本行为分析
        print("第一阶段：基本行为分析...")
        basic_results = evaluator.evaluate_video(video_path)
        
        if basic_results.get("status") != "success":
            print(f"分析失败: {basic_results.get('message', '未知错误')}")
            return basic_results
        
        # 第二阶段：生成综合报告
        print("第二阶段：生成综合报告...")
        report_data = evaluator.generate_comprehensive_report(video_path, basic_results.get("fps"))
        
        # 保存结果到JSON文件
        result_path = os.path.join(output_dir, f"{video_name}_report.json")
        with open(result_path, "w", encoding="utf-8") as f:
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
                "comprehensive_report": report_data
            }, f, ensure_ascii=False, indent=2)
        
        # 显示分析结果摘要
        elapsed_time = time.time() - start_time
        print(f"\n分析完成!")
        print(f"总用时: {elapsed_time:.1f}秒")
        print(f"结果已保存到: {result_path}")
        
        # 显示主要行为占比
        print("\n行为时间占比:")
        for behavior, proportion in report_data.get("behavior_time_proportions", {}).items():
            if proportion > 0.01:  # 只显示占比超过1%的行为
                print(f"  {behavior}: {proportion*100:.1f}%")
        
        return {
            "status": "success", 
            "message": "分析成功", 
            "result_path": result_path
        }
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"分析过程中出错: {e}"}

def process_directory(input_dir, output_dir=None, file_types=None):
    """处理目录中的所有视频文件"""
    if file_types is None:
        file_types = ['.mp4', '.avi', '.mov', '.mkv']
    
    # 创建评估器实例（只创建一次）
    evaluator = TeacherEvaluator()
    
    # 遍历目录
    video_files = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in file_types):
            video_files.append(file_path)
    
    if not video_files:
        print(f"警告: 在 {input_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件待处理")
    
    # 分析每个视频
    results = {}
    for i, video_path in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] 处理: {os.path.basename(video_path)}")
        result = analyze_video(evaluator, video_path, output_dir)
        results[video_path] = result
    
    # 汇总结果
    success_count = sum(1 for result in results.values() if result.get("status") == "success")
    print(f"\n批处理完成: 共 {len(video_files)} 个视频, {success_count} 个成功, {len(video_files) - success_count} 个失败")
    
    return results

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="批量分析视频中的教师行为")
    parser.add_argument("input", help="输入视频文件或目录路径")
    parser.add_argument("--output", "-o", help="输出目录路径")
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 路径 {args.input} 不存在")
        sys.exit(1)
    
    # 处理单个文件或目录
    if os.path.isfile(args.input):
        # 单个文件
        evaluator = TeacherEvaluator()
        analyze_video(evaluator, args.input, args.output)
    else:
        # 目录中的所有视频
        process_directory(args.input, args.output) 