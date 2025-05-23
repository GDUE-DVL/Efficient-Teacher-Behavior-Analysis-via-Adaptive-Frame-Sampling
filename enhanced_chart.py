#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
from matplotlib.patches import Rectangle, FancyBboxPatch
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta
import argparse
from collections import Counter
import matplotlib.patheffects as path_effects

def read_report(report_path):
    """读取分析报告文件"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_behavior_transitions(behavior_sequence):
    """计算行为转换统计"""
    transitions = []
    for i in range(1, len(behavior_sequence)):
        if behavior_sequence[i] != behavior_sequence[i-1]:
            transitions.append((behavior_sequence[i-1], behavior_sequence[i]))
    
    transition_counts = Counter(transitions)
    return transition_counts

def extract_behavior_segments(behavior_sequence, frame_indices, fps):
    """提取行为连续片段"""
    segments = []
    if not behavior_sequence:
        return segments
    
    current_behavior = behavior_sequence[0]
    start_idx = frame_indices[0]
    
    for i in range(1, len(behavior_sequence)):
        if behavior_sequence[i] != current_behavior:
            # 记录上一个行为片段
            duration = (frame_indices[i] - start_idx) / fps if fps else 0
            segments.append({
                'behavior': current_behavior,
                'start_frame': start_idx,
                'end_frame': frame_indices[i],
                'duration': duration
            })
            current_behavior = behavior_sequence[i]
            start_idx = frame_indices[i]
    
    # 记录最后一个行为片段
    duration = (frame_indices[-1] - start_idx) / fps if fps else 0
    segments.append({
        'behavior': current_behavior,
        'start_frame': start_idx,
        'end_frame': frame_indices[-1],
        'duration': duration
    })
    
    return segments

def format_time_mm_ss(seconds):
    """将秒数格式化为 MM:SS 格式"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def generate_enhanced_chart(report_file):
    """从报告文件生成增强的行为分析图表"""
    # 读取报告数据
    report_data = read_report(report_file)
    
    # 提取视频信息
    video_name = report_data['video_info']['name'] if 'video_info' in report_data else 'unknown'
    fps = report_data['video_info']['fps'] if 'video_info' in report_data else 30
    video_duration = report_data['video_info'].get('duration_seconds', 0) if 'video_info' in report_data else 0
    
    # 提取行为数据
    frames_data = []
    if 'basic_results' in report_data:
        frames_data = report_data['basic_results'].get('frames_data', [])
    
    # 构建行为序列
    behavior_sequence = []
    frame_indices = []
    confidences = []
    
    for frame in frames_data:
        if frame.get('status') != 'no_detection' and frame.get('behavior') is not None:
            behavior_sequence.append(frame.get('behavior', 'unknown'))
            frame_indices.append(frame.get('frame_idx', 0))
            confidences.append(frame.get('confidence', 0.0))
    
    if not behavior_sequence:
        print("警告: 未找到有效的行为序列")
        return None
    
    # 提取行为片段
    behavior_segments = extract_behavior_segments(behavior_sequence, frame_indices, fps)
    
    # 计算行为转换
    transitions = calculate_behavior_transitions(behavior_sequence)
    
    # 计算教学指标 (如果报告中存在)
    teaching_metrics = None
    if 'metrics' in report_data:
        teaching_metrics = report_data['metrics']
    elif 'teaching_metrics' in report_data:
        teaching_metrics = report_data['teaching_metrics']
    
    # 计算每种行为的占比和总时长
    behavior_counts = {}
    for behavior in behavior_sequence:
        if behavior:
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
            
    total_frames = len(behavior_sequence)
    behavior_percentages = {b: count/total_frames*100 for b, count in behavior_counts.items() if b}
    
    # 计算每种行为的总时长
    behavior_durations = {}
    for segment in behavior_segments:
        behavior = segment['behavior']
        duration = segment['duration']
        if behavior in behavior_durations:
            behavior_durations[behavior] += duration
        else:
            behavior_durations[behavior] = duration
    
    # 中文支持 - 使用系统可用的中文字体
    try:
        # 检查系统中是否有中文字体
        import matplotlib.font_manager as fm
        chinese_fonts = []
        
        # 尝试寻找常见中文字体
        for font_name in ['SimHei', 'STXihei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
            if any(font_name.lower() in f.lower() for f in fm.findSystemFonts()):
                chinese_fonts.append(font_name)
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial']
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法找到中文字体，图表上的中文可能无法正确显示")
    
    # 设置图表样式
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    
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
    
    # 中文标签
    behavior_labels = {
        "explaining": "讲解",
        "writing": "板书",
        "moving": "走动",
        "interacting": "互动",
        "standing": "站立",
        "not_in_frame": "不在画面中",
        "unknown": "未知"
    }
    
    # 转换为时间
    start_time = datetime(2023, 1, 1, 0, 0, 0)  # 使用虚拟起始时间
    times = [start_time + timedelta(seconds=idx/fps) for idx in frame_indices]
    
    # 创建图表 - 更大尺寸，更高分辨率
    fig = plt.figure(figsize=(18, 12), dpi=200, facecolor='white')
    
    # 添加主标题和副标题
    plt.suptitle(f"{video_name} - 教师行为分析", 
               fontsize=24, fontweight='bold', y=0.98)
    
    if video_duration > 0:
        plt.figtext(0.5, 0.945, 
                   f"视频时长: {format_time_mm_ss(video_duration)} | 分析帧数: {total_frames} | 采样率: {fps} FPS",
                   ha='center', fontsize=14, style='italic', color='#555555')
    
    # 创建网格布局
    gs = mpl.gridspec.GridSpec(3, 4, height_ratios=[1, 4, 2], width_ratios=[1, 1, 1, 1])
    
    # 1. 行为时间分布 (比例图)
    ax_timeline = plt.subplot(gs[1, :])
    
    # 美化背景
    ax_timeline.set_facecolor('#f8f9fa')
    ax_timeline.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
    
    # 设置时间轴范围
    if times:
        time_min = start_time
        time_max = times[-1] + timedelta(seconds=10)  # 添加一点额外空间
        ax_timeline.set_xlim(time_min, time_max)
    
    # 绘制行为片段
    unique_behaviors = sorted(set(behavior_sequence))
    behavior_values = {behavior: i for i, behavior in enumerate(unique_behaviors)}
    
    # 直接绘制带状行为时间线
    for segment in behavior_segments:
        behavior = segment['behavior']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        duration = segment['duration']
        
        if behavior in behavior_colors:
            y_pos = behavior_values[behavior]
            start_time_obj = start_time + timedelta(seconds=start_frame/fps)
            end_time_obj = start_time + timedelta(seconds=end_frame/fps)
            
            # 获取颜色
            base_color = behavior_colors[behavior]
            
            # 绘制行为条
            rect = Rectangle(
                (start_time_obj, y_pos-0.35),
                end_time_obj - start_time_obj,
                0.7,
                facecolor=base_color,
                alpha=0.8,
                edgecolor='white',
                linewidth=1.0,
                zorder=10
            )
            ax_timeline.add_patch(rect)
            
            # 对于较长的片段，添加标签
            min_duration_for_label = 3  # 秒
            if duration > min_duration_for_label:
                mid_time = start_time_obj + (end_time_obj - start_time_obj) / 2
                label_text = f"{behavior_labels.get(behavior, behavior)}"
                if duration > 10:  # 对于更长的片段，添加时长
                    label_text += f" ({format_time_mm_ss(duration)})"
                
                ax_timeline.text(
                    mid_time, y_pos,
                    label_text,
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white',
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='black')],
                    zorder=20
                )
    
    # 设置Y轴标签
    ax_timeline.set_yticks([i for i in range(len(unique_behaviors))])
    ax_timeline.set_yticklabels([behavior_labels.get(b, b) for b in unique_behaviors])
    
    # 设置X轴为时间格式
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    ax_timeline.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))  # 每1分钟一个主刻度
    ax_timeline.xaxis.set_minor_locator(mdates.SecondLocator(interval=10))  # 每10秒一个次要刻度
    
    # 添加网格线
    ax_timeline.grid(which='major', axis='x', linestyle='-', linewidth=0.8, alpha=0.3)
    ax_timeline.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.2)
    
    # 标题和标签
    ax_timeline.set_title("时间行为分析", fontsize=18, pad=15)
    ax_timeline.set_xlabel("时间 (分:秒)", fontsize=14, labelpad=10)
    ax_timeline.set_ylabel("行为类型", fontsize=14, labelpad=10)
    
    # 添加颜色图例
    handles = [plt.Rectangle((0,0),1,1, color=behavior_colors.get(b, '#999999')) for b in unique_behaviors]
    ax_timeline.legend(
        handles,
        [behavior_labels.get(b, b) for b in unique_behaviors],
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(6, len(unique_behaviors)),
        fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc'
    )
    
    # 2. 行为占比饼图
    ax_pie = plt.subplot(gs[2, 0:2])
    ax_pie.set_facecolor('#f8f9fa')
    
    # 计算显示数据
    behavior_total_time = sum(behavior_durations.values())
    pie_labels = []
    pie_sizes = []
    pie_colors = []
    
    for behavior in unique_behaviors:
        duration = behavior_durations.get(behavior, 0)
        percentage = (duration / behavior_total_time * 100) if behavior_total_time > 0 else 0
        
        pie_labels.append(f"{behavior_labels.get(behavior, behavior)}: {duration:.1f}秒 ({percentage:.1f}%)")
        pie_sizes.append(duration)
        pie_colors.append(behavior_colors.get(behavior, '#999999'))
    
    # 绘制饼图
    wedges, texts, autotexts = ax_pie.pie(
        pie_sizes,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        explode=[0.05] * len(unique_behaviors),
        shadow=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.0, 'antialiased': True}
    )
    
    # 美化饼图文字
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
    
    # 添加图例
    ax_pie.legend(
        wedges,
        pie_labels,
        title="行为时间分布",
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=11
    )
    
    ax_pie.set_title("行为时间占比", fontsize=16, pad=15)
    
    # 3. 行为转换和统计信息
    ax_stats = plt.subplot(gs[2, 2:4])
    ax_stats.set_facecolor('#f8f9fa')
    ax_stats.axis('off')
    
    # 生成统计信息文本
    stats_text = []
    stats_text.append("行为统计信息")
    stats_text.append("=" * 40)
    stats_text.append(f"总视频时长: {format_time_mm_ss(video_duration)}")
    stats_text.append(f"有效分析帧数: {total_frames}")
    stats_text.append(f"行为变化次数: {len(transitions)}")
    
    # 行为转换统计
    if transitions:
        stats_text.append("\n最常见行为转换:")
        
        # 按频率排序行为转换
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        # 取前5个最常见的转换
        for i, ((from_behavior, to_behavior), count) in enumerate(sorted_transitions[:5]):
            from_label = behavior_labels.get(from_behavior, from_behavior)
            to_label = behavior_labels.get(to_behavior, to_behavior)
            stats_text.append(f"{i+1}. {from_label} → {to_label}: {count}次")
    
    # 添加教学评估指标 (如果存在)
    if teaching_metrics:
        stats_text.append("\n教学评估指标:")
        
        # 添加核心指标
        for metric_name, display_name in [
            ("teaching_activity", "教学活跃度"),
            ("teaching_rhythm", "教学节奏"),
            ("teaching_diversity", "教学多样性"),
            ("teaching_engagement", "教学投入度"),
            ("overall_score", "总体评分")
        ]:
            if metric_name in teaching_metrics:
                value = teaching_metrics[metric_name]
                stats_text.append(f"{display_name}: {value:.1f}/10")
        
        # 添加教学风格
        if "teaching_style" in teaching_metrics:
            style = teaching_metrics["teaching_style"]
            stats_text.append(f"\n教学风格: {style}")
    
    # 显示统计信息
    ax_stats.text(
        0.05, 0.95,
        "\n".join(stats_text),
        va='top', ha='left',
        fontsize=12,
        transform=ax_stats.transAxes,
        bbox=dict(boxstyle="round,pad=1.0", 
                 facecolor='white',
                 edgecolor='#cccccc',
                 alpha=0.9)
    )
    
    # 4. 行为频率条形图
    ax_bars = plt.subplot(gs[0, 0:])
    ax_bars.set_facecolor('#f8f9fa')
    
    # 按频率排序行为
    sorted_behaviors = sorted(unique_behaviors, key=lambda b: behavior_counts.get(b, 0), reverse=True)
    
    # 绘制水平条形图
    bars = ax_bars.barh(
        range(len(sorted_behaviors)),
        [behavior_counts.get(b, 0) for b in sorted_behaviors],
        height=0.6,
        color=[behavior_colors.get(b, '#999999') for b in sorted_behaviors],
        edgecolor='white',
        linewidth=1.0,
        alpha=0.9
    )
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = (behavior_counts.get(sorted_behaviors[i], 0) / total_frames) * 100
        
        ax_bars.text(
            width + max([b.get_width() for b in bars]) * 0.02,
            bar.get_y() + bar.get_height()/2,
            f"{int(width)} ({percentage:.1f}%)",
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # 设置坐标轴
    ax_bars.set_yticks(range(len(sorted_behaviors)))
    ax_bars.set_yticklabels([behavior_labels.get(b, b) for b in sorted_behaviors])
    ax_bars.invert_yaxis()  # 让最高频率的行为显示在顶部
    
    ax_bars.set_title("行为频率分析", fontsize=16, pad=15)
    ax_bars.set_xlabel("出现次数", fontsize=12)
    
    # 美化轴
    ax_bars.spines['right'].set_visible(False)
    ax_bars.spines['top'].set_visible(False)
    ax_bars.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 紧凑布局
    plt.tight_layout(rect=[0, 0, 1, 0.93], pad=3.0)
    
    # 保存图表
    output_dir = os.path.dirname(report_file)
    output_path = os.path.join(output_dir, f"{video_name}_enhanced_analysis.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"增强版图表已保存至: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='从报告文件生成增强版行为分析图表')
    parser.add_argument('report_file', help='报告JSON文件路径')
    
    args = parser.parse_args()
    generate_enhanced_chart(args.report_file)

if __name__ == "__main__":
    main() 