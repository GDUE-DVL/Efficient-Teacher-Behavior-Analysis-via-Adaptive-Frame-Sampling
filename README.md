# YOLOv11-Pose Teacher Behavior Analysis SystemYOLOv11

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![条款:](https://img.shields.io/badge/License-MIT-yellow.svg)] (https://opensource.org/licenses/MIT)
[![Python 3.8+   Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)[! [OpenCV] (https://img.shields.io/badge/opencv - 4.5 -green.svg)] (https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)[! [PyTorch] (https://img.shields.io/badge/PyTorch-1.7 -red.svg)] (https://pytorch.org/)

An intelligent system for analyzing teacher behaviors in classroom videos using YOLOv11 pose estimation technology. This system can automatically identify, track, and evaluate various teaching activities including writing on board, explaining, interacting with students, moving around, and standing.基于YOLOv11姿态估计技术的课堂视频教师行为分析智能系统该系统可以自动识别、跟踪和评估各种教学活动，包括在黑板上写字、解释、与学生互动、走动和站立。

## 🎯 Features   ##🎯产品特点

### Core Analysis Capabilities核心分析能力
- **Pose Detection**: Real-time human pose estimation using YOLOv11 models- **姿态检测**：使用YOLOv11模型进行实时人体姿态估计
- **Behavior Recognition**: Automatic identification of 7 types of teaching behaviors:- **行为识别**：自动识别7种教学行为：
  - 📝 **Writing** (板书) - Writing on blackboard/whiteboard-📝**书写**()-黑板/白板书写
  - 💬 **Explaining** (讲解) - Lecturing and explaining content-💬**讲解**(0.001)-讲解内容
  - 🤝 **Interacting** (互动) - Interacting with students-🤝**互动** -与学生互动
  - 🚶 **Moving** (走动) - Walking around the classroom-🚶**移动** -在教室里走动
  - 🧍 **Standing** (站立) - Standing in place-🧍**站立** -原地站立
  - 👋 **Pointing** (指向) - Pointing gestures-👋**指向**（齐声）-指向手势
  - ✋ **Raising Hand** (举手) - Raising hand gestures-✋**举手**（英文）-举手示意

### Advanced Features   高级功能
- **Pose Smoothing**: Advanced smoothing algorithms to reduce detection noise- **姿态平滑**：先进的平滑算法，以减少检测噪声
- **Behavior State Machine**: Intelligent state management for behavior transitions- **行为状态机**：行为转换的智能状态管理
- **Adaptive Frame Sampling**: Dynamic frame rate adjustment for optimal performance- **自适应帧采样**：动态帧率调整的最佳性能
- **Comprehensive Analytics**: Detailed behavioral metrics and statistics-综合分析：详细的行为指标和统计数据
- **Visual Reports**: Rich visualizations and charts for analysis results- **可视化报告**：丰富的可视化和图表分析结果
- **Batch Processing**: Support for processing multiple videos simultaneously- **批处理**：支持同时处理多个视频
- **GPU Acceleration**: CUDA support for faster processing- **GPU加速**:CUDA支持更快的处理

### Output Formats   
- 📊 **Comprehensive Reports**: JSON format with detailed metrics-📊**综合报告**:JSON格式，详细指标
- 📈 **Enhanced Charts**: Beautiful matplotlib visualizations-📈**增强的图表**：美丽的matplotlib可视化
- 🎥 **Annotated Videos**: Output videos with pose and behavior annotations-🎥**注释视频**：输出视频与姿势和行为注释
- 📋 **Teaching Suggestions**: AI-generated recommendations for improvement-📋**教学建议**：人工智能生成的改进建议

## 📦 Installation   

### Prerequisites   
- Python 3.8+   - Python 3.8
- NVIDIA GPU (recommended for better performance)- NVIDIA GPU（推荐更好的性能）
- CUDA toolkit (for GPU acceleration)CUDA工具包（用于GPU加速）

### Setup Environment   

1. **Clone the repository**1. 
   ```bash   ”“bash
   git clone https://github.com/GDUE-DVL/yolov11pose_teacher_analysis.gitGit
   cd yolov11pose_teacher_analysis
   ```

2. **Create virtual environment**2. **创建虚拟环境
   ```bash   ”“bash
   python -m venv venv
   
   # On Windows   #在Windows上
   venv\Scripts\activate   venv \ \激活脚本
   
   # On Linux/macOS   #在Linux/macOS上
   source venv/bin/activate   源venv / bin /激活
   ```

3. **Install dependencies**3. * * * *安装依赖关系
   ```bash   ”“bash
   pip install -r requirements.txtPIP install -r requirements.txt
   ```

4. **Download pre-trained models**4. **下载预训练模型**
   
   You need to download YOLOv11 pose estimation models:您需要下载YOLOv11姿态估计模型：
   - `yolov8n-pose.pt` (lightweight, faster)- ' yolov8n-pose.pt '（轻量级，更快）
   - `yolov8x-pose.pt` (more accurate, slower)‘ yolov8x-pose.pt ’（更准确，更慢）
   
   Place these files in the project root directory.将这些文件放在项目根目录中。

## 🚀 Quick Start   ##🚀快速入门

### Single Video Analysis   单视频分析

```bash   ”“bash
python teacher_evaluation.py --video_path your_video.mp4 --model_path yolov8n-pose.pt——video_path your_video.mp4——model_path yolov8n-pose.pt
```

### Batch Processing   批处理

```bash   ”“bash
python analyze_batch.py /path/to/video/directory --output /path/to/outputPython analyze_batch.py /path/to/video/directory——output /path/to/output
```

### Adaptive Analysis (Recommended)自适应分析（推荐）

```bash   ”“bash
python analyze_teacher_adaptive.py your_video.mp4 --batch_size 10 --initial_skip_frames 5Python: analyze_teacher_adaptive.py your_video.mp4——batch_size 10——initial_skip_frames
```

### Enhanced Visualization增强的可视化

```bash   ”“bash
python enhanced_chart.py your_report.jsonPython enhanced_chart.py
```

## 📁 Project Structure   ##📁项目结构

```
yolov11pose_teacher_analysis/
├── 📄 teacher_evaluation.py      # Main analysis engine├──📄teacher_evaluation.py #主分析引擎
├── 📄 analyze_batch.py           # Batch processing script
├── 📄 analyze_teacher_adaptive.py # Adaptive frame sampling
├── 📄 fast_analyze.py            # Fast analysis with basic features
├── 📄 enhanced_chart.py          # Advanced visualization generator
├── 📄 SimpleHRNet.py             # Alternative HRNet model support
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file
├── 🏗️ models/                    # Model files (download separately)
│   ├── yolov8n-pose.pt
│   └── yolov8x-pose.pt
├── 📁 results/                   # Output directory (auto-created)
└── 📁 cache/                     # Cache directory (auto-created)
```

## 🛠️ Core Components

### 1. TeacherEvaluator Class
The main analysis engine with the following key methods:分析引擎主要有以下几个关键方法：

```python   ”“python
# Core functionality
- evaluate_video()                    # Main video analysis
- evaluate_video_adaptive()           # Adaptive frame sampling
- process_frame()                     # Single frame processing
- identify_teaching_behavior()        # Behavior classification
- generate_comprehensive_report()     # Report generation
```

### 2. PoseSmoother Class
Advanced pose smoothing using Savitzky-Golay filter:

```python
- update(person_id, points)          # Update pose data
- get_smoothed_points(person_id)     # Get smoothed coordinates
```

### 3. BehaviorStateMachine Class
Intelligent behavior state management:智能行为状态管理：

```python
- update(new_behavior, confidence)   # Update behavior state
- get_state()                        # Get current state
- reset()                            # Reset state machine
```

## 📊 Analysis Metrics

### Behavioral Metrics
- **Time Distribution**: Percentage of time spent on each behavior
- **Activity Level**: Overall movement and engagement metrics
- **Transition Analysis**: Behavior change patterns
- **Posture Analysis**: Upper/lower body coordination
- **Teaching Effectiveness**: Derived teaching quality indicators

### Performance Metrics
- **Processing Speed**: Frames per second analysis
- **Detection Confidence**: Average pose detection confidence
- **Stability**: Pose tracking stability metrics
- **Memory Usage**: GPU and RAM utilization

## 🎨 Visualization Features

The system generates comprehensive visualizations including:

- **Timeline Charts**: Behavior sequences over time
- **Distribution Pie Charts**: Behavior time proportions
- **Transition Heatmaps**: Behavior change patterns
- **Posture Analysis Plots**: Body coordination metrics
- **Performance Dashboards**: System performance metrics

## ⚙️ Configuration Options

### Model Selection
```bash
# Lightweight model (faster)
--model_path yolov8n-pose.pt

# High accuracy model (slower)
--model_path yolov8x-pose.pt
```

### Processing Parameters
```bash
# Frame sampling
--batch_size 10                    # Frames processed simultaneously
--initial_skip_frames 5            # Initial frame skip rate
--min_skip_frames 1                # Minimum skip rate
--max_skip_frames 10               # Maximum skip rate

# Analysis range-📊**综合报告**:JSON格式，详细指标
--start_frame 0                    # Start from specific frame
--max_frames 1000                  # Limit processing frames
```

### Output Options
```bash
--create_video                     # Generate annotated video
--time_analysis                    # Enable time-based analysis
--output_dir /path/to/output       # Custom output directory
```

## 🔧 Advanced Usage

### Custom Behavior Detection

You can extend the system by adding custom behavior detection logic:

```python
def calculate_custom_behavior_score(self, keypoints_dict):
    """Custom behavior detection implementation"""
    # Your custom logic here
    return score-📊**综合报告**:JSON格式，详细指标

# Add to TeacherEvaluator class
```

### Performance Optimization

For large-scale processing:-📊**综合报告**:JSON格式，详细指标

```python
# Enable caching-📊**综合报告**:JSON格式，详细指标
use_cache=True-📊**综合报告**:JSON格式，详细指标

# Adjust batch size based on GPU memory
batch_size=16  # Increase for better GPU utilization

# Use adaptive sampling
evaluate_video_adaptive()  # Instead of evaluate_video()
```


## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for pose estimation models
- [OpenCV](https://opencv.org/) for computer vision operations
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Matplotlib](https://matplotlib.org/) for visualization

## 📞 Support

For questions, issues, or contributions:
- 📧 Email: [gdue0921@163.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/yolov11pose_teacher_analysis/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/yolov11pose_teacher_analysis/discussions)

## 🔮 Future Roadmap

- [ ] Real-time analysis support
- [ ] Multi-teacher detection and tracking
- [ ] Integration with classroom management systems
- [ ] Mobile app for on-the-go analysis
- [ ] Cloud-based processing options
- [ ] Advanced AI coaching recommendations

---

**Made with ❤️ for educators and researchers** 
