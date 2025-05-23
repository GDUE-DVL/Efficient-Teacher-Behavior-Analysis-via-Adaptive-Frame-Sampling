# YOLOv11-Pose Teacher Behavior Analysis System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)

An intelligent system for analyzing teacher behaviors in classroom videos using YOLOv11 pose estimation technology. This system can automatically identify, track, and evaluate various teaching activities including writing on board, explaining, interacting with students, moving around, and standing.

## 🎯 Features

### Core Analysis Capabilities
- **Pose Detection**: Real-time human pose estimation using YOLOv11 models
- **Behavior Recognition**: Automatic identification of 7 types of teaching behaviors:
  - 📝 **Writing** (板书) - Writing on blackboard/whiteboard
  - 💬 **Explaining** (讲解) - Lecturing and explaining content
  - 🤝 **Interacting** (互动) - Interacting with students
  - 🚶 **Moving** (走动) - Walking around the classroom
  - 🧍 **Standing** (站立) - Standing in place
  - 👋 **Pointing** (指向) - Pointing gestures
  - ✋ **Raising Hand** (举手) - Raising hand gestures

### Advanced Features
- **Pose Smoothing**: Advanced smoothing algorithms to reduce detection noise
- **Behavior State Machine**: Intelligent state management for behavior transitions
- **Adaptive Frame Sampling**: Dynamic frame rate adjustment for optimal performance
- **Comprehensive Analytics**: Detailed behavioral metrics and statistics
- **Visual Reports**: Rich visualizations and charts for analysis results
- **Batch Processing**: Support for processing multiple videos simultaneously
- **GPU Acceleration**: CUDA support for faster processing

### Output Formats
- 📊 **Comprehensive Reports**: JSON format with detailed metrics
- 📈 **Enhanced Charts**: Beautiful matplotlib visualizations
- 🎥 **Annotated Videos**: Output videos with pose and behavior annotations
- 📋 **Teaching Suggestions**: AI-generated recommendations for improvement

## 📦 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for better performance)
- CUDA toolkit (for GPU acceleration)

### Setup Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolov11pose_teacher_analysis.git
   cd yolov11pose_teacher_analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**
   
   You need to download YOLOv11 pose estimation models:
   - `yolov8n-pose.pt` (lightweight, faster)
   - `yolov8x-pose.pt` (more accurate, slower)
   
   Place these files in the project root directory.

## 🚀 Quick Start

### Single Video Analysis

```bash
python teacher_evaluation.py --video_path your_video.mp4 --model_path yolov8n-pose.pt
```

### Batch Processing

```bash
python analyze_batch.py /path/to/video/directory --output /path/to/output
```

### Adaptive Analysis (Recommended)

```bash
python analyze_teacher_adaptive.py your_video.mp4 --batch_size 10 --initial_skip_frames 5
```

### Enhanced Visualization

```bash
python enhanced_chart.py your_report.json
```

## 📁 Project Structure

```
yolov11pose_teacher_analysis/
├── 📄 teacher_evaluation.py      # Main analysis engine
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
The main analysis engine with the following key methods:

```python
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
Intelligent behavior state management:

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

# Analysis range
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
    return score

# Add to TeacherEvaluator class
```

### Performance Optimization

For large-scale processing:

```python
# Enable caching
use_cache=True

# Adjust batch size based on GPU memory
batch_size=16  # Increase for better GPU utilization

# Use adaptive sampling
evaluate_video_adaptive()  # Instead of evaluate_video()
```

## 📈 Performance Benchmarks

| Model | Resolution | FPS | GPU Memory | Accuracy |
|-------|------------|-----|------------|----------|
| YOLOv8n-pose | 640x640 | ~30 | ~2GB | Good |
| YOLOv8x-pose | 640x640 | ~15 | ~4GB | Excellent |

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
- 📧 Email: [your-email@example.com]
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