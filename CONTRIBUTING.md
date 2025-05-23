# Contributing to YOLOv11-Pose Teacher Behavior Analysis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the YOLOv11-Pose Teacher Behavior Analysis system.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. **Search existing issues** to avoid duplicates
2. **Use a clear and descriptive title**
3. **Provide detailed information** about the problem
4. **Include steps to reproduce** the issue
5. **Specify your environment** (OS, Python version, GPU, etc.)

### Suggesting Enhancements

We welcome suggestions for new features and improvements:
1. **Check existing feature requests** first
2. **Clearly describe the enhancement** and its benefits
3. **Provide use cases** where applicable
4. **Consider backwards compatibility**

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

## 🏗️ Development Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended)
- Git

### Local Development

1. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/yolov11pose_teacher_analysis.git
   cd yolov11pose_teacher_analysis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run tests**:
   ```bash
   python -m pytest tests/
   ```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Naming**: snake_case for functions and variables, PascalCase for classes

### Code Formatting

We use `black` for code formatting:
```bash
black --line-length 100 your_file.py
```

### Type Hints

Please add type hints to all new functions:
```python
def analyze_behavior(keypoints: List[Tuple[float, float, float]]) -> Dict[str, float]:
    """Analyze behavior from keypoints."""
    pass
```

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain complex logic, not obvious code
- **README updates**: Update documentation for new features

Example docstring:
```python
def calculate_writing_score(self, keypoints_dict: Dict) -> float:
    """Calculate writing behavior confidence score.
    
    Args:
        keypoints_dict: Dictionary containing pose keypoints
        
    Returns:
        Confidence score between 0.0 and 1.0
        
    Raises:
        ValueError: If keypoints_dict is invalid
    """
    pass
```

## 🧪 Testing

### Writing Tests

- **Unit tests**: Test individual functions
- **Integration tests**: Test component interactions
- **Performance tests**: Test processing speed and memory usage

### Test Structure

```python
import pytest
from teacher_evaluation import TeacherEvaluator

class TestTeacherEvaluator:
    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = TeacherEvaluator()
    
    def test_behavior_detection(self):
        """Test behavior detection accuracy."""
        # Test implementation
        pass
    
    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        with pytest.raises(ValueError):
            self.evaluator.process_frame(None)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_teacher_evaluator.py

# Run with coverage
pytest --cov=teacher_evaluation tests/
```

## 📊 Performance Considerations

### Optimization Guidelines

1. **Memory Management**: Use `clear_gpu_memory()` appropriately
2. **Batch Processing**: Prefer batch operations over individual frame processing
3. **Caching**: Implement caching for expensive operations
4. **Profiling**: Profile code before and after optimizations

### Benchmarking

Before submitting performance improvements:
```bash
python benchmark/run_benchmarks.py --model yolov8n-pose.pt
```

## 🗂️ Project Structure

### Key Directories

```
yolov11pose_teacher_analysis/
├── teacher_evaluation.py        # Core analysis engine
├── analyze_batch.py            # Batch processing
├── analyze_teacher_adaptive.py # Adaptive analysis
├── enhanced_chart.py           # Visualization
├── tests/                      # Test files
├── docs/                       # Documentation
├── examples/                   # Usage examples
└── benchmarks/                 # Performance tests
```

### Adding New Behavior Detectors

To add a new behavior detection method:

1. **Add method to TeacherEvaluator**:
   ```python
   def calculate_new_behavior_score(self, keypoints_dict: Dict) -> float:
       """Calculate confidence score for new behavior."""
       pass
   ```

2. **Update behavior mapping**:
   ```python
   self.behavior_methods = {
       # ... existing behaviors
       "new_behavior": self.calculate_new_behavior_score
   }
   ```

3. **Add tests**:
   ```python
   def test_new_behavior_detection(self):
       """Test new behavior detection."""
       pass
   ```

4. **Update documentation**

## 🔍 Code Review Process

### Review Criteria

- **Functionality**: Does the code work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Performance**: Does it maintain or improve performance?
- **Style**: Does it follow our coding standards?

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and provide good coverage
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is acceptable
- [ ] Error handling is appropriate

## 🐛 Debugging

### Common Issues

1. **GPU Memory Errors**: Use smaller batch sizes or clear GPU memory
2. **Model Loading Issues**: Check model file paths and versions
3. **Video Format Issues**: Ensure OpenCV can read the video format

### Debug Mode

Enable debug mode for detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

evaluator = TeacherEvaluator(debug=True)
```

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### High Priority
- [ ] Real-time analysis implementation
- [ ] Multi-person detection and tracking
- [ ] Mobile/web interface development
- [ ] Advanced visualization features

### Medium Priority
- [ ] Additional behavior detection methods
- [ ] Performance optimizations
- [ ] Cloud integration
- [ ] Documentation improvements

### Low Priority
- [ ] GUI application
- [ ] Alternative model support
- [ ] Specialized analysis tools

## 🆘 Getting Help

- **Discord**: [Join our Discord server](https://discord.gg/your-server)
- **Email**: [maintainer@example.com](mailto:maintainer@example.com)
- **Issues**: [GitHub Issues](https://github.com/your-username/yolov11pose_teacher_analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/yolov11pose_teacher_analysis/discussions)

## 🏆 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Project documentation
- Annual contributor highlights

Thank you for contributing to the advancement of educational technology! 🎓✨ 