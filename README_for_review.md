# YOLOv11-Pose Teacher Behavior Analysis - Review Package

This package contains the core code for the YOLOv11-Pose based teacher behavior analysis project.

## Project Overview

This project utilizes the YOLOv8 pose estimation model to analyze teacher behaviors in classroom videos. It extracts keypoints, processes them to identify specific actions (like writing on the board, walking, interacting with students), and generates reports and visualizations summarizing the teacher's activity patterns throughout the lesson.

## Files Included

*   **`teacher_evaluation.py`**: The main script for processing a single video file. It performs pose estimation, behavior analysis, and generates output video, charts, and JSON reports.
*   **`analyze_batch.py`**: Script for processing multiple videos in a batch.
*   **`analyze_teacher_adaptive.py`**: Script implementing adaptive sampling techniques for analysis.
*   **`SimpleHRNet.py`**: Contains code related to the SimpleHRNet model, possibly used as an alternative or comparison model. (Included for completeness if relevant to experiments).
*   **`requirements.txt`**: Lists the necessary Python dependencies.
*   **`README_for_review.md`**: This file.

*(Potentially other relevant scripts like `fast_analyze.py`, `enhanced_chart.py`, etc., if they are central to the reviewed experiments)*

## How to Run the Code

1.  **Set up Environment:**
    *   Ensure you have Python 3.8+ installed.
    *   Create a virtual environment (recommended):
        ```bash
        python -m venv venv
        source venv/bin/activate # On Linux/macOS
        # venv\\Scripts\\activate # On Windows
        ```
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Run Single Video Analysis:**
    *   **Model File:** This project requires a YOLO pose estimation model file (e.g., `yolov11pose.pt`, `yolov8n-pose.pt`). Please ensure you have the appropriate model file. It is *not* included in this archive due to its size.
    *   Place the video file you want to analyze (e.g., `shifanke1.mp4`) in the same directory or provide the full path.
    *   Execute the main script, providing the path to your video and the chosen model file via the `--video_path` and `--model_path` arguments respectively:
        ```bash
        python teacher_evaluation.py --video_path path/to/your/video.mp4 --model_path path/to/your_model.pt
        ```
        For example, using `yolov11pose.pt`:
        ```bash
        python teacher_evaluation.py --video_path path/to/your/video.mp4 --model_path path/to/yolov11pose.pt
        ```
    *   Outputs (video, chart, report) will be generated in the same directory or a specified output directory.

3.  **Run Batch Analysis (if applicable):**
    *   Modify `analyze_batch.py` to specify the input directory containing videos.
    *   Run the script:
        ```bash
        python analyze_batch.py
        ```

4.  **Run Adaptive Sampling Analysis (if applicable):**
    *   Consult `analyze_teacher_adaptive.py` and `run_adaptive.sh` for usage instructions. Requires specific setup for adaptive parameters.

## Notes for Reviewers

*   The core logic for behavior state detection and analysis is within `teacher_evaluation.py`.
*   **Model files (`.pt`) are required but not included in this package.** Please obtain the necessary model weights (e.g., `yolov11pose.pt`) and specify the path using the `--model_path` argument when running the scripts.
*   The `results/` and `cache/` directories (not included in the zip) store intermediate and final outputs during runs.
*   Please refer to the original paper/report for details on the specific algorithms and metrics used.

This streamlined package provides the essential code components to understand and replicate the main experimental results. 