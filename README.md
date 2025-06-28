# Eye Gaze Tracking Project

This project implements an eye gaze tracking system using computer vision techniques. It appears to be part of a final year project focused on analyzing the impact of web camera parameters and calibration options on eye gaze tracking accuracy.

## Project Structure

- `main.py` - Main application file
- `requirements.txt` - Python dependencies
- `experiment_data/` - Directory containing experimental data
  - `participant_1/` - Data for participant 1
    - `experiment_state.json` - Experiment state and gaze data
- `shape_predictor_68_face_landmarks.dat` - Face landmark detection model (excluded from git)
- Various PDF documents related to the research project

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the face landmark detection model file (`shape_predictor_68_face_landmarks.dat`) and place it in the project root

## Usage

Run the main application:
```bash
python main.py
```

## Research Context

This project is part of research on "Experimental Analysis of the Impact of Web Camera Parameters and Calibration Options on Eye Gaze Tracking Accuracy" as indicated by the included documentation.

## Note

Large files (PDFs, .dat files) and the virtual environment are excluded from version control via `.gitignore`.