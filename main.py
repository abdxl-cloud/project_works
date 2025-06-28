import cv2
import dlib
import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime
import math

class GazeTrackingExperiment:
    def __init__(self, participant_id, base_output_dir="experiment_data"):
        self.participant_id = participant_id
        self.base_output_dir = base_output_dir
        
        # Create participant-specific directory
        self.participant_dir = os.path.join(base_output_dir, f"participant_{participant_id}")
        os.makedirs(self.participant_dir, exist_ok=True)
        
        # Initialize face detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Generate the 32 predefined test conditions
        self.test_conditions = self._generate_test_conditions()
        
        # Data storage
        self.experiment_data = []
        self.calibration_data = {}
        
        # Screen dimensions (adjust to your monitor)
        self.screen_width = 1920
        self.screen_height = 1080
        
        # State management for pause/resume (participant-specific)
        self.state_file = os.path.join(self.participant_dir, f"experiment_state.json")
        self.completed_conditions = set()
        self.total_conditions = len(self.test_conditions)
        self.current_condition_index = 0
        
        # Participant metadata
        self.participant_metadata = {
            'participant_id': participant_id,
            'experiment_start_time': datetime.now().isoformat(),
            'total_test_conditions': self.total_conditions,
            'system_info': self._get_system_info()
        }

    def _generate_test_conditions(self):
        """Generate test conditions based on actual camera capabilities"""
        try:
            # Get camera capabilities
            capabilities = self.get_camera_capabilities()
            max_width, max_height = capabilities['max_resolution']
            max_fps = capabilities['max_fps']
            autofocus_supported = capabilities['autofocus_supported']
            
            print(f"Detected camera capabilities:")
            print(f"Max Resolution: {max_width}x{max_height}")
            print(f"Max Frame Rate: {max_fps:.1f}fps")
            print(f"Autofocus Supported: {'Yes' if autofocus_supported else 'No'}")
            
            # Generate valid resolution options (only downscaling from max)
            resolutions = [None]  # Always include camera default
            
            # Add common downscaled resolutions that fit within camera's max
            common_resolutions = [
                (1920, 1080),  # 1080p
                (1280, 720),   # 720p
                (960, 540),    # qHD
                (640, 480),    # VGA
                (320, 240)     # QVGA
            ]
            
            for res_w, res_h in common_resolutions:
                if res_w <= max_width and res_h <= max_height:
                    resolutions.append((res_w, res_h))
            
            # Also add half and quarter of max resolution
            if max_width >= 640 and max_height >= 480:
                half_res = (max_width // 2, max_height // 2)
                if half_res not in resolutions:
                    resolutions.append(half_res)
            
            if max_width >= 1280 and max_height >= 960:
                quarter_res = (max_width // 4, max_height // 4)
                if quarter_res not in resolutions:
                    resolutions.append(quarter_res)
            
            # Generate valid frame rate options (only up to camera's max)
            frame_rates = [None]  # Always include camera default
            common_fps = [15, 24, 30, 60, 120]
            
            for fps in common_fps:
                if fps <= max_fps:
                    frame_rates.append(fps)
            
            # Add half and quarter of max fps
            if max_fps >= 30:
                half_fps = max_fps // 2
                if half_fps >= 10 and half_fps not in frame_rates:
                    frame_rates.append(half_fps)
            
            if max_fps >= 60:
                quarter_fps = max_fps // 4
                if quarter_fps >= 10 and quarter_fps not in frame_rates:
                    frame_rates.append(quarter_fps)
            
            # Autofocus options based on camera support
            if autofocus_supported:
                autofocus_options = [None, True, False]  # Default, On, Off
            else:
                autofocus_options = [None]  # Only default (which will be off)
            
            # Calibration variations
            calibration_points = [5, 9, 16]
            patterns = ["grid", "radial", "adaptive"]
            
            # Generate test conditions
            test_conditions = []
            test_id = 1
            
            for resolution in resolutions:
                for fps in frame_rates:
                    for autofocus in autofocus_options:
                        for points in calibration_points:
                            for pattern in patterns:
                                condition = {
                                    'test_id': f"T{test_id:02d}",
                                    'resolution': resolution,
                                    'frame_rate': fps,
                                    'autofocus': autofocus,
                                    'calibration_points': points,
                                    'pattern': pattern
                                }
                                test_conditions.append(condition)
                                test_id += 1
            
            print(f"Generated {len(test_conditions)} test conditions based on camera capabilities")
            print(f"Resolution options: {len(resolutions)}")
            print(f"Frame rate options: {len(frame_rates)}")
            print(f"Autofocus options: {len(autofocus_options)}")
            
        except Exception as e:
            print(f"Warning: Could not detect camera capabilities, using fallback values: {e}")
            # Fallback to a simplified set of conditions
            test_conditions = self._generate_fallback_conditions()
        
        return test_conditions
    
    def _generate_fallback_conditions(self):
        """Generate fallback test conditions when camera detection fails"""
        print("Camera detection failed, using conservative fallback configuration")
        
        # Conservative approach: use only camera defaults and basic downscaled options
        # This avoids trying to set parameters that might not be supported
        fallback_resolutions = [
            None,         # Camera default (safest option)
            (640, 480),   # VGA (widely supported)
            (320, 240)    # QVGA (very conservative)
        ]
        
        fallback_fps = [None, 30, 15]  # Default + conservative frame rates
        autofocus_options = [None]     # Only use camera default to avoid issues
        calibration_points = [5, 9, 16]
        patterns = ["grid", "radial", "adaptive"]
        
        test_conditions = []
        test_id = 1
        
        # Generate conservative combinations
        for resolution in fallback_resolutions:
            for fps in fallback_fps:
                for autofocus in autofocus_options:
                    for points in calibration_points:
                        for pattern in patterns:
                            condition = {
                                'test_id': f"T{test_id:02d}",
                                'resolution': resolution,
                                'frame_rate': fps,
                                'autofocus': autofocus,
                                'calibration_points': points,
                                'pattern': pattern
                            }
                            test_conditions.append(condition)
                            test_id += 1
        
        print(f"Generated {len(test_conditions)} conservative fallback test conditions")
        print("Note: Using only camera defaults and basic resolutions to ensure compatibility")
        return test_conditions
    
    def _parse_resolution(self, res_str):
        """Convert resolution string to tuple"""
        resolution_map = {
            '360p': (640, 360),
            '480p': (640, 480),
            '640p': (640, 480),  # Standard VGA
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '1440p': (2560, 1440),
            '4K': (3840, 2160)
        }
        return resolution_map.get(res_str, (640, 480))

    def _get_system_info(self):
        """Collect system information for reproducibility"""
        import platform
        return {
            'python_version': platform.python_version(),
            'system': platform.system(),
            'platform': platform.platform(),
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__
        }
    
    @staticmethod
    def list_participants(base_dir="experiment_data"):
        """List all participants with their progress"""
        if not os.path.exists(base_dir):
            return []
        
        participants = []
        for item in os.listdir(base_dir):
            if item.startswith("participant_"):
                participant_id = item.replace("participant_", "")
                participant_dir = os.path.join(base_dir, item)
                
                # Check for state file to get progress info
                state_file = os.path.join(participant_dir, "experiment_state.json")
                status = "Not started"
                progress = 0
                
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        completed = len(state.get('completed_conditions', []))
                        total = state.get('total_conditions', 0)
                        if total > 0:
                            progress = (completed / total) * 100
                            if completed == total:
                                status = "Completed"
                            else:
                                status = f"In progress ({completed}/{total})"
                    except:
                        status = "Error reading state"
                
                # Check for final results
                result_files = [f for f in os.listdir(participant_dir) 
                               if f.startswith("experiment_results_") and f.endswith(".csv")]
                if result_files and status != "In progress":
                    status = "Completed"
                    progress = 100
                
                participants.append({
                    'id': participant_id,
                    'status': status,
                    'progress': progress,
                    'directory': participant_dir
                })
        
        return sorted(participants, key=lambda x: x['id'])
    
    def get_participant_summary(self):
        """Get summary of current participant's progress"""
        return {
            'participant_id': self.participant_id,
            'completed_conditions': len(self.completed_conditions),
            'total_conditions': self.total_conditions,
            'progress_percentage': (len(self.completed_conditions) / self.total_conditions * 100) if self.total_conditions > 0 else 0,
            'participant_directory': self.participant_dir
        }
        
    def get_camera_capabilities(self):
        """Detect actual camera capabilities and supported configurations"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not open camera to detect capabilities")
        
        # Get default parameters first
        default_params = self.get_camera_parameters(cap)
        
        # Validate that we got reasonable parameters
        if default_params['resolution'][0] <= 0 or default_params['resolution'][1] <= 0:
            cap.release()
            raise RuntimeError("Camera returned invalid resolution parameters")
        
        max_width = int(default_params['resolution'][0])
        max_height = int(default_params['resolution'][1])
        max_fps = default_params['frame_rate']
        
        # Test if autofocus is actually supported by trying to set it
        autofocus_supported = False
        if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
            # Try to set autofocus and see if it actually changes
            original_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            new_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            autofocus_supported = (new_af != original_af) or (new_af == 1)
            # Restore original setting
            cap.set(cv2.CAP_PROP_AUTOFOCUS, original_af)
        
        cap.release()
        
        capabilities = {
            'max_resolution': (max_width, max_height),
            'max_fps': max_fps,
            'autofocus_supported': autofocus_supported,
            'default_params': default_params
        }
        
        return capabilities
    
    def get_default_camera_parameters(self):
        """Get the default camera parameters without setting anything"""
        capabilities = self.get_camera_capabilities()
        return capabilities['default_params']
    
    def setup_camera(self, resolution=None, frame_rate=None, autofocus=None):
        """Configure camera with specified parameters or use defaults"""
        cap = cv2.VideoCapture(0)
        
        # If no parameters specified, use defaults
        if resolution is None and frame_rate is None and autofocus is None:
            return cap
        
        # Set resolution if specified
        if resolution is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Set frame rate if specified
        if frame_rate is not None:
            cap.set(cv2.CAP_PROP_FPS, frame_rate)
        
        # Set autofocus if specified (if supported by camera)
        if autofocus is not None and hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        
        return cap
        
    def get_camera_parameters(self, cap):
        """Get actual camera parameters directly from the camera"""
        # Get actual resolution
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Get actual frame rate
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get actual autofocus setting (if supported)
        actual_autofocus = False
        if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
            autofocus_val = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            actual_autofocus = bool(autofocus_val)
        
        return {
            'resolution': (int(actual_width), int(actual_height)),
            'frame_rate': actual_fps,
            'autofocus': actual_autofocus
        }
    
    def generate_calibration_points(self, num_points, pattern, screen_w, screen_h):
        """Generate calibration points based on pattern and count"""
        points = []
        margin = 100  # Pixels from edge
        
        if pattern == 'grid':
            if num_points == 5:
                # 5-point: corners + center
                points = [
                    (margin, margin),
                    (screen_w - margin, margin),
                    (screen_w // 2, screen_h // 2),
                    (margin, screen_h - margin),
                    (screen_w - margin, screen_h - margin)
                ]
            elif num_points == 9:
                # 3x3 grid
                for i in range(3):
                    for j in range(3):
                        x = margin + j * (screen_w - 2 * margin) // 2
                        y = margin + i * (screen_h - 2 * margin) // 2
                        points.append((x, y))
            elif num_points == 16:
                # 4x4 grid
                for i in range(4):
                    for j in range(4):
                        x = margin + j * (screen_w - 2 * margin) // 3
                        y = margin + i * (screen_h - 2 * margin) // 3
                        points.append((x, y))
        
        elif pattern == 'radial':
            center_x, center_y = screen_w // 2, screen_h // 2
            max_radius = min((screen_w - 2 * margin) // 2, (screen_h - 2 * margin) // 2)
            
            # Always include center point
            points.append((center_x, center_y))
            
            if num_points > 1:
                if num_points <= 9:
                    # Single ring around center
                    for i in range(num_points - 1):
                        angle = 2 * math.pi * i / (num_points - 1)
                        radius = max_radius * 0.7  # 70% of max radius
                        x = int(center_x + radius * math.cos(angle))
                        y = int(center_y + radius * math.sin(angle))
                        points.append((x, y))
                else:
                    # Multiple concentric rings
                    remaining_points = num_points - 1
                    num_rings = min(3, remaining_points // 4)  # Up to 3 rings
                    
                    for ring in range(num_rings):
                        ring_radius = max_radius * (0.4 + 0.3 * ring)  # 40%, 70%, 100%
                        points_in_ring = remaining_points // num_rings
                        if ring < remaining_points % num_rings:
                            points_in_ring += 1
                        
                        for i in range(points_in_ring):
                            angle = 2 * math.pi * i / points_in_ring
                            x = int(center_x + ring_radius * math.cos(angle))
                            y = int(center_y + ring_radius * math.sin(angle))
                            points.append((x, y))
                        
                        remaining_points -= points_in_ring
        
        elif pattern == 'adaptive':
            # Adaptive calibration starts with basic points and adds more based on error
            # For initial implementation, we'll simulate this with a smart distribution
            points = self._generate_adaptive_points(num_points, screen_w, screen_h, margin)
        
        return points
    
    def _generate_adaptive_points(self, num_points, screen_w, screen_h, margin):
        """Generate adaptive calibration points that focus on problematic areas"""
        points = []
        
        # Start with essential points (corners + center)
        essential_points = [
            (screen_w // 2, screen_h // 2),  # Center (most important)
            (margin, margin),  # Top-left
            (screen_w - margin, margin),  # Top-right
            (margin, screen_h - margin),  # Bottom-left
            (screen_w - margin, screen_h - margin),  # Bottom-right
        ]
        
        # Add essential points up to the requested number
        for i, point in enumerate(essential_points):
            if i < num_points:
                points.append(point)
        
        # If we need more points, add them in areas typically problematic for gaze tracking
        if num_points > len(essential_points):
            # Problem areas: screen edges and intermediate positions
            additional_areas = [
                (screen_w // 2, margin),  # Top center
                (screen_w // 2, screen_h - margin),  # Bottom center
                (margin, screen_h // 2),  # Left center
                (screen_w - margin, screen_h // 2),  # Right center
                (screen_w // 4, screen_h // 4),  # Upper left quadrant
                (3 * screen_w // 4, screen_h // 4),  # Upper right quadrant
                (screen_w // 4, 3 * screen_h // 4),  # Lower left quadrant
                (3 * screen_w // 4, 3 * screen_h // 4),  # Lower right quadrant
                # Add more intermediate points
                (screen_w // 6, screen_h // 3),
                (5 * screen_w // 6, screen_h // 3),
                (screen_w // 6, 2 * screen_h // 3),
                (5 * screen_w // 6, 2 * screen_h // 3),
            ]
            
            remaining_points = num_points - len(points)
            for i in range(min(remaining_points, len(additional_areas))):
                points.append(additional_areas[i])
        
        return points[:num_points]
    
    def get_eye_landmarks(self, landmarks):
        """Extract eye landmarks from dlib face landmarks"""
        # Left eye landmarks (36-41), Right eye landmarks (42-47)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        return left_eye, right_eye
    
    def calculate_eye_center(self, eye_landmarks):
        """Calculate center of eye from landmarks"""
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        return (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
    
    def estimate_gaze_direction(self, left_eye, right_eye, face_landmarks):
        """Simple gaze estimation based on eye centers and face orientation"""
        left_center = self.calculate_eye_center(left_eye)
        right_center = self.calculate_eye_center(right_eye)
        
        # Average eye center
        eye_center_x = (left_center[0] + right_center[0]) // 2
        eye_center_y = (left_center[1] + right_center[1]) // 2
        
        # Nose tip for reference (landmark 30)
        nose_x = face_landmarks.part(30).x
        nose_y = face_landmarks.part(30).y
        
        # Simple gaze estimation (this is basic - you might want to improve this)
        gaze_x = eye_center_x - nose_x
        gaze_y = eye_center_y - nose_y
        
        return (gaze_x, gaze_y)
    
    def run_calibration(self, cap, points):
        """Run calibration phase - captures when gaze is detected"""
        print(f"Starting calibration with {len(points)} points")
        print("Instructions: Look at each green circle")
        print("System will automatically capture when your gaze is detected")
        print("Press 'p' to pause, 'q' to quit, 's' to skip current point")
        calibration_data = []
        
        for i, (target_x, target_y) in enumerate(points):
            print(f"Look at point {i+1}/{len(points)}: ({target_x}, {target_y})")
            
            # Create calibration display window with better focus handling
            cal_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(cal_img, (target_x, target_y), 20, (0, 255, 0), -1)
            cv2.putText(cal_img, f"Point {i+1}/{len(points)}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(cal_img, "Look at the green circle", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(cal_img, "Press 'p' to pause, 's' to skip", (50, self.screen_height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(cal_img, "Press 'q' to quit", (50, self.screen_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Better window creation for reliable key detection
            cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow('Calibration', 0, 0)  # Ensure window is in foreground
            cv2.imshow('Calibration', cal_img)
            cv2.waitKey(1000)  # Wait 1 second to let user focus on point
            
            # Collect gaze data when detected
            gaze_detections = []
            detection_start_time = None
            stable_detection_duration = 0.5  # Require 0.5 seconds of stable detection
            max_wait_time = 10  # Maximum 10 seconds per point
            start_time = time.time()
            paused = False
            
            print("Detecting gaze...", end="", flush=True)
            
            while time.time() - start_time < max_wait_time:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Normal gaze detection logic first
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                
                gaze_detected = False
                current_gaze = None
                
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    left_eye, right_eye = self.get_eye_landmarks(landmarks)
                    gaze_x, gaze_y = self.estimate_gaze_direction(left_eye, right_eye, landmarks)
                    
                    current_gaze = (gaze_x, gaze_y)
                    gaze_detected = True
                    break
                
                if gaze_detected and current_gaze:
                    # Start or continue detection timer
                    if detection_start_time is None:
                        detection_start_time = time.time()
                        print(".", end="", flush=True)
                    
                    # Add to recent detections
                    gaze_detections.append({
                        'timestamp': time.time(),
                        'target_x': target_x,
                        'target_y': target_y,
                        'gaze_x': current_gaze[0],
                        'gaze_y': current_gaze[1],
                        'point_index': i
                    })
                    
                    # Check if we have stable detection for required duration
                    if time.time() - detection_start_time >= stable_detection_duration:
                        print(" Captured!")
                        break
                else:
                    # Reset detection timer if gaze lost
                    detection_start_time = None
                    if len(gaze_detections) > 10:  # Keep only recent detections
                        gaze_detections = gaze_detections[-5:]
                
                # Update display to show detection status
                status_img = cal_img.copy()
                if gaze_detected:
                    cv2.circle(status_img, (target_x, target_y), 25, (0, 255, 255), 3)  # Yellow ring when detecting
                    cv2.putText(status_img, "Gaze detected...", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(status_img, "Looking for gaze...", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                
                # Force window focus and update display
                cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Calibration', status_img)
                
                # Robust key detection with multiple checks
                key_detected = False
                for _ in range(3):  # Check multiple times per frame
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # Key was pressed
                        key_detected = True
                        
                        # Visual feedback for key press
                        feedback_img = status_img.copy()
                        cv2.putText(feedback_img, "Key Press Detected", 
                                   (self.screen_width//2 - 150, self.screen_height//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow('Calibration', feedback_img)
                        cv2.waitKey(200)  # Show feedback for 200ms
                        
                        if key == ord('q'):
                            print("\nQuitting calibration...")
                            cv2.destroyWindow('Calibration')
                            return calibration_data
                        elif key == ord('p'):
                            paused = True
                            print(f"\n--- CALIBRATION PAUSED ---")
                            print(f"Currently on point {i+1}/{len(points)}")
                            print("Options:")
                            print("  'c' - Continue calibration")
                            print("  's' - Skip current point")
                            print("  'q' - Quit calibration")
                            print("Press a key: ", end="", flush=True)
                            
                            # Show paused display
                            pause_img = cal_img.copy()
                            cv2.putText(pause_img, "PAUSED", (self.screen_width//2 - 100, self.screen_height//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                            cv2.putText(pause_img, "Press 'c' to continue, 's' to skip, 'q' to quit", 
                                       (self.screen_width//2 - 300, self.screen_height//2 + 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.imshow('Calibration', pause_img)
                            
                            # Wait for user input during pause with better key detection
                            while paused:
                                for _ in range(5):  # Multiple checks for reliability
                                    pause_key = cv2.waitKey(20) & 0xFF
                                    if pause_key != 255:
                                        if pause_key == ord('c'):
                                            print("c")
                                            print("Continuing calibration...")
                                            paused = False
                                            start_time = time.time()  # Reset timer
                                            detection_start_time = None  # Reset detection
                                            break
                                        elif pause_key == ord('s'):
                                            print("s")
                                            print(f"Skipping point {i+1}")
                                            cv2.destroyWindow('Calibration')
                                            return calibration_data  # Skip this point
                                        elif pause_key == ord('q'):
                                            print("q")
                                            print("Quitting calibration...")
                                            cv2.destroyWindow('Calibration')
                                            return calibration_data
                                if not paused:
                                    break
                            
                            if not paused:
                                # Resume normal display
                                cv2.imshow('Calibration', cal_img)
                                print("Detecting gaze...", end="", flush=True)
                                break
                                
                        elif key == ord('s'):
                            print(f"\nSkipping point {i+1}")
                            break
                        
                        break  # Exit key detection loop
                    
                if key_detected and key in [ord('q'), ord('s')]:
                    break
            
            # Use the collected gaze data for this point
            if gaze_detections:
                # Use the most recent stable detections
                recent_detections = gaze_detections[-int(stable_detection_duration * 30):] if len(gaze_detections) > 5 else gaze_detections
                calibration_data.extend(recent_detections)
                print(f"Point {i+1} completed with {len(recent_detections)} samples")
            else:
                print(f"Point {i+1} failed - no stable gaze detected")
        
        cv2.destroyWindow('Calibration')
        return calibration_data

    def run_adaptive_calibration(self, cap, initial_points, max_additional_points=5):
        """Run true adaptive calibration that adds points based on validation errors"""
        print(f"Starting adaptive calibration with {len(initial_points)} initial points")
        
        # Phase 1: Initial calibration
        cal_data = self.run_calibration(cap, initial_points)
        
        # Phase 2: Quick validation to identify problem areas
        validation_grid = self.generate_calibration_points(9, 'grid', self.screen_width, self.screen_height)
        val_data = self.run_validation(cap, validation_grid, duration_per_point=1.5)
        
        if not val_data:
            return cal_data

    def save_experiment_state(self):
        """Save current experiment state for pause/resume functionality"""
        state = {
            'participant_id': self.participant_id,
            'completed_conditions': list(self.completed_conditions),
            'current_condition_index': self.current_condition_index,
            'total_conditions': self.total_conditions,
            'experiment_data': self.experiment_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"Experiment state saved to {self.state_file}")
    
    def load_experiment_state(self):
        """Load previous experiment state if exists"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.completed_conditions = set(state.get('completed_conditions', []))
                self.current_condition_index = state.get('current_condition_index', 0)
                self.total_conditions = state.get('total_conditions', 0)
                self.experiment_data = state.get('experiment_data', [])
                
                print(f"Loaded previous experiment state:")
                print(f"- Completed conditions: {len(self.completed_conditions)}/{self.total_conditions}")
                print(f"- Can resume from condition {self.current_condition_index + 1}")
                return True
                
            except Exception as e:
                print(f"Error loading state file: {e}")
                return False
        return False
    
    def get_condition_key(self, test_condition):
        """Generate unique key for a test condition"""
        return test_condition['test_id']
    
    def show_pause_menu(self):
        """Show pause menu options"""
        print("\n" + "="*50)
        print("EXPERIMENT PAUSED")
        print("="*50)
        print("Options:")
        print("1. Continue with next condition")
        print("2. Take a break (save progress and exit)")
        print("3. Skip current condition")
        print("4. Show progress summary")
        print("5. Quit experiment")
        print("="*50)
        
        while True:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                return 'continue'
            elif choice == '2':
                self.save_experiment_state()
                print("Progress saved! You can resume later by running the program again.")
                return 'break'
            elif choice == '3':
                print("Skipping current condition...")
                return 'skip'
            elif choice == '4':
                self.show_progress_summary()
                continue
            elif choice == '5':
                print("Are you sure you want to quit? (y/n): ", end="")
                confirm = input().strip().lower()
                if confirm == 'y':
                    self.save_experiment_state()
                    return 'quit'
                continue
            else:
                print("Invalid choice. Please enter 1-5.")
    
    def show_progress_summary(self):
        """Display current progress summary"""
        print("\n" + "-"*40)
        print("PROGRESS SUMMARY")
        print("-"*40)
        print(f"Participant ID: {self.participant_id}")
        print(f"Completed conditions: {len(self.completed_conditions)}/{self.total_conditions}")
        print(f"Current condition: {self.current_condition_index + 1}/{self.total_conditions}")
        
        if self.experiment_data:
            print("\nRecent results:")
            for i, result in enumerate(self.experiment_data[-3:]):  # Show last 3 results
                print(f"  {len(self.experiment_data)-2+i}. {result['resolution'][0]}x{result['resolution'][1]} "
                      f"{result['frame_rate']}fps {result['calibration_pattern']} "
                      f"- Accuracy: {result['accuracy']:.2f}")
        
        completion_rate = (len(self.completed_conditions) / self.total_conditions) * 100
        print(f"\nCompletion rate: {completion_rate:.1f}%")
        print("-"*40)
    
    def estimate_remaining_time(self):
        """Estimate remaining experiment time"""
        if not self.experiment_data:
            return "Unknown"
        
        # Calculate average time per condition from completed conditions
        avg_time_per_condition = 0
        for result in self.experiment_data:
            avg_time_per_condition += result.get('calibration_time', 0)
        
        if len(self.experiment_data) > 0:
            avg_time_per_condition /= len(self.experiment_data)
            avg_time_per_condition += 60  # Add time for validation and setup
            
            remaining_conditions = self.total_conditions - len(self.completed_conditions)
            remaining_minutes = (remaining_conditions * avg_time_per_condition) / 60
            
            return f"~{remaining_minutes:.0f} minutes"
        
        return "Unknown"
        
        # Phase 3: Identify high-error regions and add calibration points
        additional_points = []
        error_threshold = np.percentile([d['error'] for d in val_data], 75)  # Top 25% errors
        
        for val_point in val_data:
            if val_point['error'] > error_threshold and len(additional_points) < max_additional_points:
                # Add a calibration point near the high-error validation point
                target_x, target_y = val_point['target_x'], val_point['target_y']
                
                # Add slight offset to avoid exact same position
                offset_x = np.random.randint(-50, 51)
                offset_y = np.random.randint(-50, 51)
                
                new_x = max(100, min(self.screen_width - 100, target_x + offset_x))
                new_y = max(100, min(self.screen_height - 100, target_y + offset_y))
                
                additional_points.append((new_x, new_y))
        
        # Phase 4: Calibrate additional points if any were identified
        if additional_points:
            print(f"Adding {len(additional_points)} adaptive calibration points")
            additional_cal_data = self.run_calibration(cap, additional_points)
            cal_data.extend(additional_cal_data)
        
        return cal_data
        """Run calibration phase"""
        print(f"Starting calibration with {len(points)} points")
        calibration_data = []
        
        for i, (target_x, target_y) in enumerate(points):
            print(f"Look at point {i+1}/{len(points)}: ({target_x}, {target_y})")
            
            # Create calibration display window
            cal_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(cal_img, (target_x, target_y), 20, (0, 255, 0), -1)
            cv2.putText(cal_img, f"Point {i+1}/{len(points)}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Calibration', cal_img)
            cv2.waitKey(1000)  # Wait 1 second before starting collection
            
            # Collect data for this point
            start_time = time.time()
            point_data = []
            
            while time.time() - start_time < duration_per_point:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    left_eye, right_eye = self.get_eye_landmarks(landmarks)
                    gaze_x, gaze_y = self.estimate_gaze_direction(left_eye, right_eye, landmarks)
                    
                    point_data.append({
                        'timestamp': time.time(),
                        'target_x': target_x,
                        'target_y': target_y,
                        'gaze_x': gaze_x,
                        'gaze_y': gaze_y,
                        'point_index': i
                    })
                
                cv2.imshow('Calibration', cal_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            calibration_data.extend(point_data)
        
        cv2.destroyWindow('Calibration')
        return calibration_data
    
    def run_validation(self, cap, validation_points, duration_per_point=8):
        """Run validation phase - captures when gaze is detected"""
        print("Starting validation phase")
        print("Instructions: Look at each red circle")
        print("System will automatically capture when your gaze is detected")
        print("Press 'p' to pause, 'q' to quit, 's' to skip current point")
        validation_data = []
        
        for i, (target_x, target_y) in enumerate(validation_points):
            print(f"Validation point {i+1}/{len(validation_points)}")
            
            # Create validation display with better focus handling
            val_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(val_img, (target_x, target_y), 15, (0, 0, 255), -1)
            cv2.putText(val_img, "Validation", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(val_img, "Look at the red circle", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(val_img, "Press 'p' to pause, 's' to skip", (50, self.screen_height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(val_img, "Press 'q' to quit", (50, self.screen_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Better window creation for reliable key detection
            cv2.namedWindow('Validation', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Validation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow('Validation', 0, 0)  # Ensure window is in foreground
            cv2.imshow('Validation', val_img)
            cv2.waitKey(500)
            
            # Collect validation data when gaze is detected
            point_measurements = []
            detection_start_time = None
            stable_detection_duration = 0.5  # Require 0.5 seconds of stable detection
            max_wait_time = duration_per_point  # Maximum wait time per validation point
            start_time = time.time()
            paused = False
            
            print("Detecting gaze...", end="", flush=True)
            
            while time.time() - start_time < max_wait_time:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Normal gaze detection logic first
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                
                gaze_detected = False
                current_gaze = None
                
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    left_eye, right_eye = self.get_eye_landmarks(landmarks)
                    gaze_x, gaze_y = self.estimate_gaze_direction(left_eye, right_eye, landmarks)
                    
                    current_gaze = (gaze_x, gaze_y)
                    gaze_detected = True
                    break
                
                if gaze_detected and current_gaze:
                    # Start or continue detection timer
                    if detection_start_time is None:
                        detection_start_time = time.time()
                        print(".", end="", flush=True)
                    
                    # Add to measurements
                    point_measurements.append(current_gaze)
                    
                    # Check if we have stable detection for required duration
                    if time.time() - detection_start_time >= stable_detection_duration:
                        print(" Captured!")
                        break
                else:
                    # Reset detection timer if gaze lost
                    detection_start_time = None
                    if len(point_measurements) > 10:  # Keep only recent measurements
                        point_measurements = point_measurements[-5:]
                
                # Update display to show detection status
                status_img = val_img.copy()
                if gaze_detected:
                    cv2.circle(status_img, (target_x, target_y), 20, (0, 255, 255), 3)  # Yellow ring when detecting
                    cv2.putText(status_img, "Gaze detected...", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(status_img, "Looking for gaze...", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                
                # Force window focus and update display
                cv2.setWindowProperty('Validation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Validation', status_img)
                
                # Robust key detection with multiple checks
                key_detected = False
                for _ in range(3):  # Check multiple times per frame
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # Key was pressed
                        key_detected = True
                        
                        # Visual feedback for key press
                        feedback_img = status_img.copy()
                        cv2.putText(feedback_img, "Key Press Detected", 
                                   (self.screen_width//2 - 150, self.screen_height//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow('Validation', feedback_img)
                        cv2.waitKey(200)  # Show feedback for 200ms
                        
                        if key == ord('q'):
                            print("\nQuitting validation...")
                            cv2.destroyWindow('Validation')
                            return validation_data
                        elif key == ord('p'):
                            paused = True
                            print(f"\n--- VALIDATION PAUSED ---")
                            print(f"Currently on point {i+1}/{len(validation_points)}")
                            print("Options:")
                            print("  'c' - Continue validation")
                            print("  's' - Skip current point")
                            print("  'q' - Quit validation")
                            print("Press a key: ", end="", flush=True)
                            
                            # Show paused display
                            pause_img = val_img.copy()
                            cv2.putText(pause_img, "PAUSED", (self.screen_width//2 - 100, self.screen_height//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                            cv2.putText(pause_img, "Press 'c' to continue, 's' to skip, 'q' to quit", 
                                       (self.screen_width//2 - 300, self.screen_height//2 + 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.imshow('Validation', pause_img)
                            
                            # Wait for user input during pause with better key detection
                            while paused:
                                for _ in range(5):  # Multiple checks for reliability
                                    pause_key = cv2.waitKey(20) & 0xFF
                                    if pause_key != 255:
                                        if pause_key == ord('c'):
                                            print("c")
                                            print("Continuing validation...")
                                            paused = False
                                            start_time = time.time()  # Reset timer
                                            detection_start_time = None  # Reset detection
                                            break
                                        elif pause_key == ord('s'):
                                            print("s")
                                            print(f"Skipping validation point {i+1}")
                                            cv2.destroyWindow('Validation')
                                            return validation_data  # Skip this point
                                        elif pause_key == ord('q'):
                                            print("q")
                                            print("Quitting validation...")
                                            cv2.destroyWindow('Validation')
                                            return validation_data
                                if not paused:
                                    break
                            
                            if not paused:
                                # Resume normal display
                                cv2.imshow('Validation', val_img)
                                print("Detecting gaze...", end="", flush=True)
                                break
                                
                        elif key == ord('s'):
                            print(f"\nSkipping validation point {i+1}")
                            break
                        
                        break  # Exit key detection loop
                    
                if key_detected and key in [ord('q'), ord('s')]:
                    break
            
            if point_measurements:
                # Calculate average gaze for this point using recent measurements
                recent_measurements = point_measurements[-int(stable_detection_duration * 30):] if len(point_measurements) > 5 else point_measurements
                avg_gaze_x = sum(m[0] for m in recent_measurements) / len(recent_measurements)
                avg_gaze_y = sum(m[1] for m in recent_measurements) / len(recent_measurements)
                
                # Calculate error (simplified - you might want to improve this)
                error = math.sqrt((avg_gaze_x - (target_x - self.screen_width//2))**2 + 
                                (avg_gaze_y - (target_y - self.screen_height//2))**2)
                
                validation_data.append({
                    'target_x': target_x,
                    'target_y': target_y,
                    'estimated_gaze_x': avg_gaze_x,
                    'estimated_gaze_y': avg_gaze_y,
                    'error': error,
                    'num_measurements': len(recent_measurements)
                })
                print(f"Validation point {i+1} completed with {len(recent_measurements)} samples")
            else:
                print(f"Validation point {i+1} failed - no stable gaze detected")
        
        cv2.destroyWindow('Validation')
        return validation_data
    
    def run_experiment_condition(self, test_condition):
        """Run one experimental condition based on test configuration"""
        test_id = test_condition['test_id']
        
        # Check if this condition was already completed
        if test_id in self.completed_conditions:
            print(f"Test {test_id} already completed, skipping...")
            return None
        
        print(f"\n--- Running Test {test_id} ({self.current_condition_index + 1}/{self.total_conditions}) ---")
        print(f"Requested Settings:")
        if test_condition['resolution'] is not None:
            print(f"Resolution: {test_condition['resolution'][0]}x{test_condition['resolution'][1]}")
        else:
            print(f"Resolution: Camera Default")
        
        if test_condition['frame_rate'] is not None:
            print(f"Frame Rate: {test_condition['frame_rate']}fps")
        else:
            print(f"Frame Rate: Camera Default")
        
        if test_condition['autofocus'] is not None:
            print(f"Autofocus: {'On' if test_condition['autofocus'] else 'Off'}")
        else:
            print(f"Autofocus: Camera Default")
        
        print(f"Calibration: {test_condition['calibration_points']} points, {test_condition['pattern']} pattern")
        print(f"Estimated remaining time: {self.estimate_remaining_time()}")
        
        # Setup camera
        cap = self.setup_camera(
            test_condition['resolution'], 
            test_condition['frame_rate'], 
            test_condition['autofocus']
        )
        
        # Get and print actual camera parameters
        actual_params = self.get_camera_parameters(cap)
        print("\nActual Camera Parameters:")
        print(f"Resolution: {actual_params['resolution'][0]}x{actual_params['resolution'][1]}")
        print(f"Frame Rate: {actual_params['frame_rate']:.1f}fps")
        print(f"Autofocus: {'On' if actual_params['autofocus'] else 'Off'}")
        
        try:
            # Generate calibration points
            cal_points = self.generate_calibration_points(
                test_condition['calibration_points'], 
                test_condition['pattern'], 
                self.screen_width, 
                self.screen_height
            )
            
            # Generate validation points (always use 9-point grid for consistency)
            val_points = self.generate_calibration_points(
                9, 'grid', self.screen_width, self.screen_height)
            
            # Run calibration - captures automatically when gaze is detected
            start_time = time.time()
            if test_condition['pattern'] == 'adaptive':
                cal_data = self.run_adaptive_calibration(cap, cal_points)
            else:
                cal_data = self.run_calibration(cap, cal_points)
            calibration_time = time.time() - start_time
            
            # Run validation
            val_data = self.run_validation(cap, val_points)
            
            # Calculate metrics
            if val_data:
                errors = [d['error'] for d in val_data]
                measured_accuracy = sum(errors) / len(errors)  # Mean error
                precision = np.std(errors)  # Standard deviation of errors
            else:
                measured_accuracy = precision = float('inf')
            
            # Store results
            condition_result = {
                'participant_id': self.participant_id,
                'timestamp': datetime.now().isoformat(),
                'test_id': test_id,
                'resolution': f"{test_condition['resolution'][0]}x{test_condition['resolution'][1]}",
                'frame_rate': test_condition['frame_rate'],
                'autofocus': test_condition['autofocus'],
                'calibration_points': test_condition['calibration_points'],
                'pattern': test_condition['pattern'].title(),
                'measured_accuracy': measured_accuracy,
                'precision': precision,
                'calibration_time': calibration_time,
                'num_validation_points': len(val_data),
                'calibration_data': cal_data,
                'validation_data': val_data
            }
            
            self.experiment_data.append(condition_result)
            self.completed_conditions.add(test_id)
            
            print(f"Test completed:")
            print(f"  Measured accuracy: {measured_accuracy:.2f}°")
            print(f"  Precision: {precision:.2f}°")
            
            # Auto-save progress after each condition
            self.save_experiment_state()
            
            return condition_result
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_full_experiment(self):
        """Run the complete experiment with predefined test conditions"""
        print(f"Starting experiment for participant {self.participant_id}")
        print("Make sure you have downloaded shape_predictor_68_face_landmarks.dat")
        print("\nExperiment Features:")
        print("- 32 predefined test conditions")
        print("- Automatic progress saving after each test")
        print("- Can pause and resume anytime")
        print("- Press 'p' during calibration/validation to pause")
        print("- Press 'q' to quit early\n")
        
        # Try to load previous state
        resumed = self.load_experiment_state()
        
        if not resumed:
            print(f"Total tests to complete: {self.total_conditions}")
            # Randomize order to avoid systematic bias (only for new experiments)
            np.random.shuffle(self.test_conditions)
            self.current_condition_index = 0
        else:
            print(f"Resuming experiment - {len(self.completed_conditions)} tests already completed")
        
        # Filter out completed conditions if resuming
        remaining_conditions = []
        for i, condition in enumerate(self.test_conditions):
            test_id = condition['test_id']
            if test_id not in self.completed_conditions:
                remaining_conditions.append((i, condition))
        
        if not remaining_conditions:
            print("All tests completed! Generating final results...")
            self.save_results()
            return
        
        print(f"Remaining tests: {len(remaining_conditions)}")
        
        # Main experiment loop
        for condition_index, (original_index, test_condition) in enumerate(remaining_conditions):
            self.current_condition_index = original_index
            test_id = test_condition['test_id']
            
            print(f"\n{'='*60}")
            print(f"Test {condition_index + 1}/{len(remaining_conditions)} "
                  f"(Overall: {original_index + 1}/{self.total_conditions})")
            print(f"Test ID: {test_id}")
            print(f"{'='*60}")
            
            # Show pause menu before each condition
            print("\nBefore starting this test:")
            print("Press Enter to continue, or 'p' to pause: ", end="")
            user_input = input().strip().lower()
            
            if user_input == 'p':
                action = self.show_pause_menu()
                if action == 'break':
                    print("Experiment paused. Run the program again to resume.")
                    return
                elif action == 'quit':
                    print("Experiment terminated by user.")
                    return
                elif action == 'skip':
                    # Mark condition as completed to skip it
                    self.completed_conditions.add(test_id)
                    self.save_experiment_state()
                    continue
            
            try:
                result = self.run_experiment_condition(test_condition)
                if result is None:  # Test was skipped
                    continue
                    
                # Show progress after each test
                print(f"\nProgress: {len(self.completed_conditions)}/{self.total_conditions} tests completed")
                
                # Offer break after every few tests
                if (condition_index + 1) % 5 == 0 and condition_index < len(remaining_conditions) - 1:
                    print(f"\nYou've completed {condition_index + 1} tests.")
                    print("Would you like to take a break? (y/n): ", end="")
                    break_choice = input().strip().lower()
                    
                    if break_choice == 'y':
                        action = self.show_pause_menu()
                        if action == 'break':
                            print("Experiment paused. Run the program again to resume.")
                            return
                        elif action == 'quit':
                            print("Experiment terminated by user.")
                            return
                
            except KeyboardInterrupt:
                print("\n\nExperiment interrupted by user")
                action = self.show_pause_menu()
                if action == 'break':
                    print("Progress saved. Run the program again to resume.")
                    return
                elif action == 'quit':
                    print("Experiment terminated.")
                    return
                elif action == 'continue':
                    continue
                    
            except Exception as e:
                print(f"Error in test {test_id}: {e}")
                print("Saving progress and continuing...")
                self.save_experiment_state()
                continue
        
        # Experiment completed
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'='*60}")
        print(f"Total tests completed: {len(self.completed_conditions)}")
        print("Generating final results...")
        
        # Save final results
        self.save_results()
        
        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            print("Temporary state file cleaned up.")
        
        print(f"All data saved in: {self.participant_dir}")
    
    def save_results(self):
        """Save experiment results to participant-specific files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results with metadata
        results_with_metadata = {
            'participant_metadata': self.participant_metadata,
            'experiment_data': self.experiment_data,
            'completion_timestamp': datetime.now().isoformat()
        }
        
        # Save detailed JSON results
        detailed_file = os.path.join(
            self.participant_dir, 
            f"experiment_results_{self.participant_id}_{timestamp}.json"
        )
        
        with open(detailed_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        # Save CSV summary
        summary_rows = []
        for result in self.experiment_data:
            summary_rows.append({
                'Test ID': result['test_id'],
                'Resolution': result['resolution'],
                'Frame Rate': f"{result['frame_rate']}fps",
                'Autofocus': 'On' if result['autofocus'] else 'Off',
                'Calibration Points': result['calibration_points'],
                'Pattern': result['pattern'],
                'Measured Accuracy (°)': round(result['measured_accuracy'], 2),
                'Precision (°)': round(result['precision'], 2),
                'Calibration Time (s)': round(result['calibration_time'], 2),
                'Participant ID': result['participant_id'],
                'Timestamp': result['timestamp']
            })
        
        # Sort by Test ID to maintain order
        summary_rows.sort(key=lambda x: x['Test ID'])
        
        df = pd.DataFrame(summary_rows)
        csv_file = os.path.join(
            self.participant_dir,
            f"experiment_results_{self.participant_id}_{timestamp}.csv"
        )
        df.to_csv(csv_file, index=False)
        
        # Update master summary
        self._update_master_summary()
        
        print(f"\nResults saved to:")
        print(f"- Detailed: {detailed_file}")
        print(f"- Results: {csv_file}")
        print(f"- Participant directory: {self.participant_dir}")
    
    def _update_master_summary(self):
        """Update master summary file with all participants"""
        master_file = os.path.join(self.base_output_dir, "master_summary.csv")
        
        # Collect data from all participants
        all_data = []
        participants = self.list_participants(self.base_output_dir)
        
        for participant in participants:
            participant_dir = participant['directory']
            
            # Find the most recent CSV file for this participant
            csv_files = [f for f in os.listdir(participant_dir) 
                        if f.startswith("experiment_results_") and f.endswith(".csv")]
            
            if csv_files:
                latest_csv = sorted(csv_files)[-1]  # Most recent
                csv_path = os.path.join(participant_dir, latest_csv)
                
                try:
                    participant_data = pd.read_csv(csv_path)
                    all_data.append(participant_data)
                except Exception as e:
                    print(f"Warning: Could not read {csv_path}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(master_file, index=False)
            print(f"- Master summary: {master_file}")

    @staticmethod
    def generate_analysis_report(base_dir="experiment_data"):
        """Generate a comprehensive analysis report for all participants"""
        participants = GazeTrackingExperiment.list_participants(base_dir)
        
        if not participants:
            print("No participants found.")
            return
        
        print(f"\nANALYSIS REPORT")
        print("=" * 50)

    @staticmethod
    def cleanup_incomplete_participants(base_dir="experiment_data", confirm=True):
        """Remove participants with no progress (cleanup utility)"""
        participants = GazeTrackingExperiment.list_participants(base_dir)
        incomplete = [p for p in participants if p['progress'] == 0 and p['status'] == "Not started"]
        
        if not incomplete:
            print("No incomplete participants found.")
            return
        
        print(f"Found {len(incomplete)} participants with no progress:")
        for p in incomplete:
            print(f"  - {p['id']}")
        
        if confirm:
            response = input("\nDelete these participant directories? (y/N): ").strip().lower()
            if response != 'y':
                print("Cleanup cancelled.")
                return
        
        import shutil
        for p in incomplete:
            try:
                shutil.rmtree(p['directory'])
                print(f"Deleted: {p['id']}")
            except Exception as e:
                print(f"Error deleting {p['id']}: {e}")
        
        print("Cleanup completed.")
        print(f"Total participants: {len(participants)}")
        
        completed = [p for p in participants if p['status'] == "Completed"]
        in_progress = [p for p in participants if "progress" in p['status']]
        not_started = [p for p in participants if p['status'] == "Not started"]
        
        print(f"Completed: {len(completed)}")
        print(f"In progress: {len(in_progress)}")
        print(f"Not started: {len(not_started)}")
        
        if completed:
            print(f"\nCompleted participants:")
            for p in completed:
                print(f"  - {p['id']}")
        
        if in_progress:
            print(f"\nIn-progress participants:")
            for p in in_progress:
                print(f"  - {p['id']}: {p['status']}")
        
        # Check for master summary
        master_file = os.path.join(base_dir, "master_summary.csv")
        if os.path.exists(master_file):
            try:
                df = pd.read_csv(master_file)
                print(f"\nMaster dataset: {len(df)} conditions across {df['participant_id'].nunique()} participants")
                print(f"Data file: {master_file}")
            except Exception as e:
                print(f"Error reading master file: {e}")
        
        print("=" * 50)

def get_participant_id():
    """Get participant ID with validation and participant management"""
    print("PARTICIPANT MANAGEMENT")
    print("=" * 50)
    
    # Show existing participants
    participants = GazeTrackingExperiment.list_participants()
    if participants:
        print("Existing participants:")
        for i, p in enumerate(participants, 1):
            status_color = "✓" if p['status'] == "Completed" else "○" if "progress" in p['status'] else "✗"
            print(f"  {i:2}. {p['id']:15} - {status_color} {p['status']:20} ({p['progress']:.1f}%)")
        print()
    
    print("Options:")
    print("1. Create new participant")
    if participants:
        print("2. Continue existing participant")
        print("3. View participant details")
        print("4. Generate analysis report")
    print("0. Exit")
    print()
    
    while True:
        choice = input("Enter choice: ").strip()
        
        if choice == '0':
            return None
            
        elif choice == '1':
            # Create new participant
            while True:
                participant_id = input("Enter new participant ID (letters/numbers only): ").strip()
                
                if not participant_id:
                    print("Participant ID cannot be empty.")
                    continue
                
                # Validate participant ID
                if not participant_id.replace('_', '').replace('-', '').isalnum():
                    print("Participant ID should contain only letters, numbers, underscores, and hyphens.")
                    continue
                
                # Check if participant already exists
                existing_ids = [p['id'] for p in participants]
                if participant_id in existing_ids:
                    print(f"Participant '{participant_id}' already exists. Choose a different ID.")
                    continue
                
                return participant_id
                
        elif choice == '2' and participants:
            # Continue existing participant
            print("Select participant to continue:")
            for i, p in enumerate(participants, 1):
                if p['status'] != "Completed":
                    print(f"  {i}. {p['id']} - {p['status']}")
            
            try:
                selection = int(input("Enter participant number: ")) - 1
                if 0 <= selection < len(participants):
                    selected = participants[selection]
                    if selected['status'] == "Completed":
                        print("This participant has already completed the experiment.")
                        continue
                    return selected['id']
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == '3' and participants:
            # View participant details
            print("Select participant to view:")
            for i, p in enumerate(participants, 1):
                print(f"  {i}. {p['id']}")
            
            try:
                selection = int(input("Enter participant number: ")) - 1
                if 0 <= selection < len(participants):
                    participant = participants[selection]
                    print(f"\nParticipant Details:")
                    print(f"ID: {participant['id']}")
                    print(f"Status: {participant['status']}")
                    print(f"Progress: {participant['progress']:.1f}%")
                    print(f"Directory: {participant['directory']}")
                    
                    # Show recent files
                    files = os.listdir(participant['directory'])
                    if files:
                        print("Files:")
                        for file in sorted(files):
                            print(f"  - {file}")
                    print()
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
        
        elif choice == '4' and participants:
            # Generate analysis report
            GazeTrackingExperiment.generate_analysis_report()
            print()
            
        else:
            print("Invalid choice.")

def main():
    print("Gaze Tracking Parameter Experiment")
    print("==================================")
    print("32-Test Configuration Study")
    print()
    print("Directory Structure:")
    print("experiment_data/")
    print("├── participant_001/")
    print("│   ├── experiment_state.json")
    print("│   ├── experiment_results_001_20250528_143022.json")
    print("│   └── experiment_results_001_20250528_143022.csv")
    print("├── participant_002/")
    print("│   └── ...")
    print("└── master_summary.csv")
    print()
    print("Features:")
    print("- 32 predefined test conditions (hardcoded)")
    print("- Individual participant tracking")
    print("- Automatic progress saving")
    print("- Pause and resume capability")
    print("- Master data aggregation")
    print()
    
    # Get and print default camera parameters
    print("\nDetecting Camera Parameters:")
    try:
        # Create a temporary experiment instance to access camera methods
        temp_experiment = GazeTrackingExperiment("temp")
        
        # Create a temporary camera capture
        temp_cap = cv2.VideoCapture(0)
        
        if temp_cap.isOpened():
            # Get actual camera parameters
            actual_params = temp_experiment.get_camera_parameters(temp_cap)
            print("Current Camera Parameters:")
            print(f"Resolution: {actual_params['resolution'][0]}x{actual_params['resolution'][1]}")
            print(f"Frame Rate: {actual_params['frame_rate']:.1f}fps")
            print(f"Autofocus: {'On' if actual_params['autofocus'] else 'Off'}")
            
            # Release the temporary camera
            temp_cap.release()
        else:
            print("Could not open camera to detect parameters.")
    except Exception as e:
        print(f"Error detecting camera parameters: {e}")
    print()
    
    # Check for required files
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("\nERROR: shape_predictor_68_face_landmarks.dat not found!")
        print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract the .dat file to the same directory as this script.")
        return
    
    # Get participant ID
    participant_id = get_participant_id()
    if participant_id is None:
        print("Exiting program.")
        return
    
    print(f"\nSelected participant: {participant_id}")
    
    try:
        # Create experiment instance
        experiment = GazeTrackingExperiment(participant_id)
        
        # Show participant info
        summary = experiment.get_participant_summary()
        print(f"\nParticipant Summary:")
        print(f"- ID: {summary['participant_id']}")
        print(f"- Total tests: {experiment.total_conditions}")
        print(f"- Progress: {summary['completed_conditions']}/{summary['total_conditions']} tests")
        print(f"- Data directory: {summary['participant_directory']}")
        
        # Check if there's a previous session to resume
        if os.path.exists(experiment.state_file):
            print(f"\nFound previous experiment session for {participant_id}")
            print("Would you like to:")
            print("1. Resume previous session")
            print("2. Start fresh (will delete previous progress)")
            
            while True:
                choice = input("Enter choice (1 or 2): ").strip()
                if choice == '1':
                    print("Resuming previous session...")
                    break
                elif choice == '2':
                    print("Starting fresh experiment...")
                    if os.path.exists(experiment.state_file):
                        os.remove(experiment.state_file)
                    experiment.completed_conditions = set()
                    experiment.experiment_data = []
                    break
                else:
                    print("Please enter 1 or 2")
        
        print("\nIMPORTANT INSTRUCTIONS:")
        print("- The experiment will test 32 different camera configurations")
        print("- System automatically captures when your gaze is detected")
        print("- Look directly at each calibration point until captured")
        print("- Calibration points are green circles, validation points are red")
        print("- Key controls during calibration/validation:")
        print("  * Press 'p' to pause (now works reliably with visual feedback!)")
        print("  * Press 's' to skip current point if having trouble")
        print("  * Press 'q' to quit current phase")
        print("  * Press 'c' to continue when paused")
        print("- You will see 'Key Press Detected' confirmation")
        print("- You can pause anytime before a test starts")
        print("- Progress is automatically saved after each test")
        print("- Take breaks whenever you feel fatigued")
        print("- The experiment is now faster with automatic gaze detection")
        print("- You can resume later if needed")
        
        input("\nPress Enter when ready to begin...")
        
        experiment.run_full_experiment()
        print("\nExperiment completed successfully!")
        print(f"Results saved in: {experiment.participant_dir}")
        
    except KeyboardInterrupt:
        print("\nExperiment terminated by user")
        print("Progress has been saved. You can resume by running the program again.")
        
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        print("Progress has been saved if any tests were completed.")

if __name__ == "__main__":
    main()