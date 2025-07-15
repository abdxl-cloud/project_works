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
        
        # Get camera capabilities first to ensure we only use supported parameters
        self.camera_capabilities = self.get_camera_capabilities()
        
        # Generate the test conditions based on actual camera capabilities
        self.test_conditions = self._generate_test_conditions()
        
        # Data storage
        self.experiment_data = []
        
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
            'system_info': self._get_system_info(),
            'camera_capabilities': self.camera_capabilities
        }

    def _generate_test_conditions(self):
        """Generate test conditions based on actual camera capabilities
        Only using camera's default parameters, not generating downscaled options"""
        try:
            # Get camera capabilities
            capabilities = self.camera_capabilities
            autofocus_supported = capabilities['autofocus_supported']
            default_params = capabilities['default_params']
            
            print(f"Detected camera capabilities:")
            print(f"Default Resolution: {default_params['resolution'][0]}x{default_params['resolution'][1]}")
            print(f"Default Frame Rate: {default_params['frame_rate']:.1f}fps")
            print(f"Default Autofocus: {'On' if default_params['autofocus'] else 'Off'}")
            print(f"Autofocus Supported: {'Yes' if autofocus_supported else 'No'}")
            
            # Only use camera's default resolution
            resolutions = [default_params['resolution']]
            
            # Only use camera's default frame rate
            frame_rates = [default_params['frame_rate']]
            
            # Autofocus options based on camera support
            # Only include autofocus options if the camera actually supports it
            if autofocus_supported:
                autofocus_options = [default_params['autofocus'], not default_params['autofocus']]
            else:
                autofocus_options = [default_params['autofocus']]  # Only default (which will be off)
            
            # Calibration variations
            calibration_points = [5, 9, 16]
            
            # Generate test conditions
            test_conditions = []
            test_id = 1
            
            for autofocus in autofocus_options:
                for points in calibration_points:
                    condition = {
                        'test_id': f"T{test_id:02d}",
                        'resolution': resolutions[0],
                        'frame_rate': frame_rates[0],
                        'autofocus': autofocus,
                        'calibration_points': points
                    }
                    test_conditions.append(condition)
                    test_id += 1
            
            print(f"Generated {len(test_conditions)} test conditions using only camera's default parameters")
            print(f"Resolution: {resolutions[0][0]}x{resolutions[0][1]}")
            print(f"Frame rate: {frame_rates[0]:.1f}fps")
            print(f"Autofocus options: {len(autofocus_options)}")
            
        except Exception as e:
            print(f"Warning: Could not generate test conditions based on camera capabilities: {e}")
            # Fallback to a simplified set of conditions
            test_conditions = self._generate_fallback_conditions()
        
        return test_conditions
    
    def _generate_fallback_conditions(self):
        """Generate fallback test conditions when camera detection fails
        Only using camera's default parameters, not generating downscaled options"""
        print("Camera detection failed, using conservative fallback configuration")
        
        # Use only camera defaults for resolution and frame rate
        # This is the safest approach when camera detection fails
        fallback_resolution = (640, 480)  # Standard VGA as a safe default
        fallback_fps = 30               # Standard frame rate as a safe default
        autofocus_options = [False]     # Disable autofocus by default for compatibility
        calibration_points = [5, 9, 16] # Still test different calibration points
        
        test_conditions = []
        test_id = 1
        
        # Generate minimal test conditions
        for autofocus in autofocus_options:
            for points in calibration_points:
                condition = {
                    'test_id': f"T{test_id:02d}",
                    'resolution': fallback_resolution,
                    'frame_rate': fallback_fps,
                    'autofocus': autofocus,
                    'calibration_points': points
                }
                test_conditions.append(condition)
                test_id += 1
        
        print(f"Generated {len(test_conditions)} conservative fallback test conditions")
        print(f"Using safe defaults: Resolution {fallback_resolution[0]}x{fallback_resolution[1]}, {fallback_fps}fps")
        print("Note: Using only camera defaults to ensure compatibility")
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
        return self.camera_capabilities['default_params']
    
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
        
        # Set autofocus if specified and supported by camera
        if autofocus is not None and hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
            # Only try to set autofocus if it's actually supported
            if self.camera_capabilities['autofocus_supported']:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            else:
                print("Warning: Autofocus not supported by this camera, ignoring autofocus setting")
        
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
    
    def generate_calibration_points(self, num_points, screen_w, screen_h):
        """Generate calibration points based on count"""
        points = []
        margin = 100  # Pixels from edge
        
        # Default to grid pattern
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
        
        return points
    
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
    
    def map_gaze_to_screen(self, gaze_x, gaze_y, calibration_data):
        """Map relative gaze coordinates to screen coordinates using calibration data"""
        if not calibration_data or len(calibration_data) < 3:
            # Fallback: simple scaling (not accurate but prevents huge errors)
            # Scale gaze coordinates to reasonable screen range
            scale_factor = 5.0  # Adjust based on typical gaze range
            screen_x = self.screen_width // 2 + gaze_x * scale_factor
            screen_y = self.screen_height // 2 + gaze_y * scale_factor
            return (screen_x, screen_y)
        
        # Use calibration data to create a mapping
        # Simple linear interpolation based on nearest calibration points
        min_distance = float('inf')
        closest_point = None
        
        for cal_point in calibration_data:
            if 'gaze_x' in cal_point and 'gaze_y' in cal_point:
                distance = math.sqrt((gaze_x - cal_point['gaze_x'])**2 + 
                                   (gaze_y - cal_point['gaze_y'])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = cal_point
        
        if closest_point:
            # Simple offset-based mapping using closest calibration point
            gaze_offset_x = gaze_x - closest_point['gaze_x']
            gaze_offset_y = gaze_y - closest_point['gaze_y']
            
            # Scale the offset (this is a simplified approach)
            scale_factor = 3.0
            screen_x = closest_point['target_x'] + gaze_offset_x * scale_factor
            screen_y = closest_point['target_y'] + gaze_offset_y * scale_factor
            
            return (screen_x, screen_y)
        
        # Fallback if no valid calibration data
        return (self.screen_width // 2, self.screen_height // 2)
    
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
    
    def save_progress_csv(self):
        """Save current experiment progress as CSV file"""
        if not self.experiment_data:
            return  # No data to save yet
        
        # Create progress CSV with current data
        progress_rows = []
        for result in self.experiment_data:
            # Get the actual values for default parameters
            resolution_str = result['resolution']
            frame_rate_str = f"{result['frame_rate']}fps" if result['frame_rate'] is not None else f"{self.camera_capabilities['default_params']['frame_rate']:.1f}fps"
            
            # For autofocus, show the actual value instead of 'Default'
            if result['autofocus'] is None:
                autofocus_str = 'On' if self.camera_capabilities['default_params']['autofocus'] else 'Off'
            else:
                autofocus_str = 'On' if result['autofocus'] else 'Off'
            
            progress_rows.append({
                'Test ID': result['test_id'],
                'Resolution': resolution_str if result['resolution'] is not None else f"{self.camera_capabilities['default_params']['resolution'][0]}x{self.camera_capabilities['default_params']['resolution'][1]}",
                'Frame Rate': frame_rate_str,
                'Autofocus': autofocus_str,
                'Calibration Points': result['calibration_points'],
                'Measured Accuracy (°)': round(result['measured_accuracy'], 2) if not math.isinf(result['measured_accuracy']) else "N/A",
                'Precision (°)': round(result['precision'], 2) if not math.isinf(result['precision']) else "N/A",
                'Calibration Time (s)': round(result['calibration_time'], 2),
                'Participant ID': result['participant_id'],
                'Timestamp': result['timestamp']
            })
        
        # Sort by Test ID to maintain order
        progress_rows.sort(key=lambda x: x['Test ID'])
        
        df = pd.DataFrame(progress_rows)
        
        # Save as progress CSV (overwrites previous progress file)
        progress_csv_file = os.path.join(
            self.participant_dir,
            f"experiment_progress_{self.participant_id}.csv"
        )
        df.to_csv(progress_csv_file, index=False)
        
        print(f"Progress saved to CSV: {progress_csv_file} ({len(self.experiment_data)} conditions completed)")
    
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
                # Get actual values for default parameters
                resolution_str = result['resolution']
                if result['resolution'] is None:
                    resolution_str = f"{self.camera_capabilities['default_params']['resolution'][0]}x{self.camera_capabilities['default_params']['resolution'][1]}"
                
                frame_rate_str = f"{result['frame_rate']}fps" if result['frame_rate'] is not None else f"{self.camera_capabilities['default_params']['frame_rate']:.1f}fps"
                
                print(f"  {len(self.experiment_data)-2+i}. {resolution_str} "
                      f"{frame_rate_str} "
                      f"- Accuracy: {result['measured_accuracy'] if not math.isinf(result['measured_accuracy']) else 'N/A'}")
        
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
    
    def run_validation(self, cap, validation_points):
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
            max_wait_time = 8  # Maximum wait time per validation point
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
                
                # Map gaze coordinates to screen coordinates using calibration data
                estimated_screen_x, estimated_screen_y = self.map_gaze_to_screen(
                    avg_gaze_x, avg_gaze_y, cal_data)
                
                # Calculate angular error in degrees
                # Assume typical viewing distance of 60cm and screen size
                viewing_distance_cm = 60
                screen_width_cm = 30  # Approximate for typical monitor
                screen_height_cm = 20
                
                # Convert pixel differences to cm
                pixels_per_cm_x = self.screen_width / screen_width_cm
                pixels_per_cm_y = self.screen_height / screen_height_cm
                
                # Calculate error in pixels, then convert to cm
                error_pixels = math.sqrt((estimated_screen_x - target_x)**2 + 
                                       (estimated_screen_y - target_y)**2)
                error_cm = error_pixels / ((pixels_per_cm_x + pixels_per_cm_y) / 2)
                
                # Convert to angular error in degrees
                error = math.degrees(math.atan(error_cm / viewing_distance_cm))
                
                validation_data.append({
                    'target_x': target_x,
                    'target_y': target_y,
                    'estimated_gaze_x': avg_gaze_x,
                    'estimated_gaze_y': avg_gaze_y,
                    'estimated_screen_x': estimated_screen_x,
                    'estimated_screen_y': estimated_screen_y,
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
            default_res = self.camera_capabilities['default_params']['resolution']
            print(f"Resolution: Camera Default ({default_res[0]}x{default_res[1]})")
        
        if test_condition['frame_rate'] is not None:
            print(f"Frame Rate: {test_condition['frame_rate']}fps")
        else:
            default_fps = self.camera_capabilities['default_params']['frame_rate']
            print(f"Frame Rate: Camera Default ({default_fps:.1f}fps)")
        
        if test_condition['autofocus'] is not None:
            print(f"Autofocus: {'On' if test_condition['autofocus'] else 'Off'}")
        else:
            default_af = self.camera_capabilities['default_params']['autofocus']
            print(f"Autofocus: Camera Default ({'On' if default_af else 'Off'})")
        
        print(f"Calibration: {test_condition['calibration_points']} points")
        print(f"Estimated remaining time: {self.estimate_remaining_time()}")
        
        # Setup camera with proper error handling for unsupported parameters
        try:
            # Only try to set autofocus if it's supported
            autofocus_to_set = test_condition['autofocus']
            if autofocus_to_set is not None and not self.camera_capabilities['autofocus_supported']:
                print("Warning: Autofocus not supported by this camera, using default setting")
                autofocus_to_set = None
            
            cap = self.setup_camera(
                test_condition['resolution'], 
                test_condition['frame_rate'], 
                autofocus_to_set
            )
            
            # Get and print actual camera parameters
            actual_params = self.get_camera_parameters(cap)
            print("\nActual Camera Parameters:")
            print(f"Resolution: {actual_params['resolution'][0]}x{actual_params['resolution'][1]}")
            print(f"Frame Rate: {actual_params['frame_rate']:.1f}fps")
            print(f"Autofocus: {'On' if actual_params['autofocus'] else 'Off'}")
            
            # Generate calibration points
            cal_points = self.generate_calibration_points(
                test_condition['calibration_points'], 
                self.screen_width, 
                self.screen_height
            )
            
            # Generate validation points (always use 9-point grid for consistency)
            val_points = self.generate_calibration_points(
                9, self.screen_width, self.screen_height)
            
            # Run calibration - captures automatically when gaze is detected
            start_time = time.time()
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
                measured_accuracy = float('inf')
                precision = float('inf')
            
            # Store results
            resolution_str = f"{test_condition['resolution'][0]}x{test_condition['resolution'][1]}" if test_condition['resolution'] else "Default"
            
            condition_result = {
                'participant_id': self.participant_id,
                'timestamp': datetime.now().isoformat(),
                'test_id': test_id,
                'resolution': resolution_str,
                'frame_rate': test_condition['frame_rate'],
                'autofocus': test_condition['autofocus'],
                'calibration_points': test_condition['calibration_points'],
                'measured_accuracy': measured_accuracy,
                'precision': precision,
                'calibration_time': calibration_time,
                'num_validation_points': len(val_data),
                'calibration_data': cal_data,
                'validation_data': val_data,
                'actual_params': actual_params
            }
            
            self.experiment_data.append(condition_result)
            self.completed_conditions.add(test_id)
            
            print(f"Test completed:")
            if not math.isinf(measured_accuracy):
                print(f"  Measured accuracy: {measured_accuracy:.2f}°")
                print(f"  Precision: {precision:.2f}°")
            else:
                print(f"  Measured accuracy: N/A (no validation data)")
                print(f"  Precision: N/A (no validation data)")
            
            # Auto-save progress after each condition
            self.save_experiment_state()
            
            # Save progress as CSV after each condition
            self.save_progress_csv()
            
            return condition_result
            
        except Exception as e:
            print(f"Error in test {test_id}: {e}")
            # Handle the NoneType error specifically
            if "'NoneType' object is not subscriptable" in str(e):
                print("This error is likely due to a face detection issue. Skipping this test.")
            self.save_experiment_state()  # Save state even on error
            return None
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            cv2.destroyAllWindows()
    
    def run_full_experiment(self):
        """Run the complete experiment with predefined test conditions"""
        print(f"Starting experiment for participant {self.participant_id}")
        print("Make sure you have downloaded shape_predictor_68_face_landmarks.dat")
        print("\nExperiment Features:")
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
            # Save current progress as CSV when resuming
            self.save_progress_csv()
        
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
                self.show_progress_summary()
                
                # Ask if user wants to pause after each test
                print("\nPress Enter to continue to next test, or 'p' to pause: ", end="")
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
                        continue
                
            except KeyboardInterrupt:
                print("\nExperiment interrupted by user.")
                self.save_experiment_state()
                print("Progress saved. Run the program again to resume.")
                return
            except Exception as e:
                print(f"Error in test {test_id}: {e}")
                self.save_experiment_state()  # Save state even on error
                print("Error occurred, but progress was saved. You can resume later.")
                return
        
        # All tests completed
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED!")
        print("="*60)
        print(f"All {self.total_conditions} tests completed for participant {self.participant_id}")
        
        # Save final results
        self.save_results()
        
        # Generate analysis report
        self.generate_analysis_report()
        
        print("\nThank you for participating!")
        print(f"Results saved to {self.participant_dir}")
    
    def save_results(self):
        """Save experiment results to JSON and CSV files"""
        if not self.experiment_data:
            print("No data to save!")
            return
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_file = os.path.join(
            self.participant_dir,
            f"experiment_results_{self.participant_id}_{timestamp}.json"
        )
        
        with open(json_file, 'w') as f:
            json.dump({
                'participant_id': self.participant_id,
                'experiment_data': self.experiment_data,
                'participant_metadata': self.participant_metadata,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        # Save summary results as CSV
        csv_file = os.path.join(
            self.participant_dir,
            f"experiment_results_{self.participant_id}_{timestamp}.csv"
        )
        
        # Prepare rows for CSV
        csv_rows = []
        for result in self.experiment_data:
            # Get the actual values for default parameters
            resolution_str = result['resolution']
            if result['resolution'] == "Default":
                resolution_str = f"{self.camera_capabilities['default_params']['resolution'][0]}x{self.camera_capabilities['default_params']['resolution'][1]}"
            
            # For frame rate, show the actual value instead of 'Default'
            if result['frame_rate'] is None:
                frame_rate_str = f"{self.camera_capabilities['default_params']['frame_rate']:.1f}"
            else:
                frame_rate_str = f"{result['frame_rate']}"
            
            # For autofocus, show the actual value instead of 'Default'
            if result['autofocus'] is None:
                autofocus_str = 'On' if self.camera_capabilities['default_params']['autofocus'] else 'Off'
            else:
                autofocus_str = 'On' if result['autofocus'] else 'Off'
            
            csv_rows.append({
                'Test ID': result['test_id'],
                'Resolution': resolution_str,
                'Frame Rate': frame_rate_str,
                'Autofocus': autofocus_str,
                'Calibration Points': result['calibration_points'],
                'Measured Accuracy (°)': round(result['measured_accuracy'], 2) if not math.isinf(result['measured_accuracy']) else "N/A",
                'Precision (°)': round(result['precision'], 2) if not math.isinf(result['precision']) else "N/A",
                'Calibration Time (s)': round(result['calibration_time'], 2),
                'Participant ID': result['participant_id'],
                'Timestamp': result['timestamp']
            })
        
        # Sort by Test ID to maintain order
        csv_rows.sort(key=lambda x: x['Test ID'])
        
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to:")
        print(f"- CSV: {csv_file}")
        print(f"- JSON: {json_file}")
        
        # Update master summary file
        self._update_master_summary(csv_rows)
    
    def _update_master_summary(self, new_results):
        """Update master summary CSV with results from all participants"""
        master_csv = os.path.join(self.base_output_dir, "master_summary.csv")
        
        # Check if master summary exists
        if os.path.exists(master_csv):
            # Load existing data
            try:
                master_df = pd.read_csv(master_csv)
                
                # Remove any existing data for this participant
                master_df = master_df[master_df['Participant ID'] != self.participant_id]
                
                # Add new data
                new_df = pd.DataFrame(new_results)
                master_df = pd.concat([master_df, new_df], ignore_index=True)
                
            except Exception as e:
                print(f"Error updating master summary: {e}")
                # Create new dataframe if error reading existing file
                master_df = pd.DataFrame(new_results)
        else:
            # Create new master summary
            master_df = pd.DataFrame(new_results)
        
        # Save updated master summary
        master_df.to_csv(master_csv, index=False)
        print(f"Master summary updated: {master_csv}")
    
    def generate_analysis_report(self):
        """Generate analysis report with statistics"""
        if not self.experiment_data:
            print("No data to analyze!")
            return
        
        # Create report file
        report_file = os.path.join(
            self.participant_dir,
            f"analysis_report_{self.participant_id}.txt"
        )
        
        # Extract metrics for analysis
        valid_results = [r for r in self.experiment_data 
                        if not math.isinf(r['measured_accuracy']) and not math.isnan(r['measured_accuracy'])]
        
        if not valid_results:
            with open(report_file, 'w') as f:
                f.write(f"Analysis Report for Participant {self.participant_id}\n")
                f.write("No valid results for analysis!\n")
            print(f"Analysis report saved to {report_file}")
            return
        
        # Calculate overall statistics
        accuracies = [r['measured_accuracy'] for r in valid_results]
        precisions = [r['precision'] for r in valid_results]
        calibration_times = [r['calibration_time'] for r in valid_results]
        
        overall_stats = {
            'mean_accuracy': np.mean(accuracies),
            'median_accuracy': np.median(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'mean_precision': np.mean(precisions),
            'median_precision': np.median(precisions),
            'std_precision': np.std(precisions),
            'mean_calibration_time': np.mean(calibration_times),
            'median_calibration_time': np.median(calibration_times),
            'std_calibration_time': np.std(calibration_times)
        }
        
        # Group by resolution for resolution-specific analysis
        resolution_groups = {}
        for r in valid_results:
            res_key = r['resolution']
            if res_key not in resolution_groups:
                resolution_groups[res_key] = []
            resolution_groups[res_key].append(r)
        
        # Group by calibration points
        calibration_groups = {}
        for r in valid_results:
            cal_key = r['calibration_points']
            if cal_key not in calibration_groups:
                calibration_groups[cal_key] = []
            calibration_groups[cal_key].append(r)
        
        # Write report
        with open(report_file, 'w') as f:
            f.write(f"Analysis Report for Participant {self.participant_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("=================\n")
            f.write(f"Total valid tests: {len(valid_results)}\n")
            f.write(f"Mean accuracy: {overall_stats['mean_accuracy']:.2f}°\n")
            f.write(f"Median accuracy: {overall_stats['median_accuracy']:.2f}°\n")
            f.write(f"Accuracy std dev: {overall_stats['std_accuracy']:.2f}°\n")
            f.write(f"Min accuracy: {overall_stats['min_accuracy']:.2f}°\n")
            f.write(f"Max accuracy: {overall_stats['max_accuracy']:.2f}°\n\n")
            
            f.write(f"Mean precision: {overall_stats['mean_precision']:.2f}°\n")
            f.write(f"Median precision: {overall_stats['median_precision']:.2f}°\n")
            f.write(f"Precision std dev: {overall_stats['std_precision']:.2f}°\n\n")
            
            f.write(f"Mean calibration time: {overall_stats['mean_calibration_time']:.2f}s\n")
            f.write(f"Median calibration time: {overall_stats['median_calibration_time']:.2f}s\n")
            f.write(f"Calibration time std dev: {overall_stats['std_calibration_time']:.2f}s\n\n")
            
            # Resolution-specific analysis
            f.write("RESOLUTION-SPECIFIC ANALYSIS\n")
            f.write("============================\n")
            for res, results in resolution_groups.items():
                f.write(f"Resolution: {res}\n")
                f.write(f"  Tests: {len(results)}\n")
                
                res_accuracies = [r['measured_accuracy'] for r in results]
                res_precisions = [r['precision'] for r in results]
                res_cal_times = [r['calibration_time'] for r in results]
                
                f.write(f"  Mean accuracy: {np.mean(res_accuracies):.2f}°\n")
                f.write(f"  Mean precision: {np.mean(res_precisions):.2f}°\n")
                f.write(f"  Mean calibration time: {np.mean(res_cal_times):.2f}s\n\n")
            
            # Calibration points analysis
            f.write("CALIBRATION POINTS ANALYSIS\n")
            f.write("===========================\n")
            for points, results in calibration_groups.items():
                f.write(f"Calibration points: {points}\n")
                f.write(f"  Tests: {len(results)}\n")
                
                cal_accuracies = [r['measured_accuracy'] for r in results]
                cal_precisions = [r['precision'] for r in results]
                cal_times = [r['calibration_time'] for r in results]
                
                f.write(f"  Mean accuracy: {np.mean(cal_accuracies):.2f}°\n")
                f.write(f"  Mean precision: {np.mean(cal_precisions):.2f}°\n")
                f.write(f"  Mean calibration time: {np.mean(cal_times):.2f}s\n\n")
            
            # Best configurations
            f.write("BEST CONFIGURATIONS\n")
            f.write("===================\n")
            
            # Sort by accuracy
            sorted_by_accuracy = sorted(valid_results, key=lambda x: x['measured_accuracy'])
            best_accuracy = sorted_by_accuracy[0]
            f.write(f"Best accuracy: {best_accuracy['measured_accuracy']:.2f}° (Test {best_accuracy['test_id']})\n")
            f.write(f"  Resolution: {best_accuracy['resolution']}\n")
            f.write(f"  Frame rate: {best_accuracy['frame_rate'] if best_accuracy['frame_rate'] is not None else self.camera_capabilities['default_params']['frame_rate']:.1f}fps\n")
            f.write(f"  Autofocus: {'On' if best_accuracy['autofocus'] else 'Off' if best_accuracy['autofocus'] is not None else 'Default'}\n")
            f.write(f"  Calibration points: {best_accuracy['calibration_points']}\n\n")
            
            # Sort by precision
            sorted_by_precision = sorted(valid_results, key=lambda x: x['precision'])
            best_precision = sorted_by_precision[0]
            f.write(f"Best precision: {best_precision['precision']:.2f}° (Test {best_precision['test_id']})\n")
            f.write(f"  Resolution: {best_precision['resolution']}\n")
            f.write(f"  Frame rate: {best_precision['frame_rate'] if best_precision['frame_rate'] is not None else self.camera_capabilities['default_params']['frame_rate']:.1f}fps\n")
            f.write(f"  Autofocus: {'On' if best_precision['autofocus'] else 'Off' if best_precision['autofocus'] is not None else 'Default'}\n")
            f.write(f"  Calibration points: {best_precision['calibration_points']}\n\n")
            
            # Sort by calibration time (fastest)
            sorted_by_time = sorted(valid_results, key=lambda x: x['calibration_time'])
            fastest_calibration = sorted_by_time[0]
            f.write(f"Fastest calibration: {fastest_calibration['calibration_time']:.2f}s (Test {fastest_calibration['test_id']})\n")
            f.write(f"  Resolution: {fastest_calibration['resolution']}\n")
            f.write(f"  Frame rate: {fastest_calibration['frame_rate'] if fastest_calibration['frame_rate'] is not None else self.camera_capabilities['default_params']['frame_rate']:.1f}fps\n")
            f.write(f"  Autofocus: {'On' if fastest_calibration['autofocus'] else 'Off' if fastest_calibration['autofocus'] is not None else 'Default'}\n")
            f.write(f"  Calibration points: {fastest_calibration['calibration_points']}\n")
        
        print(f"Analysis report saved to {report_file}")
    
    @staticmethod
    def cleanup_incomplete_data(base_dir="experiment_data"):
        """Clean up incomplete participant data"""
        if not os.path.exists(base_dir):
            print("No experiment data directory found.")
            return
        
        participants = GazeTrackingExperiment.list_participants(base_dir)
        
        if not participants:
            print("No participant data found.")
            return
        
        print("\nIncomplete Participant Data:")
        incomplete = [p for p in participants if p['status'] != "Completed"]
        
        if not incomplete:
            print("No incomplete participant data found.")
            return
        
        for i, p in enumerate(incomplete):
            print(f"{i+1}. Participant {p['id']} - {p['status']} ({p['progress']:.1f}%)")
        
        print("\nOptions:")
        print("1. Keep all incomplete data")
        print("2. Archive incomplete data")
        print("3. Delete incomplete data")
        print("4. Select specific participants")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            print("Keeping all incomplete data.")
            return
        elif choice == '2':
            # Create archive directory
            archive_dir = os.path.join(base_dir, "archived_data")
            os.makedirs(archive_dir, exist_ok=True)
            
            for p in incomplete:
                participant_dir = p['directory']
                archive_participant_dir = os.path.join(archive_dir, f"participant_{p['id']}")
                
                # Move directory to archive
                import shutil
                shutil.move(participant_dir, archive_participant_dir)
                print(f"Archived participant {p['id']} data to {archive_participant_dir}")
            
            print(f"All incomplete data archived to {archive_dir}")
            
        elif choice == '3':
            confirm = input("Are you sure you want to DELETE incomplete data? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Deletion cancelled.")
                return
            
            for p in incomplete:
                participant_dir = p['directory']
                
                # Delete directory
                import shutil
                shutil.rmtree(participant_dir)
                print(f"Deleted participant {p['id']} data")
            
            print("All incomplete data deleted.")
            
        elif choice == '4':
            selected = input("Enter participant numbers to process (comma-separated): ").strip()
            try:
                selected_indices = [int(x.strip()) - 1 for x in selected.split(',')]
                selected_participants = [incomplete[i] for i in selected_indices if 0 <= i < len(incomplete)]
                
                if not selected_participants:
                    print("No valid participants selected.")
                    return
                
                print("\nSelected participants:")
                for p in selected_participants:
                    print(f"Participant {p['id']} - {p['status']} ({p['progress']:.1f}%)")
                
                print("\nOptions:")
                print("1. Keep selected data")
                print("2. Archive selected data")
                print("3. Delete selected data")
                
                action = input("Enter your choice (1-3): ").strip()
                
                if action == '1':
                    print("Keeping selected data.")
                    return
                elif action == '2':
                    # Create archive directory
                    archive_dir = os.path.join(base_dir, "archived_data")
                    os.makedirs(archive_dir, exist_ok=True)
                    
                    for p in selected_participants:
                        participant_dir = p['directory']
                        archive_participant_dir = os.path.join(archive_dir, f"participant_{p['id']}")
                        
                        # Move directory to archive
                        import shutil
                        shutil.move(participant_dir, archive_participant_dir)
                        print(f"Archived participant {p['id']} data to {archive_participant_dir}")
                    
                    print(f"Selected data archived to {archive_dir}")
                    
                elif action == '3':
                    confirm = input("Are you sure you want to DELETE selected data? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print("Deletion cancelled.")
                        return
                    
                    for p in selected_participants:
                        participant_dir = p['directory']
                        
                        # Delete directory
                        import shutil
                        shutil.rmtree(participant_dir)
                        print(f"Deleted participant {p['id']} data")
                    
                    print("Selected data deleted.")
                    
                else:
                    print("Invalid choice.")
                    
            except Exception as e:
                print(f"Error processing selection: {e}")
        else:
            print("Invalid choice.")
    
    @staticmethod
    def create_new_participant(base_dir="experiment_data"):
        """Create a new participant ID"""
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        
        # Get existing participant IDs
        existing_ids = []
        for item in os.listdir(base_dir):
            if item.startswith("participant_"):
                try:
                    participant_id = item.replace("participant_", "")
                    existing_ids.append(participant_id)
                except:
                    pass
        
        # Generate new ID options
        next_numeric_id = "001"
        if existing_ids:
            try:
                numeric_ids = [int(pid) for pid in existing_ids if pid.isdigit()]
                if numeric_ids:
                    next_numeric_id = f"{max(numeric_ids) + 1:03d}"
            except:
                pass
        
        print("\nCreate New Participant")
        print("=======================")
        print("Options:")
        print(f"1. Use next numeric ID: {next_numeric_id}")
        print("2. Enter custom ID")
        
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == '1':
            new_id = next_numeric_id
        elif choice == '2':
            new_id = input("Enter custom participant ID: ").strip()
            
            # Check if ID already exists
            if new_id in existing_ids:
                print(f"Warning: Participant ID '{new_id}' already exists!")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Cancelled.")
                    return None
        else:
            print("Invalid choice.")
            return None
        
        print(f"Created new participant ID: {new_id}")
        return new_id
    
    @staticmethod
    def continue_experiment(base_dir="experiment_data"):
        """Continue an existing participant's experiment"""
        participants = GazeTrackingExperiment.list_participants(base_dir)
        
        if not participants:
            print("No participant data found.")
            return None
        
        # Filter to show only incomplete participants
        incomplete = [p for p in participants if p['status'] != "Completed"]
        
        if not incomplete:
            print("No incomplete experiments found.")
            print("All participants have completed their experiments.")
            return None
        
        print("\nIncomplete Experiments:")
        for i, p in enumerate(incomplete):
            print(f"{i+1}. Participant {p['id']} - {p['status']} ({p['progress']:.1f}%)")
        
        choice = input("\nEnter participant number to continue, or 0 to cancel: ").strip()
        
        try:
            choice_num = int(choice)
            if choice_num == 0:
                print("Cancelled.")
                return None
            elif 1 <= choice_num <= len(incomplete):
                selected = incomplete[choice_num - 1]
                print(f"Selected participant {selected['id']}")
                return selected['id']
            else:
                print("Invalid selection.")
                return None
        except:
            print("Invalid input.")
            return None
    
    @staticmethod
    def view_participant_info(base_dir="experiment_data"):
        """View detailed information about a participant"""
        participants = GazeTrackingExperiment.list_participants(base_dir)
        
        if not participants:
            print("No participant data found.")
            return
        
        print("\nAll Participants:")
        for i, p in enumerate(participants):
            print(f"{i+1}. Participant {p['id']} - {p['status']} ({p['progress']:.1f}%)")
        
        choice = input("\nEnter participant number to view details, or 0 to cancel: ").strip()
        
        try:
            choice_num = int(choice)
            if choice_num == 0:
                print("Cancelled.")
                return
            elif 1 <= choice_num <= len(participants):
                selected = participants[choice_num - 1]
                GazeTrackingExperiment._view_participant_details(selected)
            else:
                print("Invalid selection.")
        except:
            print("Invalid input.")
    
    @staticmethod
    def _view_participant_details(participant):
        """View detailed information about a specific participant"""
        participant_dir = participant['directory']
        
        print("\n" + "="*60)
        print(f"PARTICIPANT {participant['id']} DETAILS")
        print("="*60)
        print(f"Status: {participant['status']}")
        print(f"Progress: {participant['progress']:.1f}%")
        print(f"Directory: {participant_dir}")
        
        # Check for state file
        state_file = os.path.join(participant_dir, "experiment_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                completed = len(state.get('completed_conditions', []))
                total = state.get('total_conditions', 0)
                timestamp = state.get('timestamp', 'Unknown')
                
                print(f"\nExperiment State:")
                print(f"- Last updated: {timestamp}")
                print(f"- Completed conditions: {completed}/{total}")
                
                # Show recent test conditions
                experiment_data = state.get('experiment_data', [])
                if experiment_data:
                    print("\nRecent Test Conditions:")
                    for i, result in enumerate(experiment_data[-3:]):  # Show last 3 results
                        print(f"  {i+1}. Test {result.get('test_id', 'Unknown')}:")
                        print(f"     Resolution: {result.get('resolution', 'Unknown')}")
                        print(f"     Frame Rate: {result.get('frame_rate', 'Unknown')}")
                        print(f"     Autofocus: {result.get('autofocus', 'Unknown')}")
                        print(f"     Calibration Points: {result.get('calibration_points', 'Unknown')}")
                        print(f"     Accuracy: {result.get('measured_accuracy', 'Unknown')}")
            except Exception as e:
                print(f"Error reading state file: {e}")
        
        # Check for result files
        result_files = [f for f in os.listdir(participant_dir) 
                       if f.startswith("experiment_results_") and f.endswith(".csv")]
        
        if result_files:
            print("\nResult Files:")
            for i, f in enumerate(result_files):
                file_path = os.path.join(participant_dir, f)
                file_size = os.path.getsize(file_path) / 1024  # KB
                file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {i+1}. {f} ({file_size:.1f} KB, {file_date})")
        
        # Check for analysis report
        report_file = os.path.join(participant_dir, f"analysis_report_{participant['id']}.txt")
        if os.path.exists(report_file):
            print("\nAnalysis Report:")
            print(f"  {report_file}")
            
            view_report = input("\nView analysis report? (y/n): ").strip().lower()
            if view_report == 'y':
                try:
                    with open(report_file, 'r') as f:
                        report_content = f.read()
                    
                    print("\n" + "="*60)
                    print("ANALYSIS REPORT")
                    print("="*60)
                    print(report_content)
                except Exception as e:
                    print(f"Error reading report: {e}")
        
        print("\nPress Enter to continue...")
        input()

def main():
    """Main function to run the experiment"""
    print("\n" + "="*60)
    print("GAZE TRACKING EXPERIMENT")
    print("="*60)
    print("This program tests gaze tracking performance under different camera settings.")
    
    # Check for required files
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("\nERROR: Required file 'shape_predictor_68_face_landmarks.dat' not found!")
        print("Please download it from: https://github.com/davisking/dlib-models")
        print("\nPress Enter to exit...")
        input()
        return
    
    # Create base directory
    base_dir = "experiment_data"
    os.makedirs(base_dir, exist_ok=True)
    
    # Detect default camera parameters for information
    try:
        print("\nDetecting camera capabilities...")
        temp_experiment = GazeTrackingExperiment("temp")
        camera_capabilities = temp_experiment.camera_capabilities
        
        print(f"\nDetected Camera:")
        print(f"- Max Resolution: {camera_capabilities['max_resolution'][0]}x{camera_capabilities['max_resolution'][1]}")
        print(f"- Max Frame Rate: {camera_capabilities['max_fps']:.1f}fps")
        print(f"- Autofocus Supported: {'Yes' if camera_capabilities['autofocus_supported'] else 'No'}")
        print(f"- Default Resolution: {camera_capabilities['default_params']['resolution'][0]}x{camera_capabilities['default_params']['resolution'][1]}")
        print(f"- Default Frame Rate: {camera_capabilities['default_params']['frame_rate']:.1f}fps")
        print(f"- Default Autofocus: {'On' if camera_capabilities['default_params']['autofocus'] else 'Off'}")
        
        # Clean up temporary experiment
        import shutil
        shutil.rmtree(os.path.join(base_dir, "participant_temp"), ignore_errors=True)
        
    except Exception as e:
        print(f"Warning: Could not detect camera capabilities: {e}")
        print("The experiment will still run, but with limited camera parameter options.")
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Start new experiment (new participant)")
        print("2. Continue experiment (existing participant)")
        print("3. View participant details")
        print("4. Clean up incomplete data")
        print("5. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            participant_id = GazeTrackingExperiment.create_new_participant(base_dir)
            if participant_id:
                experiment = GazeTrackingExperiment(participant_id, base_dir)
                experiment.run_full_experiment()
        
        elif choice == '2':
            participant_id = GazeTrackingExperiment.continue_experiment(base_dir)
            if participant_id:
                experiment = GazeTrackingExperiment(participant_id, base_dir)
                experiment.run_full_experiment()
        
        elif choice == '3':
            GazeTrackingExperiment.view_participant_info(base_dir)
        
        elif choice == '4':
            GazeTrackingExperiment.cleanup_incomplete_data(base_dir)
        
        elif choice == '5':
            print("\nExiting program. Thank you!")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
