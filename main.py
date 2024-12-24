import cv2
import numpy as np
from utils.weapon_detector import WeaponDetector
from analysis.llm_agent import SceneAnalyzer
import os
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FrameData:
    frame: np.ndarray
    frame_number: int
    timestamp: float


class ThreatAnalyzer:
    def __init__(self, api_key: str):
        self.analyzer = SceneAnalyzer(api_key=api_key)
        self.frame_queue = queue.Queue()
        self.analysis_results = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        self.analyses_list = []
        self.current_analysis_index = 0
        self.auto_advance = True

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._analysis_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()

    def _analysis_worker(self):
        while self.is_running:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                result = self.analyzer.analyze_threat(frame_data.frame, frame_data.frame_number)
                if result != {"status": "Analysis already completed"}: self.analysis_results.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in analysis worker: {str(e)}")

    def submit_frame(self, frame, frame_number):
        self.frame_queue.put(FrameData(
            frame=frame.copy(),
            frame_number=frame_number,
            timestamp=time.time()
        ))

    def get_latest_result(self) -> Optional[dict]:
        try:
            result = self.analysis_results.get_nowait()
            if result:
                self.analyses_list.append(result)
                # Auto-advance to the latest analysis
                if self.auto_advance:
                    self.current_analysis_index = len(self.analyses_list) - 1
            return result
        except queue.Empty:
            return None

    def get_current_analysis(self) -> Optional[dict]:
        """Get the currently selected analysis based on index"""
        if self.analyses_list:
            analysis = self.analyses_list[self.current_analysis_index]
            # Add navigation information to the analysis
            analysis['navigation'] = {
                'current_index': self.current_analysis_index,
                'total_analyses': len(self.analyses_list)
            }
            if isinstance(analysis.get('analysis'), dict):
                threat_level = analysis['analysis'].get('threat_level', '')
                if threat_level == 'HIGH':
                    analysis['show_alert'] = True
            return analysis
        return None

    def next_analysis(self):
        if self.analyses_list:
            self.current_analysis_index = (self.current_analysis_index + 1) % len(self.analyses_list)

    def prev_analysis(self):
        if self.analyses_list:
            self.current_analysis_index = (self.current_analysis_index - 1) % len(self.analyses_list)


def draw_robot_icon(image, x, y, size=30):
    """Draw a simple robot icon using OpenCV shapes"""
    # Head (rectangle)
    cv2.rectangle(image, (x, y), (x + size, y + size), (255, 255, 255), 1)

    # Eyes (circles)
    eye_size = size // 8
    left_eye_x = x + size // 4
    right_eye_x = x + 3 * size // 4
    eyes_y = y + size // 3
    cv2.circle(image, (left_eye_x, eyes_y), eye_size, (255, 255, 255), -1)
    cv2.circle(image, (right_eye_x, eyes_y), eye_size, (255, 255, 255), -1)

    # Mouth (line)
    mouth_y = y + 2 * size // 3
    cv2.line(image, (x + size // 4, mouth_y), (x + 3 * size // 4, mouth_y), (255, 255, 255), 1)

    # Antenna
    cv2.line(image, (x + size // 2, y), (x + size // 2, y - size // 4), (255, 255, 255), 1)
    cv2.circle(image, (x + size // 2, y - size // 4), 2, (255, 255, 255), -1)


def draw_text_with_background(image: np.ndarray, text: str, position: tuple,
                            font_scale: float = 0.6, thickness: int = 2,  # Increased thickness
                            text_color: tuple = (255, 255, 255),
                            bg_color: tuple = (0, 0, 0)) -> int:
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Add stronger contrast and larger padding
    (text_w, text_h) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    x, y = position
    # Increased padding for better readability
    cv2.rectangle(image, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10),
                 bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
    return y + text_h + 15  # Increased line spacing


def create_info_overlay(frame: np.ndarray, analysis_result: Dict[str, Any], weapon_detected: bool) -> np.ndarray:
    """Create an enhanced overlay with weapon status and analysis carousel"""
    overlay = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    # Calculate carousel dimensions and position with increased size
    carousel_width = min(500, int(frame_width * 0.4))  # Wider carousel
    carousel_height = min(250, int(frame_height * 0.4))  # Taller carousel
    carousel_x = frame_width - carousel_width - 400
    carousel_y = 10

    # Draw weapon detection status (top left)
    status_color = (0, 0, 255) if weapon_detected else (0, 255, 0)
    status_text = "WEAPON DETECTED" if weapon_detected else "SECURE"
    draw_text_with_background(overlay, status_text,
                              (20, 120), 0.7, 2,
                              (255, 255, 255), status_color)

    # Create semi-transparent carousel background
    overlay_bg = overlay.copy()
    cv2.rectangle(overlay_bg,
                  (carousel_x, carousel_y),
                  (carousel_x + carousel_width, carousel_y + carousel_height),
                  (0, 0, 0), -1)

    # Blend background
    alpha = 0.7
    cv2.addWeighted(overlay_bg, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw carousel border
    cv2.rectangle(overlay,
                  (carousel_x, carousel_y),
                  (carousel_x + carousel_width, carousel_y + carousel_height),
                  (255, 255, 255), 1)

    # Draw robot icon at top left of carousel
    robot_size = 30
    robot_x = carousel_x + 10
    robot_y = carousel_y + 10
    draw_robot_icon(overlay, robot_x, robot_y, robot_size)

    # Draw navigation controls below carousel
    control_y = carousel_y + carousel_height + 10
    cv2.rectangle(overlay,
                  (carousel_x, control_y),
                  (carousel_x + 30, control_y + 20),
                  (255, 255, 255), 1)
    cv2.putText(overlay, "A", (carousel_x + 10, control_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(overlay,
                  (carousel_x + carousel_width - 30, control_y),
                  (carousel_x + carousel_width, control_y + 20),
                  (255, 255, 255), 1)
    cv2.putText(overlay, "D", (carousel_x + carousel_width - 20, control_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if analysis_result:
        status = analysis_result.get('status', '').upper()
        analysis = analysis_result.get('analysis', '')

        content_x = carousel_x + 60  # Start text after robot
        header_y = carousel_y + 25

        # Display status header
        draw_text_with_background(overlay, f"STATUS: {status}",
                                  (content_x, header_y),
                                  0.5, 1, (255, 255, 255), (0, 0, 0))

        content_y = header_y + 30
        max_width = carousel_width - 80  # Ensure text fits within carousel

        if isinstance(analysis, str):
            # Split and process the text
            sections = analysis.split('\n')
            current_y = content_y

            for section in sections:
                if not section.strip():
                    continue

                # Check if this is a header line
                is_header = any(key in section for key in ["THREAT:", "SUSPECT:", "Risk Level:"])
                x_position = content_x if is_header else content_x + 10

                # Word wrap the text
                words = section.strip().split()
                line = ""
                for word in words:
                    test_line = line + " " + word if line else word
                    (line_width, _) = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

                    if line_width < (carousel_width - 90):
                        line = test_line
                    else:
                        if current_y + 20 < carousel_y + carousel_height:
                            current_y = draw_text_with_background(overlay, line,
                                                                  (x_position, current_y),
                                                                  0.4, 1, (255, 255, 255), (0, 0, 0))
                        line = word

                if line and current_y + 20 < carousel_y + carousel_height:
                    current_y = draw_text_with_background(overlay, line,
                                                          (x_position, current_y),
                                                          0.4, 1, (255, 255, 255), (0, 0, 0))

            # Display navigation counter
            if analysis_result.get('navigation'):
                nav = analysis_result['navigation']
                nav_text = f"Analysis {nav['current_index'] + 1}/{nav['total_analyses']}"
                draw_text_with_background(overlay, nav_text,
                                          (carousel_x + carousel_width // 2 - 50, control_y + 15),
                                          0.5, 1, (255, 255, 255), (0, 0, 0))

        elif isinstance(analysis, dict) and status == "FINAL_ASSESSMENT":
            current_y = content_y + 20
            threat_level = analysis.get('threat_level', 'UNKNOWN')
            location = analysis.get('location', '')
            recommendation = analysis.get('final_recommendation', '')

            # Display threat level
            draw_text_with_background(overlay, f"THREAT LEVEL: {threat_level}",
                                      (content_x, current_y),
                                      0.6, 2, (255, 255, 255),
                                      (0, 0, 255) if threat_level == "HIGH" else (255, 165, 0))
            current_y += 30

            # Display location if available
            if location:
                draw_text_with_background(overlay, f"LOCATION: {location}",
                                          (content_x, current_y),
                                          0.4, 1, (255, 255, 255), (0, 0, 0))
                current_y += 25

            # Display recommendation with word wrap
            if recommendation:
                words = recommendation.split()
                line = ""
                for word in words:
                    test_line = line + " " + word if line else word
                    (line_width, _) = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

                    if line_width < (carousel_width - 90):
                        line = test_line
                    else:
                        if current_y + 20 < carousel_y + carousel_height:
                            current_y = draw_text_with_background(overlay, line,
                                                                  (content_x, current_y),
                                                                  0.4, 1, (255, 255, 255), (0, 0, 0))
                        line = word

                if line and current_y + 20 < carousel_y + carousel_height:
                    draw_text_with_background(overlay, line,
                                              (content_x, current_y),
                                              0.4, 1, (255, 255, 255), (0, 0, 0))

    return overlay


def process_video(video_path: str, model_path: str = 'safeschool/train/weights/best.pt',
                  conf_threshold: float = 0.75):
    detector = WeaponDetector(model_path, conf_threshold)
    threat_analyzer = ThreatAnalyzer(api_key=os.getenv("OPENAI_API_KEY"))
    threat_analyzer.start()

    while True:  # Outer loop for video replay
        cap = cv2.VideoCapture(video_path)
        weapon_detection_started = False
        frame_count = 0
        consecutive_detections = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
                    continue

                frame_count += 1
                results = detector.predict(frame)
                weapon_detected = any(cls in [0, 1, 2] for cls in results.boxes.cls)

                if weapon_detected:
                    consecutive_detections += 1
                    if not weapon_detection_started and consecutive_detections >= 3:
                        print("Starting threat analysis...")
                        weapon_detection_started = True
                    if weapon_detection_started:
                        threat_analyzer.submit_frame(frame, frame_count)
                else:
                    consecutive_detections = 0

                # Get any new analysis results
                threat_analyzer.get_latest_result()

                # Get current analysis to display
                current_analysis = threat_analyzer.get_current_analysis()

                display_frame = results.plot()
                display_frame = create_info_overlay(display_frame, current_analysis,
                                                    weapon_detected)

                cv2.imshow('SafeSchool Detection', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
                elif key == ord('a'):  # Previous analysis
                    threat_analyzer.prev_analysis()
                elif key == ord('d'):  # Next analysis
                    threat_analyzer.next_analysis()

        except KeyboardInterrupt:
            break
        finally:
            cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/input1.mp4"
    process_video(video_path)