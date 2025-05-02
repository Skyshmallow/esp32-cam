import cv2
import numpy as np
import requests
import time
import urllib.request
import threading
from io import BytesIO
import os
from datetime import datetime

# ESP32-CAM API endpoints
CAMERA_IP = "192.168.46.186"
CAMERA_CONTROL_URL = f"http://{CAMERA_IP}/action"
VIDEO_STREAM_URL = f"http://{CAMERA_IP}:81/stream"

# Movement parameters
MOVE_COOLDOWN = 0.12  # Seconds between camera movements (reduced from 0.5)
CENTER_THRESHOLD = 0.1  # Distance from center before moving camera (reduced from 0.15)
MOVEMENT_SMOOTHING = 2  # Number of frames to consider for smoothing movement

# Global stop event to allow external termination
stop_tracking_event = threading.Event()


class MjpegStreamReader:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.stopped = False
        self.thread = None
        self.last_frame_time = 0

    def start(self):
        """Start the MJPEG stream reader thread"""
        print(f"Starting MJPEG stream reader for {self.url}")
        self.thread = threading.Thread(target=self.read_stream, daemon=True)
        self.thread.start()
        return self

    def read_stream(self):
        """Read MJPEG stream directly using urllib"""
        try:
            # Create request with timeout
            stream = urllib.request.urlopen(self.url, timeout=10)
            bytes_data = bytes()

            # Read stream until stopped
            while not self.stopped:
                bytes_data += stream.read(1024)
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end

                # If we found a complete JPEG frame
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b + 2]  # Extract the JPEG
                    bytes_data = bytes_data[b + 2:]  # Remove processed data

                    # Decode JPEG to OpenCV format
                    try:
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.frame = frame
                            self.last_frame_time = time.time()
                    except Exception as e:
                        print(f"Error decoding frame: {e}")
        except Exception as e:
            print(f"Stream reader error: {e}")
            self.stopped = True

    def read(self):
        """Return the current frame"""
        return self.frame

    def is_active(self):
        """Check if stream is active and receiving frames"""
        if self.frame is None:
            return False
        # Consider stream dead if no frames for 5 seconds
        return (time.time() - self.last_frame_time) < 5

    def stop(self):
        """Stop the stream reader thread"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1)


class HeadTrackingCamera:
    def __init__(self, follow_person=True, record=False):
        # Track and record settings
        self.follow_person = follow_person
        self.record = record

        # Video recording properties
        self.video_writer = None
        self.recording_started = False
        self.recording_path = ""
        self.fps = 20  # Target frames per second for recording
        self.last_frame_write_time = 0
        self.frame_interval = 1.0 / self.fps  # Time between frame writes

        # Initialize face detection cascade classifier
        print("Initializing face detector...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # For profile face detection (side view)
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        # For smoothing movements
        self.last_move_time = 0
        self.running = False
        self.last_direction = None
        self.consecutive_same_direction = 0

        # For position smoothing
        self.prev_head_positions = []

        # Initialize stream reader
        self.stream_reader = None
        self.frame_height = 0
        self.frame_width = 0

    def create_video_writer(self):
        """Initialize the video writer with appropriate settings"""
        if not os.path.exists('recordings'):
            os.makedirs('recordings')

        # Create a filename with current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = f"recordings/recording_{timestamp}.mp4"

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec

        # Create the video writer
        self.video_writer = cv2.VideoWriter(
            self.recording_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        if self.video_writer.isOpened():
            print(f"Recording started: {self.recording_path}")
            self.recording_started = True
            return True
        else:
            print("Failed to create video writer")
            return False

    def connect_to_stream(self):
        """Connect to the ESP32-CAM video stream"""
        print(f"Connecting to ESP32-CAM stream at {VIDEO_STREAM_URL}")

        # Close previous stream if exists
        if self.stream_reader is not None:
            self.stream_reader.stop()

        # Start new stream reader
        self.stream_reader = MjpegStreamReader(VIDEO_STREAM_URL).start()

        # Wait for first frame
        max_wait = 10  # Maximum seconds to wait
        start = time.time()
        while self.stream_reader.read() is None:
            if time.time() - start > max_wait:
                raise Exception(f"Timeout waiting for first frame from stream")
            time.sleep(0.5)

        # Get frame dimensions
        frame = self.stream_reader.read()
        self.frame_height, self.frame_width = frame.shape[:2]
        print(f"Connected to stream. Frame size: {self.frame_width}x{self.frame_height}")

        # Initialize video writer if recording is enabled
        if self.record and not self.recording_started:
            self.create_video_writer()

    def move_camera(self, direction):
        """Send command to move the ESP32-CAM in specified direction (up, down, left, right)"""
        if not self.follow_person:
            return False

        current_time = time.time()

        # Apply cooldown to prevent excessive movements
        if current_time - self.last_move_time < MOVE_COOLDOWN:
            return False

        # Adaptive movement - slow down if repeatedly moving in same direction
        if direction == self.last_direction:
            self.consecutive_same_direction += 1
            # Increase cooldown for repeated movements progressively
            if self.consecutive_same_direction > 3:
                cooldown_multiplier = min(1 + (self.consecutive_same_direction * 0.2), 3.0)
                if current_time - self.last_move_time < MOVE_COOLDOWN * cooldown_multiplier:
                    return False
        else:
            self.consecutive_same_direction = 0

        self.last_direction = direction

        # Send the movement command to the ESP32-CAM
        try:
            url = f"{CAMERA_CONTROL_URL}?go={direction}"
            print(f"Moving camera: {direction}")
            response = requests.get(url, timeout=1)

            if response.status_code == 200:
                self.last_move_time = current_time
                return True
            else:
                print(f"Failed to move camera. Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error moving camera: {e}")
            return False

    def detect_heads(self, frame):
        """Detect faces (heads) in the frame using Haar cascades"""
        # Convert to grayscale for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalize histogram to improve detection in different lighting
        gray = cv2.equalizeHist(gray)

        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # If no frontal faces found, try profile faces
        if len(faces) == 0:
            faces = self.profile_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Try the other side profile by flipping the image horizontally
            if len(faces) == 0:
                flipped = cv2.flip(gray, 1)
                profile_faces = self.profile_face_cascade.detectMultiScale(
                    flipped,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                # Convert coordinates back to original image
                for i, (x, y, w, h) in enumerate(profile_faces):
                    # Flip x-coordinate: new_x = img_width - x - width
                    profile_faces[i][0] = self.frame_width - x - w

                faces = profile_faces

        if len(faces) > 0:
            print(f"Detected {len(faces)} faces")
        return faces

    def get_largest_head(self, faces):
        """Find the largest face (likely closest to camera)"""
        if len(faces) == 0:
            return None

        # Get the box with largest area
        largest_face = max(faces, key=lambda box: box[2] * box[3])
        return largest_face

    def smooth_head_position(self, face_box):
        """Apply smoothing to head position to reduce jitter"""
        # Add current position to history
        self.prev_head_positions.append(face_box)

        # Keep only the latest positions for smoothing
        if len(self.prev_head_positions) > MOVEMENT_SMOOTHING:
            self.prev_head_positions.pop(0)

        # Average the positions
        if len(self.prev_head_positions) > 0:
            avg_x = sum(box[0] for box in self.prev_head_positions) / len(self.prev_head_positions)
            avg_y = sum(box[1] for box in self.prev_head_positions) / len(self.prev_head_positions)
            avg_w = sum(box[2] for box in self.prev_head_positions) / len(self.prev_head_positions)
            avg_h = sum(box[3] for box in self.prev_head_positions) / len(self.prev_head_positions)

            return (int(avg_x), int(avg_y), int(avg_w), int(avg_h))

        return face_box

    def adjust_camera_position(self, head_box):
        """Move ESP32-CAM to keep the head centered"""
        if not self.follow_person:
            return

        x, y, w, h = head_box
        head_center_x = x + w / 2
        head_center_y = y + h / 2

        frame_center_x = self.frame_width / 2
        frame_center_y = self.frame_height / 2

        # Calculate how far the head is from center (as fraction of frame size)
        x_offset = (head_center_x - frame_center_x) / self.frame_width
        y_offset = (head_center_y - frame_center_y) / self.frame_height

        print(f"Head offset: x={x_offset:.2f}, y={y_offset:.2f}")

        # Prioritize movement based on which offset is larger
        if abs(y_offset) > abs(x_offset) and abs(y_offset) > CENTER_THRESHOLD:
            if y_offset > 0:
                # Head is below center - move camera down
                self.move_camera("down")
            else:
                # Head is above center - move camera up
                self.move_camera("up")
        elif abs(x_offset) > CENTER_THRESHOLD:
            if x_offset > 0:
                # Head is to the right of center - move camera right
                self.move_camera("left")
            else:
                # Head is to the left of center - move camera left
                self.move_camera("right")

    def display_debug_frame(self, frame, faces):
        """Show debug information on frame"""
        debug_frame = frame.copy()

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw center point of face
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(debug_frame, (center_x, center_y), 3, (0, 255, 0), -1)

        # Draw crosshairs at center of frame
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        cv2.line(debug_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(debug_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)

        # Draw tracking thresholds
        threshold_w = int(self.frame_width * CENTER_THRESHOLD)
        threshold_h = int(self.frame_height * CENTER_THRESHOLD)
        cv2.rectangle(debug_frame,
                      (center_x - threshold_w, center_y - threshold_h),
                      (center_x + threshold_w, center_y + threshold_h),
                      (255, 0, 0), 1)

        # Add text for debugging info
        cv2.putText(debug_frame, "ESP32-CAM Head Tracking", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if len(faces) > 0:
            cv2.putText(debug_frame, f"Detected: {len(faces)} faces", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show recording status if enabled
        if self.record and self.recording_started:
            cv2.putText(debug_frame, "RECORDING", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Add red recording dot
            cv2.circle(debug_frame, (100, 60), 5, (0, 0, 255), -1)

        # Display frame
        cv2.imshow("ESP32-CAM Head Tracking", debug_frame)
        key = cv2.waitKey(1)

        # Press 'q' to quit
        if key == ord('q'):
            self.running = False
        # Press 'r' to toggle recording
        elif key == ord('r'):
            self.toggle_recording()

    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.record and self.recording_started:
            self.stop_recording()
            self.record = False
        else:
            self.record = True
            if self.frame_width > 0 and not self.recording_started:
                self.create_video_writer()

    def write_frame(self, frame):
        """Write a frame to the video file with frame rate control"""
        if not self.record or not self.recording_started or self.video_writer is None:
            return

        current_time = time.time()

        # Control frame rate to match target FPS
        if current_time - self.last_frame_write_time >= self.frame_interval:
            self.video_writer.write(frame)
            self.last_frame_write_time = current_time

    def stop_recording(self):
        """Stop video recording"""
        if self.recording_started and self.video_writer is not None:
            self.video_writer.release()
            print(f"Recording saved: {self.recording_path}")
            self.recording_started = False

    def run(self, show_debug=True):
        """Main tracking loop"""
        try:
            self.connect_to_stream()
            self.running = True

            print("Starting head tracking...")
            if self.follow_person:
                print("Person tracking enabled")
            if self.record:
                print("Video recording enabled")

            consecutive_no_detections = 0
            last_valid_head = None

            while self.running:
                # Check if the external stop event is set
                if stop_tracking_event.is_set():
                    print("Stopping tracking due to external event...")
                    self.running = False
                    break

                # Check if stream is active
                if not self.stream_reader.is_active():
                    print("Stream inactive, attempting to reconnect...")
                    self.connect_to_stream()
                    time.sleep(1)
                    continue

                # Get frame from stream
                frame = self.stream_reader.read()
                if frame is None:
                    print("No frame available")
                    time.sleep(0.1)
                    continue

                # Make a copy to avoid modifying the stream reader's frame
                frame = frame.copy()

                # Write frame to video if recording is enabled
                self.write_frame(frame)

                if self.follow_person:
                    # Detect faces (heads)
                    faces = self.detect_heads(frame)

                    # Find largest face
                    largest_face = self.get_largest_head(faces)

                    # Track face if found
                    if largest_face is not None:
                        # Save this as last valid head
                        last_valid_head = largest_face
                        consecutive_no_detections = 0

                        # Apply smoothing to reduce jitter
                        smoothed_face = self.smooth_head_position(largest_face)

                        # Move camera to track the face
                        self.adjust_camera_position(smoothed_face)

                        # For debug display
                        if show_debug:
                            self.display_debug_frame(frame, [smoothed_face])
                    else:
                        consecutive_no_detections += 1

                        # If we just lost track briefly, show the last known position
                        if consecutive_no_detections < 10 and last_valid_head is not None and show_debug:
                            self.display_debug_frame(frame, [last_valid_head])
                        elif show_debug:
                            self.display_debug_frame(frame, [])
                else:
                    # Just display the frame without tracking
                    if show_debug:
                        self.display_debug_frame(frame, [])

                # Small delay to reduce CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Tracking stopped by user")
        except Exception as e:
            print(f"Error in tracking: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            print("Head tracking stopped")

    def cleanup(self):
        """Clean up resources before exit"""
        if self.stream_reader is not None:
            self.stream_reader.stop()

        if self.recording_started and self.video_writer is not None:
            self.stop_recording()

        cv2.destroyAllWindows()

    def stop(self):
        """Stop the tracking"""
        self.running = False
        self.cleanup()


def newLook(follow_person=True, record=False, show_debug=True):
    """Main function to start the ESP32-CAM head tracking and/or recording"""
    global stop_tracking_event
    # Reset the stop event at the beginning of the function
    stop_tracking_event.clear()

    try:
        print("ESP32-CAM Camera System")
        print(f"Video stream: {VIDEO_STREAM_URL}")
        print(f"Camera control: {CAMERA_CONTROL_URL}")

        if show_debug:
            print("Press 'q' to quit")
            print("Press 'r' to toggle recording")

        # Start tracking
        tracker = HeadTrackingCamera(follow_person=follow_person, record=record)
        tracker.run(show_debug=show_debug)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Indicate we're no longer tracking
        print("newLook function completed")