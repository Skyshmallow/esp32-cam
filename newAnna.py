import cv2
import numpy as np
import requests
import threading
import time
import speech_recognition as sr
import google.generativeai as genai
import edge_tts
import asyncio
import sounddevice as sd
import soundfile as sf
import mediapipe as mp
import queue
import io
import audioop

# Configuration
ESP32_CAM_STREAM_URL = "http://192.168.46.186:81/stream"
ESP32_CONTROL_URL = "http://192.168.46.186/action"
GOOGLE_KEY = "AIzaSyBvE04SnLektunSmuCKk0CnzvFSScZupG8"
GEMINI_MODEL = "gemini-2.0-flash-lite"
VOICE = "en-US-AnaNeural"
# VOICE        = "kk-KZ-AigulNeural"
WAKE_PHRASE = "ok google"

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables
frame_queue = queue.Queue(maxsize=1)
current_position = {"pan": 0, "tilt": 0}
voice_command_active = False
response_queue = queue.Queue()

# Generate and play a start listening beep sound (higher pitch)
def generate_start_beep():
    sample_rate = 44100
    duration = 0.2  # seconds
    frequency = 1500  # Hz (higher pitch)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add fade-in
    fade_samples = int(0.05 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    beep[:fade_samples] = beep[:fade_samples] * fade_in
    
    sd.play(beep, sample_rate)
    sd.wait()

# Generate and play a stop listening beep sound (lower pitch, descending)
def generate_stop_beep():
    sample_rate = 44100
    duration = 0.3  # seconds
    start_freq = 1200  # Hz
    end_freq = 800  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create descending tone
    frequency = np.linspace(start_freq, end_freq, int(sample_rate * duration))
    beep = 0.5 * np.sin(2 * np.pi * t * frequency)
    
    # Add fade-out
    fade_samples = int(0.1 * sample_rate)
    fade_out = np.linspace(1, 0, fade_samples)
    beep[-fade_samples:] = beep[-fade_samples:] * fade_out
    
    sd.play(beep, sample_rate)
    sd.wait()

# Function to send commands to ESP32-CAM
def move_camera(direction):
    global current_position
    
    try:
        response = requests.get(f"{ESP32_CONTROL_URL}?go={direction}", timeout=1)
        
        # Update position tracking
        if direction == "up":
            current_position["tilt"] += 10
        elif direction == "down":
            current_position["tilt"] -= 10
        elif direction == "left":
            current_position["pan"] -= 10
        elif direction == "right":
            current_position["pan"] += 10
            
        return response.status_code == 200
    except requests.exceptions.RequestException:
        print(f"Failed to send {direction} command to camera")
        return False

# Function to track human in frame using MediaPipe Pose
def track_human():
    global frame_queue
    
    # Initialize pose detector with standard MediaPipe API
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # Calculate center of frame and movement thresholds
    frame_center_x_offset = 100  # Allow some margin before moving
    frame_center_y_offset = 80
    move_cooldown = 0  # Prevent rapid movement
    
    while True:
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = frame_queue.get()
            if frame is None:
                continue
                
            height, width = frame.shape[:2]
            frame_center_x = width // 2
            frame_center_y = height // 2
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)
            
            # Check if pose landmarks were detected
            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)
                
                # Get nose keypoint (0 is nose in MediaPipe Pose)
                nose_landmark = results.pose_landmarks.landmark[0]
                nose_x = int(nose_landmark.x * width)
                nose_y = int(nose_landmark.y * height)
                
                # Highlight nose position
                cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
                
                # Determine if camera movement is needed (when cooldown allows)
                if move_cooldown <= 0:
                    move_x = 0
                    move_y = 0
                    
                    if nose_x < frame_center_x - frame_center_x_offset:
                        move_x = -1  # move camera left
                    elif nose_x > frame_center_x + frame_center_x_offset:
                        move_x = 1   # move camera right
                        
                    if nose_y < frame_center_y - frame_center_y_offset:
                        move_y = 1   # move camera up
                    elif nose_y > frame_center_y + frame_center_y_offset:
                        move_y = -1  # move camera down
                    
                    # Send camera movement commands
                    if move_x == -1:
                        move_camera("left")
                        move_cooldown = 5
                    elif move_x == 1:
                        move_camera("right")
                        move_cooldown = 5
                        
                    if move_y == -1:
                        move_camera("down")
                        move_cooldown = 5
                    elif move_y == 1:
                        move_camera("up")
                        move_cooldown = 5
            
            if move_cooldown > 0:
                move_cooldown -= 1
                
            # Display the frame - add visual indicator when listening
            if voice_command_active:
                cv2.putText(frame, "Listening...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow('ESP32-CAM Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error in human tracking: {e}")
            
    cv2.destroyAllWindows()

# Function to stream video from ESP32-CAM
def stream_video():
    global frame_queue
    
    cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)
    
    if not cap.isOpened():
        print("Failed to open stream URL")
        return
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to retrieve frame, reconnecting...")
                cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)
                time.sleep(1)
                continue
            
            # Update the frame queue
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        except Exception as e:
            print(f"Error in video streaming: {e}")
            time.sleep(1)
    
    cap.release()

# Function to listen for wake phrase
def listen_for_wake_word():
    global voice_command_active
    
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for wake phrase...")
                audio = recognizer.listen(source, phrase_time_limit=2)
                
            try:
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                
                if WAKE_PHRASE in text:
                    print("Wake phrase detected!")
                    voice_command_active = True
                    generate_start_beep()
                    process_voice_command()
            except sr.UnknownValueError:
                # Speech not understood, continue listening
                pass
            except sr.RequestError as e:
                print(f"Google Speech Recognition service error: {e}")
                time.sleep(1)
                
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            time.sleep(1)

# Function to process voice command after wake word
def process_voice_command():
    global voice_command_active
    
    recognizer = sr.Recognizer()
    max_command_duration = 8  # Maximum time for command in seconds
    
    try:
        with sr.Microphone() as source:
            print("Listening for command...")
            
            # Fixed time listening
            audio = recognizer.listen(source, phrase_time_limit=max_command_duration)
            
            # Play the stop listening beep to indicate recording is complete
            generate_stop_beep()
            
            try:
                command = recognizer.recognize_google(audio)
                print(f"Command: {command}")
                
                # Send command to Gemini
                threading.Thread(target=process_with_gemini, args=(command,), daemon=True).start()
                
            except sr.UnknownValueError:
                print("Could not understand command")
                asyncio.run(speak_text("Sorry, I didn't catch that."))
            except sr.RequestError as e:
                print(f"Google Speech Recognition service error: {e}")
                asyncio.run(speak_text("Sorry, I'm having trouble understanding you right now."))
            
    except Exception as e:
        print(f"Error processing voice command: {e}")
        
    finally:
        voice_command_active = False

# Function to process command with Gemini
def process_with_gemini(text):
    try:
        # Configure Gemini API
        genai.configure(api_key=GOOGLE_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Get response from Gemini
        response = model.generate_content(text)
        response_text = response.text
        print(f"Gemini response: {response_text}")
        
        # Convert response to speech
        asyncio.run(speak_text(response_text))
        
    except Exception as e:
        print(f"Error with Gemini processing: {e}")
        asyncio.run(speak_text("Sorry, I couldn't process your request."))

# Function to convert text to speech
async def speak_text(text):
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        audio_data = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
                
        audio_data.seek(0)
        audio_array, sample_rate = sf.read(audio_data)
        sd.play(audio_array, sample_rate)
        sd.wait()
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# Main function
def main():
    # Start video streaming thread
    video_thread = threading.Thread(target=stream_video, daemon=True)
    video_thread.start()
    
    # Start human tracking thread
    tracking_thread = threading.Thread(target=track_human, daemon=True)
    tracking_thread.start()
    
    # Start wake word detection thread
    voice_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    voice_thread.start()
    
    print("System started. Press Ctrl+C to exit.")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()