Follow that steps to use:

Open any ide(python or vs code):
Put that files:
new gpt.py
newLook.py




Initial Setup:
Connect the ESP32-CAM to a power source (via USB or power bank).

Create a mobile hotspot with:
SSID: 11t
Password: 123456789

Wait for the ESP32-CAM to connect to the hotspot

Check your phone’s connected devices list and find the IP address of the ESP32-CAM.



Automatic Control Mode (Tracking + Voice Commands)
_________________________________________
 On your PC:
First time only: Open a terminal and install dependencies:
    pip install -r requirements.txt
Make sure the camera’s IP address is correctly updated in NewLook.py.

Run the tracking system:  
             python gpt.py


Speak commands such as:
"follow me", "track me" — to start following
"follow and record" — to track and save the video
"follow without recording"
"stop following", "stop tracking" — to stop tracking and save recording (if enabled)


For YouTube feature (optional):
Say: "play [video name] on YouTube"
That’s it — the camera now tracks and responds to your voice commands!