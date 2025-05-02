# main.py - Main Virtual Assistant Script
import os
import requests
import pyttsx3
import speech_recognition as sr
import datetime
import webbrowser
import wikipedia
import pywhatkit
import smtplib
import subprocess
import threading
import time
import sys
from dotenv import load_dotenv
import google.generativeai as genai  # Added for Gemini integration

# Load environment variables
load_dotenv()
USER = os.getenv("SIR") or "User"
BOTNAME = os.getenv("BOTNAME") or "Assistant"
EMAIL = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_KEY = "AIzaSyBvE04SnLektunSmuCKk0CnzvFSScZupG8"  # Google API Key
GEMINI_MODEL = "gemini-2.0-flash-lite"  # Gemini model name

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

# Import the newLook module and get access to the stop_tracking_event
# Use a try-except to handle potential import errors gracefully
try:
    from newLook import newLook, stop_tracking_event

    tracking_module_available = True
except ImportError:
    print("Warning: newLook module not found. Person tracking will be disabled.")
    tracking_module_available = False
    # Create a dummy event object if the module is not available
    stop_tracking_event = threading.Event()

# Global variables to track the person tracking thread
person_tracking_thread = None
tracking_active = False
talking_mode = False  # Flag to track if we're in talking mode

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice


# Speak Function
def speak(audio):
    print(f"{BOTNAME}: {audio}")
    engine.say(audio)
    engine.runAndWait()


# Wish Me Function
def wish_me():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak(f"Good Morning {USER}!")
    elif 12 <= hour < 18:
        speak(f"Good Afternoon {USER}!")
    else:
        speak(f"Good Evening {USER}!")
    speak(f"I am {BOTNAME}. How can I help you?")


# Take Command Function
def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-US')
        print(f"{USER}: {query}\n")
    except Exception as e:
        print("Say that again please...")
        return "None"
    return query.lower()


# Send Email Function
def send_email(to, subject, body):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL, EMAIL_PASSWORD)
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(EMAIL, to, message)
        server.quit()
        speak("Email has been sent successfully!")
    except Exception as e:
        speak("Sorry, I was not able to send the email.")
        print(e)


# Simple Calculator using speech
def calculate(expression):
    try:
        result = eval(expression)
        speak(f"The answer is {result}")
    except Exception as e:
        speak("Sorry, I couldn't calculate that.")
        print(e)


# Open installed apps
def open_application(app_name):
    try:
        if 'calculator' in app_name:
            subprocess.Popen('calc.exe')
            speak("Opening Calculator")
        elif 'arduino' in app_name:
            subprocess.Popen(r"C:\Users\Lenovo\AppData\Roaming\Microsoft\Windows\Start Menu\Programs")
            speak("Opening Arduino IDE")
        elif 'steam' in app_name:
            subprocess.Popen(r"C:\Program Files (x86)\Steam\Steam.exe")
            speak("Opening Steam")
        else:
            speak(f"Application {app_name} is not configured yet.")
    except Exception as e:
        speak(f"Failed to open {app_name}")
        print(e)


# Voice Command for Notes
def take_notes():
    speak("What would you like to note down?")
    note = take_command()
    if note != "None":
        with open("notes.txt", "a") as file:
            file.write(f"{datetime.datetime.now()}: {note}\n")
        speak("Your note has been saved.")


# Set Alarm Function
def set_alarm():
    speak("Please tell me the time for the alarm in HH:MM format.")
    alarm_time = input("Enter the alarm time in HH:MM format: ")  # Take input manually
    if alarm_time != "None":
        try:
            alarm_hour, alarm_minute = map(int, alarm_time.split(":"))
            current_time = datetime.datetime.now()
            alarm_time_obj = current_time.replace(hour=alarm_hour, minute=alarm_minute, second=0, microsecond=0)
            if alarm_time_obj < current_time:
                alarm_time_obj += datetime.timedelta(days=1)  # Set alarm for next day
            speak(f"Setting alarm for {alarm_time_obj.strftime('%H:%M:%S')}")

            # Run alarm in a separate thread
            def alarm_thread():
                while True:
                    if datetime.datetime.now() >= alarm_time_obj:
                        speak("Time's up! Alarm ringing.")
                        webbrowser.open("https://www.youtube.com/watch?v=enYdAxVcNZA")
                        break
                    time.sleep(1)

            threading.Thread(target=alarm_thread, daemon=True).start()
            speak("Alarm set successfully.")
        except ValueError:
            speak("Invalid time format. Please use HH:MM format.")


# Website Search
def website_search(query):
    if 'anime' in query:
        speak("Opening Anime website.")
        webbrowser.open("https://jut.su/anime/")
    elif 'korean website' in query:
        speak("Opening Dorama website.")
        webbrowser.open("https://doramy.club/")
    elif 'portal' in query:
        speak("Opening SDU Portal.")
        webbrowser.open("https://my.sdu.edu.kz/index.php")
    elif 'moodle' in query:
        speak("Opening Moodle website.")
        webbrowser.open("https://moodle.sdu.edu.kz/login/index.php")


# Direct wrapper for newLook function
def newLook_wrapper(follow_person, record, show_debug=True):
    global tracking_active

    try:
        # Clear the stop event before starting
        stop_tracking_event.clear()

        # Start tracking
        newLook(follow_person=follow_person, record=record, show_debug=show_debug)

    except Exception as e:
        print(f"Error in newLook wrapper: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure we update the tracking state
        tracking_active = False
        print("newLook wrapper completed")


# Function to toggle person following mode
def toggle_person_following(follow=True, record=True, show_debug=True):
    global person_tracking_thread, tracking_active

    # Check if tracking module is available
    if not tracking_module_available:
        speak("Person tracking functionality is not available. Please make sure newLook.py is in the same directory.")
        return

    try:
        # If tracking is active, stop it first
        if tracking_active and person_tracking_thread and person_tracking_thread.is_alive():
            speak("Stopping current tracking session...")
            stop_tracking_event.set()  # Signal the thread to stop

            # Wait for the thread to finish with a timeout
            person_tracking_thread.join(timeout=5)

            if person_tracking_thread.is_alive():
                print("Warning: Tracking thread did not terminate properly")

            # Reset tracking state
            tracking_active = False
            person_tracking_thread = None
            speak("Tracking stopped successfully.")

        # Start new tracking if requested
        if follow:
            speak("Initializing person tracking mode...")
            tracking_active = True

            # Create and start a new thread for person tracking
            person_tracking_thread = threading.Thread(
                target=newLook_wrapper,
                args=(follow, record, show_debug),
                daemon=True
            )
            person_tracking_thread.start()

            speak("Now following person movements.")
            if record:
                speak("Recording is enabled.")
            else:
                speak("Recording is disabled.")
        else:
            tracking_active = False
            speak("Person tracking is deactivated.")

    except Exception as e:
        speak("There was an error with the person tracking functionality.")
        print(f"Error in person tracking: {e}")
        import traceback
        traceback.print_exc()


# Function to handle conversations with Gemini
def talk_with_gemini(query):
    try:
        # Generate a response from Gemini
        response = gemini_model.generate_content(query)
        # Check if the response is valid
        if response and hasattr(response, 'text'):
            response_text = response.text
            # Speak the response
            speak(response_text)
        else:
            speak("I couldn't get a response from Gemini at the moment.")
    except Exception as e:
        speak("Sorry, there was an error communicating with Gemini.")
        print(f"Gemini error: {e}")


# Main Function
def main():
    global talking_mode
    wish_me()
    
    while True:
        query = take_command()
        
        # Check if we should enter or exit talking mode
        if 'let\'s talk' in query:
            talking_mode = True
            speak("Entering conversation mode with Gemini. Say 'stop talking' when you want to exit this mode.")
            continue
        elif 'stop talking' in query and talking_mode:
            talking_mode = False
            speak("Exiting conversation mode. Back to regular commands.")
            continue
            
        # If in talking mode, send all queries to Gemini
        if talking_mode:
            if query != "None":
                talk_with_gemini(query)
            continue
            
        # Regular command processing when not in talking mode
        if 'wikipedia' in query:
            query = query.replace('wikipedia', '').strip()
            if query == "":
                speak('What should I search on Wikipedia?')
                query = take_command()
            if query != "None" and query.strip() != "":
                speak(f'Searching Wikipedia for {query}...')
                try:
                    results = wikipedia.summary(query, sentences=2)
                    speak("According to Wikipedia")
                    print(results)
                    speak(results)
                except Exception as e:
                    speak("Sorry, I couldn't find anything on Wikipedia.")
                    print(e)

        elif 'open youtube' in query:
            speak('Opening YouTube')
            webbrowser.open("https://www.youtube.com")

        elif 'open google' in query:
            speak('Opening Google')
            webbrowser.open("https://www.google.com")

        elif 'play' in query:
            song = query.replace('play', '').strip()
            if song:
                speak(f"Playing {song} on YouTube")
                try:
                    # Use pywhatkit to play the song directly
                    pywhatkit.playonyt(song)
                except Exception as e:
                    speak(f"Sorry, I couldn't play {song}. Opening search results instead.")
                    print(e)
                    query_url = f"https://www.youtube.com/results?search_query={song.replace(' ', '+')}"
                    webbrowser.open(query_url)
            else:
                speak("Please tell me what you'd like to play.")

        elif 'search on google' in query:
            speak("What should I search for?")
            search_query = take_command()
            if search_query != "None":
                webbrowser.open(f"https://www.google.com/search?q={search_query}")
                speak(f"Here are the results for {search_query} on Google.")

        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"{USER}, the time is {strTime}")

        elif 'ip address' in query:
            try:
                ip = requests.get('https://api.ipify.org').text
                speak(f"Your IP Address is {ip}")
            except Exception as e:
                speak("Sorry, I couldn't fetch your IP address.")

        elif 'send whatsapp message' in query:
            speak("Tell me the phone number including country code.")
            phone_number = input("Enter phone number: ")
            speak("What message should I send?")
            message = take_command()
            pywhatkit.sendwhatmsg_instantly(phone_number, message)
            speak("Message has been sent!")

        elif 'send email' in query:
            speak("Please tell me the email address.")
            to = input("Enter recipient email: ")
            speak("What is the subject?")
            subject = take_command()
            speak("What is the body?")
            body = take_command()
            send_email(to, subject, body)

        elif 'news' in query:
            try:
                url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
                news = requests.get(url).json()
                articles = news["articles"]
                for article in articles[:5]:
                    speak(article["title"])
            except Exception as e:
                speak("Sorry, I couldn't fetch news headlines.")

        elif 'weather' in query:
            speak("Please tell me the city name.")
            city = take_command()
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
                weather_data = requests.get(url).json()
                if weather_data["cod"] == 200:
                    temp = weather_data["main"]["temp"]
                    description = weather_data["weather"][0]["description"]
                    speak(f"The temperature in {city} is {temp}°C with {description}.")
                else:
                    speak("City not found.")
            except Exception as e:
                speak("Sorry, I couldn't fetch the weather information.")

        elif 'trending movies' in query:
            try:
                url = "https://api.themoviedb.org/3/trending/movie/day?api_key=4ca6db56909dbf602a88a278a634cdf0"
                response = requests.get(url)
                data = response.json()
                for movie in data['results'][:5]:
                    speak(movie['title'])
            except Exception as e:
                speak("Sorry, I couldn't fetch trending movies.")

        elif 'joke' in query:
            try:
                joke = requests.get("https://official-joke-api.appspot.com/random_joke").json()
                speak(joke["setup"])
                speak(joke["punchline"])
            except:
                speak("I couldn't find a joke at the moment.")

        elif 'advice' in query:
            try:
                advice = requests.get("https://api.adviceslip.com/advice").json()
                speak(advice["slip"]["advice"])
            except:
                speak("I couldn't find advice right now.")

        elif 'who are you' in query:
            speak(f"I am {BOTNAME}, your personal assistant.")

        elif 'who am i' in query:
            speak(f"You are {USER}, my boss.")

        elif 'open calculator' in query or 'open arduino' in query or 'open steam' in query:
            open_application(query)

        elif 'calculate' in query:
            speak("What should I calculate?")
            expression = take_command()
            calculate(expression)

        elif 'take notes' in query:
            take_notes()

        elif 'set alarm' in query:
            set_alarm()

        # Person following commands with proper implementation
        elif 'follow me' in query or 'track me' in query:
            speak("Activating person following mode in background.")
            toggle_person_following(follow=True, record=True)

        elif 'stop following' in query or 'stop tracking' in query:
            speak("Deactivating person following mode.")
            toggle_person_following(follow=False, record=False)

        elif 'follow without recording' in query:
            speak("Following without recording.")
            toggle_person_following(follow=True, record=False)

        elif 'follow and record' in query:
            speak("Following with recording enabled.")
            toggle_person_following(follow=True, record=True)

        elif 'tracking status' in query:
            if tracking_active and person_tracking_thread and person_tracking_thread.is_alive():
                speak("Person tracking is currently active.")
            else:
                speak("Person tracking is currently inactive.")

        elif 'exit' in query or 'quit' in query or 'stop' in query or 'goodbye' in query or 'bye' in query:
            # Make sure to stop tracking before exiting
            if tracking_active:
                speak("Stopping person tracking before exit.")
                toggle_person_following(follow=False, record=False)
            speak("Goodbye, have a great day!")
            break

        elif 'anime' in query or 'korean website' in query or 'portal' in query or 'moodle' in query:
            website_search(query)

        else:
            speak("I didn't understand that. Please say it again.")


if __name__ == "__main__":
    main()