#  TASK SPAM DETECTION SYSTEM USING NAVIE BAYES AND LOGISTIC REGRESSION WITH VOICE INPUT AND OUTPUT BY PURNANKIT SIRVI 
import threading
import tkinter as tk
from tkinter import messagebox
import pickle
import speech_recognition as sr
import pyttsx3
from pygame import mixer
from datetime import datetime
mixer.init()

#LOAD MODELS
model_nb = pickle.load(open("model_nb.pkl", "rb"))
model_lr = pickle.load(open("model_lr.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
#DATETIME FUNCTION
def get_datetime():
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%I:%M:%S %p")
    return date, time
def update_clock():
    date, time = get_datetime()
    datetime_label.config(text=f"Date: {date} | Time: {time}")
    root.after(1000, update_clock)   # update every second

#TEXT TO SPEECH
def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

#SPEECH TO TEXT
def take_voice_input():
    mixer.music.load("music.mp3")
    mixer.music.play()
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        result_label_nb.config(text="Listening...", fg="sky blue")
        root.update()

        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
    mixer.music.load("music 1.mp3")
    mixer.music.play()

    try:
        text = recognizer.recognize_google(audio)
        text_input.delete("1.0", tk.END)
        text_input.insert(tk.END, text)
        return text

    except Exception as e:
        messagebox.showerror("Voice Error", str(e))
        return None

# PREDICTION LOGIC
def process_message(msg, speak_output=False):
    msg_vec = vectorizer.transform([msg])

    result_nb = model_nb.predict(msg_vec)[0]
    result_lr = model_lr.predict(msg_vec)[0]

# HERE I USE NAVIE BAYES TO PREDICT WHETHER THE MESSAGE IS SPAM OR NOT
    if result_nb == 1:
        result_label_nb.config(text="Naive Bayes: SPAM", fg="red")
    else:
        result_label_nb.config(text="Naive Bayes: HAM", fg="green")

# HERE I USE THE SAME LOGIC FOR LOGISTIC REGRESSION AS WELL
    if result_lr == 1:
        result_label_lr.config(text="Logistic Regression: SPAM", fg="red")
    else:
        result_label_lr.config(text="Logistic Regression: HAM", fg="green")

   
    if speak_output:# IF THE USER WANTS TO HEAR THE RESULT THEN THIS BLOCK WILL EXECUTE
       now = datetime.now()

       date_time_voice = now.strftime("Today is %A, %B %d, %Y and the time is %I:%M %p")

       if result_nb == 1 or result_lr == 1:
           speak(f"{date_time_voice}. Warning! This message is spam")
       else:
           speak(f"{date_time_voice}. This message is safe")
def predict_message():
    msg = text_input.get("1.0", tk.END).strip()

    if msg == "":
        messagebox.showwarning("Warning", "Enter a message")
        return

    process_message(msg, speak_output=False) 

def voice_predict():
    threading.Thread(target=run_voice).start()

def run_voice():
    msg = take_voice_input()

    if msg is None:
        result_label_nb.config(text="Could not understand voice", fg="orange")
        result_label_lr.config(text="")
        speak("Sorry, I could not understand")
        return

    process_message(msg, speak_output=True)


def clear_text():
    text_input.delete("1.0", tk.END)
    result_label_nb.config(text="")
    result_label_lr.config(text="")

# GRAPHICAL USER INTERFACE-GUI SETUP
root = tk.Tk()
root.title("SPAM SMS DETECTION SYSTEM")
root.geometry("1980x1080+0+0")
root.state("zoomed")
root.config(bg="#060436")

title = tk.Label(root, text="SPAM SMS DETECTION SYSTEM",
                 font=("Times New Roman", 18, "bold"),
                 bg="#060436", fg="white")
title.pack(pady=10)
datetime_label = tk.Label(root,
                          font=("Times New Roman", 10, "bold"),
                          bg="#060436", fg="cyan")
datetime_label.pack(pady=5)

text_input = tk.Text(root, height=6, width=50,
                     font=("Arial", 12),
                     bg="#2c2f36", fg="white",
                     insertbackground="white")
text_input.pack(pady=10)

frame = tk.Frame(root, bg="#060436")
frame.pack(pady=10)

btn_predict = tk.Button(frame, text="Predict",
                        command=predict_message,
                        bg="#075021", fg="white",
                        width=12, font=("Times New Roman", 11, "bold"))
btn_predict.grid(row=0, column=0, padx=10)

btn_voice = tk.Button(frame, text="Speak",
                      command=voice_predict,
                      bg="#1B2A82", fg="white",
                      width=12, font=("Times New Roman", 11, "bold"))
btn_voice.grid(row=0, column=1, padx=10)

btn_clear = tk.Button(frame, text="Clear",
                      command=clear_text,
                      bg="#d80202", fg="white",
                      width=12, font=("Times New Roman", 11, "bold"))
btn_clear.grid(row=0, column=2, padx=10)

result_label_nb = tk.Label(root, text="",
                          font=("Times New Roman", 14, "bold"),
                          bg="#060436", fg="white")
result_label_nb.pack(pady=10)

result_label_lr = tk.Label(root, text="",
                          font=("Times New Roman", 14, "bold"),
                          bg="#060436", fg="white")
result_label_lr.pack(pady=10)

update_clock()
root.mainloop()     