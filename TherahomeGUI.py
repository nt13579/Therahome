import tkinter as tk
import os
from bash import bash
import time
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import StreamHandler as st
# from selenium.webdriver.du.options import Options
# import webbrowser as wb

# Create a window object
window = tk.Tk()
window.configure(bg="#6B7860")

# Create an event handler
def handle_keypress(event):
    print("The button was clicked!")

def handle_click(event):
    print("The button was clicked!")

def open_game(event
    #, version = 'selenium'
    ):
    print("opening game")
    # if version == 'local':
    #    print('Opening')
    #    bash('/home/nickt/MSC2-Linux/MSC2.x86_64 &')
    #    bash('wmctrl -r Madalin Stunt Cars 2 -e 0,2700,0,640,480')
    #    print('resizing')
    # elif version == 'selenium':
    #    chrome_options = Options()
    #    chrome_options.add_experimental_option("detach", True)
    #    driver = webdriver.Chrome(options=chrome_options)
    #    driver.get('https://madalinstuntcars2.io/%27')
    #    driver.set_window_position(3220, 0)

def start_recording(event):
    print('recording')
    # stream = st.StreamHandler([0], classify_behavior=True, dropFrames = True, 
    #                  behavior_model_path=os.path.join(main_dir, 'SVM_BasicMovement.sav'),
    #                  implement_control=True)
    # stream.beginCapture(pcf ,labelVideo=True, shuffle =1)

window.rowconfigure(0, weight=1, minsize=250)
window.rowconfigure(1, weight=1, minsize=50)
for i in range(4):
    window.columnconfigure(i, weight=1, minsize=75)  

logo_frame = tk.Frame(window, bg="#A4BCA4")
logo_frame.grid(row=0, columnspan=4, padx=10, pady=10)

# adjust the path for your system
therahome_logo = tk.PhotoImage(file='/home/jtstever/Downloads/Therahome.png')
logo = tk.Label(logo_frame, image=therahome_logo)
logo.pack()

frame0 = tk.Frame(window, relief=tk.RAISED, borderwidth=1, bg="#A4BCA4")
frame0.grid(row=1, column=0, padx=10, pady=10)

button0 = tk.Button(frame0, text=f"Launch Game", bg="#A4BCA4")
button0.bind("<Button-1>", open_game)
button0.pack(padx=2, pady=2)

frame1 = tk.Frame(window, relief=tk.RAISED, borderwidth=1, bg="#A4BCA4")
frame1.grid(row=1, column=1, padx=10, pady=10)

button1 = tk.Button(frame1, text=f"Start Recording", bg="#A4BCA4")
button1.bind("<Button-1>", start_recording)
button1.pack(padx=2, pady=2)

frame2 = tk.Frame(window, relief=tk.RAISED, borderwidth=1, bg="#A4BCA4")
frame2.grid(row=1, column=2, padx=10, pady=10)

button2 = tk.Button(frame2, text=f"View Instructions", bg="#A4BCA4")
# TODO: Add instructions page/method
# button2.bind("<Button-1>", view_instructions)
button2.pack(padx=2, pady=2)

frame3 = tk.Frame(window, relief=tk.RAISED, borderwidth=1, bg="#A4BCA4")
frame3.grid(row=1, column=3, padx=10, pady=10)

button3 = tk.Button(frame3, text=f"Close", bg="#A4BCA4", command=window.destroy)
# @ NICK do we need a stop recording button or will it save everything even if it just closes?
# button3.bind("<Button-1>", stop_recording)
button3.pack(padx=2, pady=2)

# Run the event loop
window.mainloop()
