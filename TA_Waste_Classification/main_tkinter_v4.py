
#* IMPORT ALL LIBRARY
from re import L, X
import tkinter as tk
from tkinter import BOTTOM, Canvas, Listbox, filedialog
from tkinter.messagebox import showinfo
from tkinter.ttk import *
from tokenize import Triple
from PIL import ImageTk, Image
import cv2
from matplotlib.pyplot import text
import numpy as np
import os
import shutil, sys
import time
from setuptools import Command
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import RPi.GPIO as GPIO
import serial

#global kit
#kit = ServoKit(channels=16)
GPIO.setmode(GPIO.BCM)
global relay_gpio
relay_gpio = 17
GPIO.setup(relay_gpio, GPIO.OUT)

# SERIAL
#serialcom = serial.Serial('/dev/ttyACM1', 9600, timeout=1)

result_text = ''
class_name = ''
x_origin = 0
y_origin = 0
lebar = 0
tinggi = 0


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

# UTILS

def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  global bbox
  global result_text
  global class_name
  global x_origin
  global y_origin
  global lebar
  global tinggi

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    #print(str(bbox))
    #print(str(type(bbox)))
    x_origin = bbox.origin_x
    y_origin = bbox.origin_y
    lebar = bbox.width
    tinggi = bbox.height
    start_point = x_origin, y_origin
    end_point = x_origin + lebar, y_origin + tinggi
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.classes[0]
    class_name = category.class_name
    probability = round(category.score, 2)
    result_text = class_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + bbox.origin_x,
                     _MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image

#* CREATE MAIN WINDOW
master = tk.Tk()
master.title("Waste Classification")
master.geometry("1100x600")
master.resizable(False, False)

frame = tk.Frame(master)
frame.pack()

#* LABEL FOR TITLE
top_frame = Frame(master)
top_frame.pack(side = tk.TOP)
model_label = tk.Label(top_frame, text="WASTE CLASSIFICATION LEARNING MODULE", justify='center', width=40, font= 14)
model_label.pack(side= tk.LEFT, pady=20, fill= tk.BOTH)

#* LEFT WIDGET
left_frame = Frame(master)
left_frame.pack(side= tk.LEFT)

#* SHOWING VIDEO
accuracy_threshold = 0.6
max_detection_obj = 1

frame = np.random.randint(0,255,[100,100,3],dtype='uint8')
img = ImageTk.PhotoImage(Image.fromarray(frame))
video_show = Label(master) #,image=img)
video_show.place(x = 20, y = 75)

global count_id
count_id = 0-1
def kirim_data(class_name):
    if class_name == 'glass':
        data_yang_dikirim = 0
        #serialcom.write('0'.encode())
        print('data yang dikirim ' + str(data_yang_dikirim))
    elif class_name == 'metal':
        data_yang_dikirim = 1
        #serialcom.write('1'.encode())
        print('data yang dikirim ' + str(data_yang_dikirim))
    elif class_name == 'paper':
        data_yang_dikirim = 2
        #serialcom.write('2'.encode())
        print('data yang dikirim ' + str(data_yang_dikirim))
    elif class_name == 'plastic':
        data_yang_dikirim = 3
        #serialcom.write('3'.encode())
        print('data yang dikirim ' + str(data_yang_dikirim))


def real_time ():
    
    counter, fps = 0, 0
    #start_time = time.time()

    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0

    global count_id
    count_id = 0
    
    start_count = False
    counting = True

    global frame
    global cam
    cam = cv2.VideoCapture(0)
    global width
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    global height
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    while cam.isOpened() and is_start:
        ret, frame = cam.read()
        #Update the image to tkinter...
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if current_method != 'Template Matching':
            #used_model = os.path.join(current_path, 'saved_model')
            #used_model = os.path.join(used_model, current_method)
            #base_options = core.BaseOptions(file_name = used_model)
            detection_options = processor.DetectionOptions(max_results = max_detection_obj, score_threshold = accuracy_threshold)
            options = vision.ObjectDetectorOptions(base_options = base_options, detection_options = detection_options)
            detector = vision.ObjectDetector.create_from_options(options)

            # Create a TensorImage object from the RGB image.
            input_tensor = vision.TensorImage.create_from_array(frame)

            # Run object detection estimation using the model.
            detection_result = detector.detect(input_tensor)
            #print(class_name)
            #print(str(x_origin) + str(y_origin))

            # Draw keypoints and edges on input image
            frame = visualize(frame, detection_result)
            
            result_class = result_text
            class_result.configure(text="RESULT = " + result_class.upper())
        
        # TITIK AWAL BENDA
        front_object = (x_origin + int(lebar/2), y_origin)
        cv2.circle(frame, front_object, 1, (255, 0, 0), 2)
        
        # AREA DETEKSI
        #print('width' + str(width))
        #print('height' + str(height))
        start_area1 = (0, int(3*height/4))
        end_area1 = (int(width), int(3*height/4))
        cv2.line(frame, start_area1, end_area1, (0, 255, 0), 2)
        start_area2 = (0, int(2*height/4))
        end_area2 = (int(width), int(2*height/4))
        cv2.line(frame, start_area2, end_area2, (255, 0, 0), 2)
        
        # MEMBERI ID
        
        if (y_origin) < (int(3*tinggi/4)) and (y_origin) > (int(2*tinggi/4)) and counting == True:
            start_count = True
            if start_count == True:
                count_id += 1
                counting = False
                print(count_id)
                kirim_data(class_name)
            
        elif ((y_origin) <= (int(2*tinggi/4)) or (y_origin) >= (int(3*tinggi/4))) and counting == False:
            counting = True
            start_count = False

        #print(count_id)
        

        # Calculate the FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        # Show the FPS
        #fps_text = 'FPS = {:.1f}'.format(fps)
        cv2.putText(frame, fps, (25, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        
        img_update = ImageTk.PhotoImage(Image.fromarray(frame))
        video_show.configure(image=img_update)
        video_show.image=img_update
        video_show.update()

        if not ret:
            print("failed to grab frame")
            break

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")

            cam.release()
            cv2.destroyAllWindows()
            break    

#* RIGHT WIDGET
right_frame = Frame(master)
right_frame.pack(side= tk.RIGHT)

#* COMBOBOX METHODS
current_method = ''
selected_method = tk.StringVar()

methods_combobox = Combobox(master, textvariable=selected_method)
current_path = os.getcwd()
global method_list
method_list = os.listdir(os.path.join(current_path, 'saved_model'))
#method_list.append('Template Matching')
#banyak_method = len(method_list)
methods_combobox['values'] = method_list
methods_combobox['state'] = 'readonly'
methods_combobox.place(x=820, y=170)


def method_changed (event):
    showinfo(
        title='Current Method',
        message=f'The Method Has Been Changed to {selected_method.get()}',
    )
    global current_method, used_model, base_options
    current_method = str(selected_method.get())
    print('DALAM FUNGSI ' + current_method)
    used_model = os.path.join(current_path, 'saved_model')
    used_model = os.path.join(used_model, current_method)
    base_options = core.BaseOptions(file_name = used_model)

    method_label.configure(text="CURRENT METHOD USE : " + current_method)
    
methods_combobox.bind('<<ComboboxSelected>>', method_changed)

#* START BUTTON
def start_btn_action():
    global is_start
    is_start = True
    GPIO.output(relay_gpio, GPIO.HIGH)
    print('Motor ON')
    print(current_method)
    print(is_start)
    real_time()

global is_start
is_start = False
start_button = Button(master, text="START", command=start_btn_action)
start_button.place(x=750, y=200)

#* STOP BUTTON
def stop_btn_action():
    global is_start
    is_start = False
    GPIO.output(relay_gpio, GPIO.LOW)
    print('Motor OFF')
    cam.release()
    cv2.destroyAllWindows()
    print("Stopped!")

stop_button = Button(master, text="STOP", command=stop_btn_action)
stop_button.place(x=850, y=200)

#* ADD BUTTON
destination_path = os.path.join(current_path, 'saved_model')

def browseFiles():
    file_name = filedialog.askopenfilename(initialdir='/',
                                            title='Select a Model',
                                            filetypes=(('Tensorflow Lite Model (.tflite)', '*.tflite'),('All Files', '*.*')))
    #method_label.configure(text="File Opened: " + str(file_name))
    shutil.copy(file_name, destination_path)
    method_list = os.listdir(os.path.join(current_path, 'saved_model'))
    methods_combobox['values'] = method_list
    print("Successfullly added new model")
    
add_button = Button(master, text="ADD", command=browseFiles)
add_button.place(x=950, y=200)

bottom_frame = Frame(master)
bottom_frame.pack(side=tk.BOTTOM)

#* PREDICTION RESULT
prediction_label = Label(master, text = "RESULT")
prediction_label.place(x=750, y=250)

#* CURRENT METHOD USE
method_label = Label(master, text = "CURRENT METHOD USE :")
method_label.place(x=750, y=275)

#* RESULT
class_result = Label(master, text = "CLASS = ")
class_result.place(x=750, y=300)

#* SHOWING GUI
master.mainloop()

#CLOSING GUI
cv2.destroyAllWindows()
GPIO.output(relay_gpio, GPIO.LOW)
print('Motor OFF')