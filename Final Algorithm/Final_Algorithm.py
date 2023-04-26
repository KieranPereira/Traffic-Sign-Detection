import torch
import cv2
import numpy as np
import pyttsx3
import pafy
import pandas as pd
import time

# Loads the PyTorch model and yolov5 repository for exporting information
model = torch.hub.load('ultralytics/yolov5', 'custom', path= "best.pt")
model.eval()
previous_label = None
frame_count = 0
# Initialises the text-to-speech algorithm
engine = pyttsx3.init()

# Defining the function to create and narrate predictions
def predict(frame):
    global previous_label, frame_count
    # resizing the frame to a square image, matching the input size of the model
    resized_frame = cv2.resize(frame,(1248,1248))
    # Creating predictions and store them
    results = model(frame)
    #Creates a pandas dataframe that stores all predictions, confidences and locations for bounding box
    df = results.pandas().xyxy[0]

    # Storing the labels for each prediction
    labels = {1: 'regulatory--keep-right--g1',
        2: 'regulatory--priority-over-oncoming-vehicles--g1',
        4: 'regulatory--maximum-speed-limit-35--g2',
        6: 'warning--curve-left--g2',
        9: 'warning--pedestrians-crossing--g4',
        12: 'regulatory--maximum-speed-limit-45--g3',
        14: 'regulatory--one-way-right--g2',
        15: 'regulatory--yield--g1',
        16: 'regulatory--one-way-straight--g1',
        17: 'warning--curve-right--g1',
        18: 'regulatory--pedestrians-only--g2',
        20: 'regulatory--no-entry--g1',
        23: 'warning--crossroads--g3',
        26: 'regulatory--no-stopping--g15',
        38: 'regulatory--no-overtaking--g2',
        41: 'regulatory--stop--g1',
        43: 'regulatory--maximum-speed-limit-30--g1',
        48: 'regulatory--end-of-maximum-speed-limit-30--g2',
        52: 'warning--height-restriction--g2',
        54: 'warning--double-curve-first-left--g2',
        58: 'regulatory--turn-right--g1',
        60: 'regulatory--turn-left--g1',
        61: 'regulatory--no-parking-or-no-stopping--g3',
        62: 'warning--roundabout--g1',
        64: 'regulatory--maximqum-speed-limit-60--g1',
        66: 'regulatory--maximum-speed-limit-40--g1',
        69: 'warning--road-bump--g1',
        70: 'warning--uneven-road--g6',
        71: 'regulatory--maximum-speed-limit-50--g1',
        72: 'regulatory--no-parking--g5',
        76: 'regulatory--maximum-speed-limit-100--g1',
        78: 'regulatory--maximum-speed-limit-5--g1',
        81: 'warning--children--g2',
        83: 'regulatory--no-u-turn--g3',
        87: 'regulatory--go-straight-or-turn-left--g1',
        88: 'regulatory--bicycles-only--g1',
        90: 'regulatory--one-way-left--g1',
        93: 'regulatory--give-way-to-oncoming-traffic--g1',
        97: 'warning--narrow-bridge--g1',
        98: 'regulatory--turn-right-ahead--g1',
        100: 'regulatory--maximum-speed-limit-70--g1',
        103: 'regulatory--pass-on-either-side--g2',
        111: 'regulatory--u-turn--g1',
        112: 'regulatory--keep-left--g1',
        113: 'regulatory--go-straight--g1',
        118: 'regulatory--road-closed-to-vehicles--g3',
        119: 'regulatory--no-left-turn--g3',
        122: 'regulatory--no-right-turn--g1',
        129: 'regulatory--maximum-speed-limit-90--g1',
        130: 'regulatory--maximum-speed-limit-110--g1',
        136: 'warning--winding-road-first-left--g1',
        137: 'warning--turn-right--g1',
        144: 'warning--traffic-signals--g3',
        149: 'warning--winding-road-first-right--g3',
        154: 'regulatory--maximum-speed-limit-20--g1',
        155: 'regulatory--maximum-speed-limit-25--g2',
        156: 'regulatory--no-motor-vehicles-except-motorcycles--g2',
        172: 'regulatory--maximum-speed-limit-55--g2',
        175: 'warning--pass-left-or-right--g2',
        180: 'regulatory--maximum-speed-limit-80--g1',
        183: 'warning--roadworks--g1',
        193: 'regulatory--road-closed--g2',
        197: 'warning--school-zone--g2',
        211: 'regulatory--no-turn-on-red--g1',
        213: 'warning--road-narrows-right--g1',
        225: 'regulatory--maximum-speed-limit-led-100--g1',
        227: 'regulatory--maximum-speed-limit-10--g1',
        233: 'regulatory--roundabout--g1',
        236: 'warning--bicycles-crossing--g1',
        241: 'warning--turn-left--g1',
        243: 'warning--stop-ahead--g9',
        244: 'regulatory--mopeds-and-bicycles-only--g1',
        245: 'regulatory--end-of-speed-limit-zone--g1',
        256: 'regulatory--no-motor-vehicles--g1',
        262: 'regulatory--no-straight-through--g1',
        264: 'warning--offset-roads--g3',
        265: 'regulatory--maximum-speed-limit-120--g1',
        266: 'regulatory--go-straight-or-turn-right--g3',
        271: 'regulatory--buses-only--g1',
        283: 'regulatory--maximum-speed-limit-led-80--g1',
        291: 'warning--road-widens-right--g1',
        299: 'regulatory--end-of-maximum-speed-limit-70--g2',
        300: 'warning--traffic-merges-left--g2',
        307: 'regulatory--minimum-safe-distance--g1',
        314: 'regulatory--stop-signals--g1',
        320: 'regulatory--width-limit--g1',
        326: 'regulatory--no-turns--g1',
        327: 'regulatory--maximum-speed-limit-15--g1',
        329: 'regulatory--maximum-speed-limit-led-60--g1',
        337: 'warning--two-way-traffic--g2',
        357: 'warning--road-narrows-left--g1',
        363: 'regulatory--maximum-speed-limit-65--g2',
        392: 'warning--restricted-zone--g1'}
    #Reads all valid predictions
    if pd.api.types.is_numeric_dtype(df["class"]):
        if (df["class"]% 1 ==0).all():

            #Stores all classes and their confidences to find the prediction with the maximum confidence
            confidences = []
            for i in df["confidence"]:
                confidences.append(float(i))
                max_conf = max(confidences)

            #Applies a threshold for the maximum confidence 
            if max_conf >=0.7:

                # Formats the best prediction a string for the text-to-speech engine
                current_prediction =df.loc[df['confidence'] == max_conf,'class'].values[0]
                text = labels[int(current_prediction)]
                text = text.split("--")
                text = str("{} Sign Detected".format(text[1]))

                #Checks if the best prediction has been read in the last 20 frames (10 seconds)
                if text == previous_label:
                    frame_count += 1
                else:
                    previous_label = text
                    frame_count = 0
                if frame_count >= 20:
                    frame_count = 0
                
                #Narrates the best prediction
                if frame_count == 0:
                    engine.say(text)
                    engine.runAndWait()
            
# Initializes the video
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
# Loops over teach frame of the video
numberofframes = 0
start = time.time()
while True:
    numberofframes += 1
    # Reads each frame
    ret, frame = cap.read()

    # Checks if the frame was successfully read
    if not ret:
        break

    # Showing the frame in a window - used for illustrating testing but can comment out
    cv2.imshow("frame", frame)
    # Make a prediction and narrate the results
    predict(frame)

    # Check if the user has pressed the "q" key to terminate the while loop
    if cv2.waitKey(1) == ord("q"):
        break
# Terminates the video
cap.release()
cv2.destroyAllWindows()
# Calculates the duration and framerate
duration = time.time() -start
print(duration)
framerate = numberofframes// duration
print("Framerate: ",framerate)
