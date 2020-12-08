import logging
import os
from datetime import datetime
import cv2
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import preprocessing 
import numpy as np

classes = ['apple','avocado', 'banana',  'kaki', 'lemon', 'orange', 'pumpkin']

def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S_%f") 
    filename = f'{out}{dt_string}.png'
    logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)


def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button
        return 'q'
    if k == 32: # space bar
        return 'space'
    if k == 112: # p key
        return 'p'
    if k == 115: # s key
        return 's' 
    if k == 109: # m key
        return 'm'       
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def add_text(text, frame):
    # Put some rectangular box on the image
    # cv2.putText()
    return NotImplementedError


def predict_frame(frame, model, classes):
    # convert from bgr to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # continue with the pre-processing
    numpy_image = keras.preprocessing.image.img_to_array(frame_rgb)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = keras.applications.mobilenet_v2.preprocess_input(image_batch)
    
    # make a prediction
    predictions = model.predict(processed_image)

    # make a list with class and prediction probability
    predictions_list = []

    for i in range(len(classes)):
        predictions_temp = [classes[i], predictions[0,i].round(3)]
        predictions_list.append(predictions_temp)
    #convert list to string for writing on the frame
    # using list comprehension 
    predictions_str = ' '.join([str(elem) for elem in predictions_list]) 
    # convert the numpy array to string for writing on the frame
    # predictions = np.array2string(predictions, precision=3, 
    #                               separator=', ', suppress_small=True)


    return predictions_str

def predict_frame_mobilenet(frame, model, classes):
    # convert from bgr to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # continue with the pre-processing
    numpy_image = keras.preprocessing.image.img_to_array(frame_rgb)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = keras.applications.mobilenet_v2.preprocess_input(image_batch)
    
    # make a prediction
    predictions = model.predict(processed_image)   

    label_mobilenet = keras.applications.mobilenet_v2.decode_predictions(
    predictions, top=5) 

    return label_mobilenet