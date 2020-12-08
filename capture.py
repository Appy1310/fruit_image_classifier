import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam, predict_frame, predict_frame_mobilenet

import tensorflow.keras as keras
from tensorflow.keras.models import load_model

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import preprocessing 
from datetime import datetime



model = load_model("./models/model_mobilenet_allfruits.h5")
model_mobilenet = load_model('./models/Base_model_Mobilenet.h5')
classes = ['apple','avocado', 'banana',  'kaki', 'lemon', 'orange', 'pumpkin']
last_detected = datetime.now()



if __name__ == "__main__":

    # folder to write images to
    #out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # 640x360
    # 640.0 x 480.0
    #webcam = init_cam(640, 480)
    webcam = init_cam(640, 360)
    key = None

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
            # get key event
            key = key_action()

            # draw a [224x224] rectangle into the middle of the frame
            cv2.rectangle(frame,(0+88,0+8),(224+88,224+8),(0,0,0),2)
            #cv2.rectangle(frame,(0+880,0+80),(448+880,448+80),(0,0,0),2) 

            image = frame[0+8: 224+8, 0+88: 224+88, :]
            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                #image = frame[0+8: 224+8, 0+88: 224+88, :]
                write_image(out_folder, image)  
            
            
            if key == 'p':
                last_detected = datetime.now()                              
            elif (datetime.now() - last_detected).total_seconds() < 5:
                                      
                    # write the predictions on the frame
                    # find the predictions
                    predictions = predict_frame(image, model, classes)
                    #print(predictions)
                    cv2.putText(frame, predictions, (10, 300),
                            cv2.FONT_HERSHEY_PLAIN, .6, (0,0,0),
                            1, cv2.LINE_AA)            
            
        

            # if key == 's':
            #     last_detected = datetime.now()
            # elif (datetime.now() - last_detected).total_seconds() < 3:
            #         cv2.putText(frame, 'Are you satisfied? Otherwise press m', (10, 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, .3, (255,153,51),
            #                 1, cv2.LINE_AA)

            # if key == 'm':
            #     last_detected = datetime.now()

            # elif (datetime.now() - last_detected).total_seconds() < 3:
            #     predictions_mobilenet = predict_frame_mobilenet(image, model_mobilenet, classes)
            #     cv2.putText(frame, predictions_mobilenet, (10, 100),
            #                 cv2.FONT_HERSHEY_SIMPLEX, .3, (255,153,51),
            #                 1, cv2.LINE_AA)

                           

             
            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()

    predictions_mobilenet = predict_frame_mobilenet(image, model_mobilenet, classes)
    print(predictions_mobilenet)    
