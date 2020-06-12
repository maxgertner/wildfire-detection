
import cv2
import os
import sys
import math
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


from twilio.rest import Client
from flask import Flask, render_template
import time
import imutils
import datetime

from flask_opencv_streamer.streamer import Streamer
import datetime
import time
import numpy


def construct_fire_detector (x,y, training=False):
    
    network = input_data(shape=[None, y, x, 3])

    conv1_7_7 = conv_2d(network, 64, 5, strides=2, activation='relu',
                        name = 'conv1_7_7_s2')

    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)

    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',
                               name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 128,3, activation='relu',
                        name='conv2_3_3')

    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2,
                            name='pool2_3_3_s2')

    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu',
                               name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu',
                                      name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,
                               activation='relu', name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu',
                                      name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5,
                               activation='relu', name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1,
                                    activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5,
                                 inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu',
                               name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1,
                                      activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,
                               activation='relu',name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1,
                                      activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96,
                               filter_size=5,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,
                                    name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,
                                    activation='relu', name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2,
                            name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu',
                               name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu',
                                      name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,
                               activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1,
                                      activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,
                               activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,
                                    name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1,
                                    activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3,
                                 inception_4a_5_5, inception_4a_pool_1_1],
                                mode='concat', axis=3, name='inception_4a_output')

    pool5_7_7 = avg_pool_2d(inception_4a_output, kernel_size=5, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(loss, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)
    else:
        network = loss;

    model = tflearn.DNN(network, checkpoint_path='inceptiononv1onfire',
                        max_checkpoints=1, tensorboard_verbose=3)
    
    return model



# network input sizes
rows = 224
cols = 224

#create the cnn model

model = construct_fire_detector(rows,cols , training=True)
print("Constructed SP-InceptionV1-OnFire ...")

model.load(os.path.join("models/SP-InceptionV1-OnFire",
                         "sp-inceptiononv1onfire"),weights_only=True)
print("Loaded CNN network weights ...")


# display and loop settings

windowName = "Analyzed video"
keepProcessing = True;

MY_NUMBER = '+65022234175'

account_sid = "AC7bc55574057f033caa8f149901bbb31b"
auth_token = "d5f738d7f380aa68de8142b5c96bc790"

video = cv2.VideoCapture(1)
print("Loaded video ...")

    # create window

#cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

#client = Client(account_sid, auth_token)
    # get video properties

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
frame_time = round(1000/fps);

frame_count = 0

port = 3030
require_login = False
streamer = Streamer(port, require_login)

#def send_message(body):
 #   client = Client(account_sid, auth_token)
  #  client.messages.create(
   #   to='+16502234175',
     # from_="+18312784028",
    #  body="A fire has been detected!")

while (keepProcessing):

        # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount();

        # get video frame from file, handle end of file

    ret, frame = video.read()
    if not ret:
            print("... end of video file reached");
            break;

    # re-size image to network input size and perform prediction
    #new_frame = cv2.imshow('fire', 'fire.jpg')
    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

    unprocessed_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

    #cv2.imshow('Raw-video', unprocessed_frame)

        # OpenCV imgproc SLIC superpixels implementation below

    slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
    slic.iterate(10)

        # getLabels method returns the different superpixel segments
    segments = slic.getLabels()
        #print(len(np.unique(segments)))
    
        # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(segments)):

            # Construct a mask for the segment
        mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255

            # get contours (first checking if OPENCV >= 4.x)

        if (int(cv2.__version__.split(".")[0]) >= 4):
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
    
        superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)
        #cv2.imshow("superpixel", small_frame);
        #print(superpixel)
            # use loaded model to make prediction on given superpixel segments
        output = model.predict([superpixel])
        
        cv2.drawContours(unprocessed_frame, contours, -1, (255,0,0), 1)

        #cv2.imshow('Segmented video', unprocessed_frame)

        


        if output[0][0] >= 0.70:
                # draw the contour
                # if prediction for FIRE was TRUE (round to 1), draw GREEN contour for superpixel
                    #Then save the image of the fire and send it my phone
            cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)
            cv2.imwrite('fire.png', frame)
            #send_message('fire')
        else:
                # if prediction for FIRE was FALSE, draw RED contour for superpixel
            cv2.drawContours(small_frame, contours, -1, (0,0,2255), 1)

        # stop the timer and convert to ms. (to see how long processing and display takes)

    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) *1000;

        # image display and key handling
    #cv2.imshow(windowName, small_frame);
    
    # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
    frame_count = frame_count + 1
    streamer.update_frame(small_frame)
    
    if not streamer.is_streaming:
        streamer.start_streaming()

    cv2.waitKey(30)
    
    key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
    if (key == 27):
        keepProcessing = False
    elif (key == ord('f')):
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.destroyAllWindows()

