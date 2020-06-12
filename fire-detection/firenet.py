import cv2
import os
import sys
import math
import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression


from flask_opencv_streamer.streamer import Streamer

def construct_firenet (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################


from flask import Flask, render_template, Response

rows = 224
cols = 224
    
video = cv2.VideoCapture(0)

model = construct_firenet (rows, cols, training=False)

port = 3030
require_login = False
streamer = Streamer(port, require_login)

while True:

    ret, frame = video.read()

    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = model.predict([small_frame])

    if round(output[0][0]) == 1:
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
        cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
            cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
    else:
        cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
        cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
            cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

    
    streamer.update_frame(frame)

    if not streamer.is_streaming:
        streamer.start_streaming()

    cv2.waitKey(30)




    
