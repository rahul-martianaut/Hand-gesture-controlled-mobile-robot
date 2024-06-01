#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils.box_utils import calc_bounding_rect, calc_landmark_list, pre_process_landmark, logging_csv, draw_landmarks, draw_bounding_rect, draw_info, draw_info_text
from utils.box_utils import get_args, select_mode 
from model import KeyPointClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))



class GestureRecognitionNode(Node):
    
    def __init__(self):
        super().__init__('gesture_recognition_node')
        self.publisher_ = self.create_publisher(String, 'gesture', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)


    def timer_callback(self):

        use_brect = True

        # Camera preparation 
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 534)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 300)

        # Model load 
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        keypoint_classifier = KeyPointClassifier()

        

        # Read labels 
        with open('/home/faps/hand-gesture_ws/src/hand_gesture_control/scripts/model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            keypoint_classifier_labels_reader = csv.reader(f)   
            keypoint_classifier_labels = list(keypoint_classifier_labels_reader)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ] #flat list
            
        
        # FPS Measurement
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        mode = 0

        while True:
            fps = cvFpsCalc.get()

            # Process Key (ESC: end) 
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)
            #print(mode)

            # Camera capture 
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation 
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True


            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list
                                )

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    gesture = hand_sign_id 
                    
                    msg = String()
                    msg.data = str(gesture)
                    self.publisher_.publish(msg)

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    # debug_image = draw_info_text(
                    #     debug_image,
                    #     brect,
                    #     handedness,
                    #     keypoint_classifier_labels[hand_sign_id],
                        
                    # )
            

            debug_image = draw_info(debug_image, fps, mode, number)

            # Screen reflection
            cv.imshow('Hand Gesture Recognition', debug_image)

            

        cap.release()
        cv.destroyAllWindows()



def main(args=None):
    rclpy.init(args=args)

    node = GestureRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
