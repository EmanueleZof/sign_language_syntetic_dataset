import os
import cv2
import sys
import csv
import numpy as np
import mediapipe as mp

import library.utils as Utils

from tqdm import tqdm
from google.colab.patches import cv2_imshow

class Preprocessor:
    def __init__(self):
        self.MAX_FRAME_NUM = 100
        self.MIN_DETECTION_CONFIDENCE = 0.5
        self.MIN_TRACKING_CONFIDENCE = 0.5
        self.POSE_KEYPOINTS = 33
        self.FACE_KEYPOINTS = 468
        self.LH_KEYPOINTS = 21
        self.RH_KEYPOINTS = 21
        self.OUTPUT_FILE = Utils.OUTPUT_DIR + "keyframes.csv"

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        Utils.create_dir(Utils.OUTPUT_DIR)
        self._scaffold_landmarks()
    
    def _save_csv_file(self, file_name, write_mode, data):
        with open(file_name, mode=write_mode, newline="") as f:
            csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(data)

    def _scaffold_landmarks(self):
        num_coords = self.POSE_KEYPOINTS + self.FACE_KEYPOINTS + self.LH_KEYPOINTS + self.RH_KEYPOINTS
        landmarks = ["class"]

        for val in range(1, num_coords+1):
            landmarks += ["x{}".format(val), "y{}".format(val), "z{}".format(val), "v{}".format(val)]

        self._save_csv_file(self.OUTPUT_FILE, "w", landmarks)

    def _draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks( image, 
                                        results.face_landmarks, 
                                        self.mp_holistic.FACEMESH_TESSELATION, 
                                        self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

        self.mp_drawing.draw_landmarks( image, 
                                        results.right_hand_landmarks, 
                                        self.mp_holistic.HAND_CONNECTIONS, 
                                        self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        self.mp_drawing.draw_landmarks( image, 
                                        results.left_hand_landmarks, 
                                        self.mp_holistic.HAND_CONNECTIONS, 
                                        self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        self.mp_drawing.draw_landmarks( image, 
                                        results.pose_landmarks, 
                                        self.mp_holistic.POSE_CONNECTIONS, 
                                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    def _get_pose_keypoints(self, landmarks):
        if landmarks:
            pose = []
            for res in landmarks.landmark:
                pose.append(np.array([res.x, res.y, res.z, res.visibility]))
            return list(np.array(pose).flatten())
        
        return list(np.zeros(self.POSE_KEYPOINTS*4))

    def _get_face_keypoints(self, landmarks):
        if landmarks:
            face = []
            for res in landmarks.landmark:
                face.append(np.array([res.x, res.y, res.z, 0.0]))
            return list(np.array(face).flatten())
        
        return list(np.zeros(self.FACE_KEYPOINTS*4))

    def _get_left_hand_keypoints(self, landmarks):
        if landmarks:
            lh = []
            for res in landmarks.landmark:
                lh.append(np.array([res.x, res.y, res.z, 0.0]))
            return list(np.array(lh).flatten())
        
        return list(np.zeros(self.LH_KEYPOINTS*4))

    def _get_right_hand_keypoints(self, landmarks):
        if landmarks:
            rh = []
            for res in landmarks.landmark:
                rh.append(np.array([res.x, res.y, res.z, 0.0]))
            return list(np.array(rh).flatten())
        
        return list(np.zeros(self.RH_KEYPOINTS*4))

    def _extract_keypoints(self, results):
        pose = self._get_pose_keypoints(results.pose_landmarks)
        face = self._get_face_keypoints(results.face_landmarks)
        lh = self._get_left_hand_keypoints(results.left_hand_landmarks)
        rh = self._get_right_hand_keypoints(results.right_hand_landmarks)

        return pose + face + lh + rh

    def _process_video(self, video, class_name, show=False):
        cap = cv2.VideoCapture(video)

        if (cap.isOpened() == False):
            sys.exit('Error opening video stream or file')

        with self.mp_holistic.Holistic(min_detection_confidence=self.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE) as holistic:
            for frame_num in range(self.MAX_FRAME_NUM):
                ret, frame = cap.read()

                if (ret == False):
                    frame = np.zeros((512, 512, 3), np.uint8)
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if (show == True):
                    self._draw_landmarks(image, results)
                    cv2_imshow(image)

                if (ret == True):
                    keypoints = self._extract_keypoints(results)
                    keypoints.insert(0, class_name)

                self._save_csv_file(self.OUTPUT_FILE, "a", keypoints)
                
            cap.release()
            cv2.destroyAllWindows()

    def process(self, files_dir, class_name, show=False):
        for file in tqdm(os.listdir(files_dir)):
            print(file)

            #self._process_video(file, class_name, show)