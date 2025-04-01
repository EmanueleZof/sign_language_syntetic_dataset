import cv2
import sys
import numpy as np
import mediapipe as mp

class Preprocessor:
    def __init__(self):
        self.MAX_FRAME_NUM = 100
        self.MIN_DETECTION_CONFIDENCE = 0.5
        self.MIN_TRACKING_CONFIDENCE = 0.5

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        
    def _draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks( image, 
                                        results.face_landmarks, 
                                        mp_holistic.FACEMESH_TESSELATION, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

        self.mp_drawing.draw_landmarks( image, 
                                        results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        self.mp_drawing.draw_landmarks( image, 
                                        results.left_hand_landmarks, 
                                        mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        self.mp_drawing.draw_landmarks( image, 
                                        results.pose_landmarks, 
                                        mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    def _process_video(self, video, show=False):
        cap = cv2.VideoCapture(video)

        if (cap.isOpened() == False):
            sys.exit('Error opening video stream or file')

        with self.mp_holistic.Holistic(min_detection_confidence=self.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE) as holistic:
            for frame_num in range(self.MAX_FRAME_NUM):
                ret, frame = cap.read()

                if ret == False:
                    frame = np.zeros((512, 512, 3), np.uint8)
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if (show == True):
                    self._draw_landmarks(image, results)
                    cv2_imshow(image)

                #keypoints = extract_keypoints(results)

                #utils.create_folder(save_dir)
                #file_name = os.path.join(save_dir, str(frame_num))
                #np.save(file_name, keypoints)

                #cv2.imshow(image)
                
            cap.release()
            cv2.destroyAllWindows()

    def process(self, file, output_dir="."):
        self._process_video(file, True)