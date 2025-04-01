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

        self.output_dir = "."
        

def _process_video(self, video):
    cap = cv2.VideoCapture(video)

    if (cap.isOpened() == False):
        sys.exit('Error opening video stream or file')

    with mp_holistic.Holistic(min_detection_confidence=self.MIN_DETECTION_CONFIDENCE, min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE) as holistic:
        for frame_num in range(self.MAX_FRAME_NUM):
            ret, frame = cap.read()

            if ret == False:
                frame = np.zeros((512, 512, 3), np.uint8)
            
            image, results = detect_landmarks(frame, holistic)
            #_draw_landmarks(image, results)
            keypoints = extract_keypoints(results)

            #utils.create_folder(save_dir)
            #file_name = os.path.join(save_dir, str(frame_num))
            #np.save(file_name, keypoints)

            #cv2.imshow(image)
            cv2_imshow(image)
        cap.release()
        cv2.destroyAllWindows()