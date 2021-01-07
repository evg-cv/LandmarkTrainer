import cv2
import dlib

from imutils import face_utils
from src.face_detector.detector import FaceDetector
from settings import PROFILE_MOUTH_MODEL


class LandMarkDetector:

    def __init__(self):
        self.face_detector = FaceDetector()
        self.predictor = dlib.shape_predictor(PROFILE_MOUTH_MODEL)

    def detect_mouth_landmarks(self, frame_path):
        faces = self.face_detector.detect_face(frame_path=frame_path)
        image = cv2.imread(frame_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for face in faces:
            left = face.rect.left()
            top = face.rect.top()
            right = face.rect.right()
            bottom = face.rect.bottom()
            # draw box over face_detector
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            shape = self.predictor(gray_image, face.rect)
            shape = face_utils.shape_to_np(shape)

            for (left, top) in shape:
                cv2.circle(image, (left, top), 1, (0, 0, 255), -1)
                cv2.imshow("point landmark", image)
                cv2.waitKey()

        cv2.imshow("landmark face_detector", image)
        cv2.waitKey()


if __name__ == '__main__':
    LandMarkDetector().detect_mouth_landmarks(frame_path="")
