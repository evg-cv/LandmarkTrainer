import dlib
import cv2

from settings import FACE_DETECTOR_MODEL


class FaceDetector:

    def __init__(self):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_MODEL)

    def detect_face(self, frame_path):
        face_rects = []
        image = cv2.imread(frame_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_cnn = self.cnn_face_detector(gray_image, 1)
        # loop over detected faces
        for face in faces_cnn:
            face_rects.append(face)
            # draw box over face_detector
            # cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        return face_rects


if __name__ == '__main__':

    FaceDetector().detect_face(frame_path="")
