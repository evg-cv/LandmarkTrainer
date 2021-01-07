import tensorflow as tf
import numpy as np
import cv2

from settings import MOUTH_TF_MODEL, DETECT_CONFIDENCE


class MouthDetector:

    def __init__(self):

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MOUTH_TF_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect_objects(self, image_np):
        # Expand dimensions since the models expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        return self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                             feed_dict={self.image_tensor: image_np_expanded})

    def detect(self, frame):

        [frm_height, frm_width] = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # st_time = time.time()
        (boxes, scores, classes, _) = self.detect_objects(frame_rgb)
        # print(time.time() - st_time)
        mouth_rect_list = []
        lips_rect_list = []

        for i in range(len(scores[0])):
            if scores[0][i] > DETECT_CONFIDENCE and classes[0][i] == 1:
                # print(scores[0][i])
                x1, y1 = int(boxes[0][i][1] * frm_width), int(boxes[0][i][0] * frm_height)
                x2, y2 = int(boxes[0][i][3] * frm_width), int(boxes[0][i][2] * frm_height)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                mouth_rect_list.append([x1, y1, x2, y2])
            elif scores[0][i] > DETECT_CONFIDENCE and classes[0][i] == 2:
                x1, y1 = int(boxes[0][i][1] * frm_width), int(boxes[0][i][0] * frm_height)
                x2, y2 = int(boxes[0][i][3] * frm_width), int(boxes[0][i][2] * frm_height)
                # if (x2 - x1) * (y2 - y1) / (frm_height * frm_width) > 0.015:
                #     continue
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                lips_rect_list.append([x1, y1, x2, y2])
                # cv2.imshow("door frame", frame)
                # cv2.waitKey()

        return mouth_rect_list, lips_rect_list


if __name__ == '__main__':
    mouth_detector = MouthDetector()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame_ = cap.read()
        # rot_frame = cv2.rotate(frame_, cv2.ROTATE_90_CLOCKWISE)
        # resized_frame = cv2.resize(rot_frame, None, fx=0.5, fy=0.5)
        mouths, lips = mouth_detector.detect(frame=frame_)
        for lip in lips:
            left, top, right, bottom = lip
            cv2.rectangle(frame_, (left, top), (right, bottom), (0, 0, 255), 2)
        for mouth in mouths:
            left, top, right, bottom = mouth
            cv2.rectangle(frame_, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("lip frame", frame_)
        if cv2.waitKey(1) & ord('q') == 0xFF:
            break

    cap.release()
    cv2.destroyAllWindows()
