import cv2

from imutils.video import FPS
# from src.tracking.tracker import create_lip_tracker, track_lips
from src.lip_detector.detector import MouthDetector
from src.tracking.tracker import estimate_lip_status
from settings import VIDEO_PATH, TRACK_QUALITY, TRACK_CYCLE


class MouthStatue:

    def __init__(self):

        self.lip_trackers = {}
        self.current_lip_id = 1
        self.lip_attributes = {}
        self.mouth_detector = MouthDetector()

    def main(self):

        if VIDEO_PATH == "":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(VIDEO_PATH)
        # cnt = 0
        fps = FPS().start()
        while True:

            _, show_img = cap.read()
            # rot_frame = cv2.rotate(show_img, cv2.ROTATE_90_CLOCKWISE)
            # resized_image = cv2.resize(show_img, None, fx=0.5, fy=0.5)
            # resized_image = show_img
            _, lip_positions = self.mouth_detector.detect(frame=show_img)
            lips = []
            for coordinates in lip_positions:
                left, top, right, bottom = coordinates
                lips.append([left, top, right, bottom])

            result_img = estimate_lip_status(lip_rects=lips, frame=show_img)

            # fids_to_delete = []
            # for fid in self.lip_trackers.keys():
            #     tracking_quality = self.lip_trackers[fid].update(resized_image)
            #
            #     # If the tracking quality is good enough, we must delete this tracker
            #     if tracking_quality < TRACK_QUALITY:
            #         fids_to_delete.append(fid)
            #
            # for fid in fids_to_delete:
            #     # print("Removing fid " + str(fid) + " from list of trackers")
            #     self.lip_trackers.pop(fid, None)
            #     self.lip_attributes.pop(fid, None)
            #
            # if cnt % TRACK_CYCLE == 0:
            #     self.lip_trackers, self.lip_attributes, self.current_lip_id, result_img = \
            #         create_lip_tracker(detect_img=resized_image, trackers=self.lip_trackers,
            #                                attributes=self.lip_attributes, lip_id=self.current_lip_id)
            # else:
            #     result_img, self.lip_attributes = track_lips(frame=resized_image,
            #                                                      trackers=self.lip_trackers,
            #                                                      attributes=self.lip_attributes)
            #
            # cnt += 1

            cv2.imshow("image", result_img)
            # time.sleep(0.05)
            fps.update()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        # kill open cv things
        fps.stop()
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    MouthStatue().main()
