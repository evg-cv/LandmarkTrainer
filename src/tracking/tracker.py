import cv2
import dlib
import collections
# import time
import numpy as np

from src.lip_detector.detector import MouthDetector
from src.filter.nms import non_max_suppression_slow
from src.filter.tracker_filter import filter_undetected_trackers
from settings import MARGIN

object_detector = MouthDetector()


def estimate_lip_status(lip_rects, frame):

    if len(lip_rects) == 2:
        height, width = frame.shape[:2]
        lip1_cx = 0.5 * (lip_rects[0][0] + lip_rects[0][2])
        lip2_cx = 0.5 * (lip_rects[1][0] + lip_rects[1][2])
        # cv2.rectangle(frame, (lip_rects[0][0], lip_rects[0][1]), (lip_rects[0][2], lip_rects[0][3]), (0, 0, 255), 1)
        # cv2.rectangle(frame, (lip_rects[1][0], lip_rects[1][1]), (lip_rects[1][2], lip_rects[1][3]), (0, 0, 255), 1)
        if abs(lip1_cx - lip2_cx) < lip_rects[0][2] - lip_rects[0][0]:
            lip_diff = abs(min(lip_rects[0][3], lip_rects[1][3]) - max(lip_rects[0][1], lip_rects[1][1]))
            # if lip_diff < MOUTH_THRESH:
            #     cv2.putText(frame, str("Mouth: 0%"), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, height / 500,
            #                 (0, 255, 0), 1)
            # else:
            percent = min(int(lip_diff / max((lip_rects[0][2] - lip_rects[0][0]), (lip_rects[1][2] - lip_rects[1][0]))
                          * 100), 100)
            cv2.putText(frame, str(f"Mouth: {percent}%"), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, height / 500,
                        (0, 255, 0), 1)

    return frame


def track_lips(frame, trackers, attributes):

    all_track_rects = []
    all_track_keys = []

    for fid in trackers.keys():

        tracked_position = trackers[fid].get_position()
        t_left = int(tracked_position.left())
        t_top = int(tracked_position.top())
        t_right = int(tracked_position.right())
        t_bottom = int(tracked_position.bottom())
        all_track_rects.append([t_left, t_top, t_right, t_bottom])
        all_track_keys.append(fid)

    filter_ids = non_max_suppression_slow(boxes=np.array(all_track_rects), keys=all_track_keys)

    for idx in filter_ids:
        attributes.pop(idx)
        trackers.pop(idx)

    lip_rects = []

    for fid in trackers.keys():

        tracked_position = trackers[fid].get_position()
        t_left = int(tracked_position.left())
        t_top = int(tracked_position.top())
        t_right = int(tracked_position.right())
        t_bottom = int(tracked_position.bottom())
        t_center_x = int(0.5 * (t_left + t_right))
        t_center_y = int(0.5 * (t_top + t_bottom))
        lip_rects.append([t_left, t_top, t_right, t_bottom])

        attributes[fid]["centers"].append([t_center_x, t_center_y])
        # cv2.rectangle(frame, (t_left, t_top), (t_right, t_bottom), (0, 0, 255), 3)
    frame = estimate_lip_status(lip_rects=lip_rects, frame=frame)

    return frame, attributes


def create_lip_tracker(detect_img, trackers, attributes, lip_id):

    # st_time = time.time()
    # lip_positions = person_detector.detect_person(frame=img)
    _, lip_positions = object_detector.detect(frame=detect_img)
    # print(time.time() - st_time)

    detected_centers = []

    for coordinates in lip_positions:

        left, top, right, bottom = coordinates
        # cv2.rectangle(detect_img, (left, top), (right, bottom), (0, 0, 255), 2)
        x_bar = 0.5 * (left + right)
        y_bar = 0.5 * (top + bottom)
        detected_centers.append([left, top, right, bottom])

        matched_fid = None

        for fid in trackers.keys():

            tracked_position = trackers[fid].get_position()
            t_left = int(tracked_position.left())
            t_top = int(tracked_position.top())
            t_right = int(tracked_position.right())
            t_bottom = int(tracked_position.bottom())

            # calculate the center point
            t_x_bar = 0.5 * (t_left + t_right)
            t_y_bar = 0.5 * (t_top + t_bottom)

            # check if the center point of the face_detector is within the rectangle of a tracker region.
            # Also, the center point of the tracker region must be within the region detected as a face_detector.
            # If both of these conditions hold we have a match

            if t_left <= x_bar <= t_right and t_top <= y_bar <= t_bottom and left <= t_x_bar <= right \
                    and top <= t_y_bar <= bottom:
                matched_fid = fid
                trackers.pop(fid)
                tracker = dlib.correlation_tracker()
                tracker.start_track(detect_img, dlib.rectangle(left - MARGIN, top - MARGIN, right + MARGIN,
                                                               bottom + MARGIN))
                trackers[matched_fid] = tracker
                attributes[matched_fid]["undetected"] = 0

                # cv2.rectangle(detect_img, (t_left, t_top), (t_right, t_bottom), (0, 0, 255), 3)
        # If no matched fid, then we have to create a new tracker
        if matched_fid is None:
            # print("Creating new tracker " + str(lip_id))
            # Create and store the tracker
            tracker = dlib.correlation_tracker()
            tracker.start_track(detect_img, dlib.rectangle(left - MARGIN, top - MARGIN, right + MARGIN,
                                                           bottom + MARGIN))
            trackers[lip_id] = tracker

            temp_dict = collections.defaultdict()
            temp_dict["id"] = str(lip_id)
            temp_dict["centers"] = [[x_bar, y_bar]]
            temp_dict["undetected"] = 0
            attributes[lip_id] = temp_dict
            # cv2.rectangle(detect_img, (left, top), (right, bottom), (0, 0, 255), 3)

            # Increase the currentFaceID counter
            lip_id += 1

    trackers, attributes = filter_undetected_trackers(trackers=trackers, attributes=attributes,
                                                      detected_rects=detected_centers)
    lip_rects = []
    for fid in trackers.keys():

        tracked_position = trackers[fid].get_position()
        t_left = int(tracked_position.left())
        t_top = int(tracked_position.top())
        t_right = int(tracked_position.right())
        t_bottom = int(tracked_position.bottom())
        lip_rects.append([t_left, t_top, t_right, t_bottom])
    detect_img = estimate_lip_status(lip_rects=lip_rects, frame=detect_img)

    return trackers, attributes, lip_id, detect_img


if __name__ == '__main__':

    track_lips(frame=cv2.imread(""), trackers={}, attributes={})
