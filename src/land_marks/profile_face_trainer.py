import os
import glob
import cv2
import dlib
import ntpath
import xml.etree.ElementTree as ET

from xml.etree.ElementTree import Element, SubElement
from xml.dom import minidom
from settings import PROFILE_FACE_XML, FACE_DETECTOR_MODEL


def create_xml_profile_face():
    cnn_face_detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_MODEL)
    frame_dir = "/home/main/Downloads/cfp-dataset/Data/Images"
    coordinate_dir = "/home/main/Downloads/cfp-dataset/Data/Fiducials"
    image_sub_dirs = os.walk(frame_dir)
    coordinate_sub_dirs = os.walk(coordinate_dir)
    root = Element('dataset')
    name = SubElement(root, 'name')
    name.text = "iBUG face_detector point dataset - training images"
    comment = SubElement(root, 'comment')
    images = SubElement(root, 'images')
    comment.text = "This folder contains data downloaded from: \n" \
                   "http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/ \n The dataset is actually a " \
                   "combination of the AFW, HELEN, iBUG, and LFPW face_detector landmark datasets.  \n But the iBUG people " \
                   "have aggregated it all together and gave them a consistent set of 68 landmarks across all the " \
                   "images, \n thereby turning it into one big dataset. \n Note that we have adjusted the " \
                   "coordinates of the points from the MATLAB convention of 1 being the first index to 0 being the " \
                   "first index. \n So the coordinates in this file are in the normal C 0-indexed coordinate system. " \
                   "\n We have also added left right flips (i.e. mirrors) of each image and also appropriately " \
                   "flipped the landmarks. \n This doubles the size of the dataset. \n Each of the mirrored versions " \
                   "of the images has a filename that ends with _mirror.jpg. \n Finally, note that the bounding" \
                   "boxes are from dlib's default face_detector detector. \n For the faces the detector failed to detect, " \
                   "we guessed at what the bounding box would have been had the detector found it and used that. "

    for image_sub_dir, coordinate_sub_dir in zip(image_sub_dirs, coordinate_sub_dirs):

        frames = glob.glob(os.path.join(image_sub_dir[0], 'profile', "*.*"))
        for frame_path in frames:
            txt_file_name = ntpath.basename(frame_path).replace(".jpg", ".txt")
            co_txt = os.path.join(coordinate_sub_dir[0], "profile", txt_file_name)
            frame = cv2.imread(frame_path)

            height, width = frame.shape[:2]
            if height > 1000 or width > 1000:
                continue
            file = open(co_txt, 'r')
            coordinates = file.read().split("\n")[:-1]
            if len(coordinates) != 30:
                continue

            if float(coordinates[0].split(",")[0]) > float(coordinates[9].split(",")[0]):
                profile_dir = "right"
            else:
                profile_dir = "left"

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_cnn = cnn_face_detector(gray_image, 1)
            if not faces_cnn:
                continue
            print(frame_path)
            left = max(faces_cnn[0].rect.left(), 0)
            top = max(faces_cnn[0].rect.top(), 0)
            right = faces_cnn[0].rect.right()
            bottom = faces_cnn[0].rect.bottom()
            image = SubElement(images, 'image')
            image.set('file', frame_path)
            box = SubElement(image, 'box')
            box.set('top', str(top))
            box.set('left', str(left))
            box.set('width', str(right - left))
            box.set('height', str(bottom - top))
            for i, coordinate in enumerate(coordinates):
                if i in [8, 18, 19, 20, 21, 22, 26, 27, 28, 29]:
                    part = SubElement(box, 'part')
                    if i == 8:
                        part.set('name', '9')
                    elif i == 18:
                        part.set('name', '28')
                    elif i == 19:
                        part.set('name', '29')
                    elif i == 20:
                        part.set('name', '30')
                    elif i == 21:
                        part.set('name', '31')
                    elif i == 22:
                        part.set('name', '34')
                    elif i == 26:
                        part.set('name', '52')
                    elif i == 27:
                        part.set('name', '63')
                    elif i == 28:
                        part.set('name', '58')
                    elif i == 29:
                        part.set('name', '67')
                # else:
                #     if profile_dir == "left":
                #         if i == 0:
                #             part.set('name', '1')
                #         elif i == 1:
                #             part.set('name', '2')
                #         elif i == 2:
                #             part.set('name', '3')
                #         elif i == 3:
                #             part.set('name', '4')
                #         elif i == 4:
                #             part.set('name', '5')
                #         elif i == 5:
                #             part.set('name', '6')
                #         elif i == 6:
                #             part.set('name', '7')
                #         elif i == 7:
                #             part.set('name', '8')
                #         elif i == 9:
                #             part.set('name', '18')
                #         elif i == 10:
                #             part.set('name', '20')
                #         elif i == 11:
                #             part.set('name', '22')
                #         elif i == 12:
                #             part.set('name', '37')
                #         elif i == 13:
                #             part.set('name', '38')
                #         elif i == 14:
                #             part.set('name', '39')
                #         elif i == 15:
                #             part.set('name', '40')
                #         elif i == 16:
                #             part.set('name', '42')
                #         elif i == 17:
                #             part.set('name', '41')
                #         elif i == 23:
                #             part.set('name', '33')
                #         elif i == 24:
                #             part.set('name', '32')
                #         elif i == 25:
                #             part.set('name', '49')
                #     else:
                #         if i == 0:
                #             part.set('name', '17')
                #         elif i == 1:
                #             part.set('name', '16')
                #         elif i == 2:
                #             part.set('name', '15')
                #         elif i == 3:
                #             part.set('name', '14')
                #         elif i == 4:
                #             part.set('name', '13')
                #         elif i == 5:
                #             part.set('name', '12')
                #         elif i == 6:
                #             part.set('name', '11')
                #         elif i == 7:
                #             part.set('name', '10')
                #         elif i == 9:
                #             part.set('name', '27')
                #         elif i == 10:
                #             part.set('name', '25')
                #         elif i == 11:
                #             part.set('name', '23')
                #         elif i == 12:
                #             part.set('name', '46')
                #         elif i == 13:
                #             part.set('name', '45')
                #         elif i == 14:
                #             part.set('name', '44')
                #         elif i == 15:
                #             part.set('name', '43')
                #         elif i == 16:
                #             part.set('name', '47')
                #         elif i == 17:
                #             part.set('name', '48')
                #         elif i == 23:
                #             part.set('name', '35')
                #         elif i == 24:
                #             part.set('name', '36')
                #         elif i == 25:
                #             part.set('name', '55')

                    part.set("x", str(int(float(coordinate.split(",")[0]))))
                    part.set("y", str(int(float(coordinate.split(",")[1]))))

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    f = open(PROFILE_FACE_XML, "w")
    f.write(xml_str)

    return


if __name__ == '__main__':
    create_xml_profile_face()
