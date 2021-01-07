import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(CUR_DIR, 'utils', 'model')
FACE_DETECTOR_MODEL = os.path.join(MODEL_DIR, 'mmod_human_face_detector.dat')
LANDMARK_MODEL = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
MOUTH_TRAIN_XML = os.path.join(MODEL_DIR, 'lip_detector.xml')
MOUTH_MODEL = os.path.join(MODEL_DIR, 'lip_detector.dat')
PROFILE_FACE_XML = os.path.join(MODEL_DIR, 'profile_face.xml')
PROFILE_MOUTH_MODEL = os.path.join(MODEL_DIR, 'profile_mouth.dat')
MOUTH_TF_MODEL = os.path.join(MODEL_DIR, 'lip_model_v1.pb')

DETECT_CONFIDENCE = 0.6
MOUTH_THRESH = 5
UNDETECTED_THRESH = 1
MARGIN = 0
TRACK_QUALITY = 2
TRACK_CYCLE = 20
OVERLAP_THRESH = 0.7


VIDEO_PATH = ""
