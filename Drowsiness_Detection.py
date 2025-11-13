# drowsiness_detector.py
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time

def eye_aspect_ratio(eye):
    # compute euclidean distances for the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # horizontal distance (eye width)
    C = distance.euclidean(eye[0], eye[3])
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Hyperparameters (tune experimentally)
EAR_THRESHOLD = 0.25    # below this EAR -> eye considered closed
CONSEC_FRAMES = 20     # number of consecutive frames before alert

# Initialize dlib's face detector (HOG) and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Grab the indexes for the left and right eye from the 68-landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Video capture
cap = cv2.VideoCapture(0)
flag = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize to speed up and convert to grayscale
        frame = imutils.resize(frame, width=660)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        rects = detector(gray, 0)

        for rect in rects:
            # get facial landmarks and convert to NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # compute EAR for both eyes and average
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # draw eye contours for visualization
            leftHull = cv2.convexHull(leftEye)
            rightHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

            # check if EAR is below the threshold
            if ear < EAR_THRESHOLD:
                flag += 1
                # if eyes closed for long enough, alert
                if flag >= CONSEC_FRAMES:
                    cv2.putText(frame, "**************************ALERT!*******************************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "**************************ALERT!*******************************", (10, 480),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag = 0  # reset counter when eyes open

        # show frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
