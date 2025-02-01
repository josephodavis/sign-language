import copy
import itertools
import mediapipe as mp
import cv2
import keras
import numpy as np

# prepare landmarks to be normalized based on the frame
def calculateLandmarkList(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # keypoint
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point

# normalize all landmarks to be between 0, 1
def normalize(landmarkList):
    temp = copy.deepcopy(landmarkList)

    # convert to relative coordinates --> distance from palm
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp):
        # palm landmark
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1],  landmark_point[2]

        temp[index][0] = temp[index][0] - base_x
        temp[index][1] = temp[index][1] - base_y
        temp[index][2] = temp[index][2] - base_z

    # convert to one-dimensional list
    temp = list(itertools.chain.from_iterable(temp))

    # normalize for landmarks to be in relation to each other
    max_value = max(list(map(abs, temp)))
    def normalizeMore(n):
        return n / max_value

    temp = list(map(normalizeMore, temp))

    return temp

def prediction(frame, landmark_list):
    values = normalize(calculateLandmarkList(frame, hand_landmarks))

    values = np.array(values, dtype=np.float32)  # Convert list to NumPy array
    values = values.reshape(1, -1)  # Reshape to match model input shape (1, 63)
    res = model.predict(values)

    pred = np.argmax(res)
    max_confidence = np.max(res)

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

    return letters[pred], max_confidence

model = keras.models.load_model('/Users/josephdavis/PycharmProjects/MediaPipe_Test/model.keras')

mpHands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
styles = mp.solutions.drawing_styles

with (mpHands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands):

    cam = cv2.VideoCapture(1)
    while True:
        ret, frame = cam.read()

        frame.flags.writeable = True
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        res = hands.process(frame)

        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mpHands.HAND_CONNECTIONS,
                    styles.get_default_hand_landmarks_style(),
                    styles.get_default_hand_connections_style()
                )
                pred, confidence = prediction(frame, hand_landmarks)
                frame = cv2.flip(frame, 1)
                frame = cv2.putText(frame, pred,
                                    (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 10,
                                    (0, 0, 0), 10)
                frame = cv2.putText(frame, "confidence: " + str(confidence),
                                    (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (0, 0, 0), 5)
                frame = cv2.flip(frame, 1)

        frame = cv2.flip(frame, 1)

        cv2.imshow('hand', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()