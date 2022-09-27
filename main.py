import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import keras.models

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    out = np.concatenate([pose, face, left, right])
    return out

def create_action_dir(file_path,actions, no_sequences):
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(file_path,action, str(sequence)))
            except:
                pass
    print("Folders created........")

def create_dataset(file_path, actions, no_sequences, sequence_length):
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):

                    ret, frame = cap.read()
                    frame, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame, results)

                    if frame_num == 0:
                        cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(200)
                    else:
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)

                    landmarks = extract_landmarks(results)
                    npy_path = os.path.join(file_path, action, str(sequence), str(frame_num))
                    np.save(npy_path, landmarks)
                    if cv2.waitKey(10) == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()

def create_features_and_labels(file_path, actions, no_sequences, sequence_length):
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(file_path, action, str(sequence), f'{frame_num}.npy'))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    sequences = np.array(sequences)
    labels = to_categorical(labels).astype(int)
    return sequences, labels

def create_pickles():
    sequences, labels = create_features_and_labels(file_path, actions, no_sequences, sequence_length)
    fo = open('Sequences.pkl', 'wb')
    pickle.dump(sequences, fo)
    fo.close()

    fp = open('labels.pkl', 'wb')
    pickle.dump(labels, fp)
    fp.close()

def get_data_from_pickle(filename):
    fo = open(filename, 'rb')
    data = pickle.load(fo)
    fo.close()
    return data

def create_model(features, labels):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(features.shape[1],features.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    return model

def get_metrics(model,X_test,Y_test):
    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1).tolist()
    Y_test = np.argmax(Y_test, axis=1).tolist()
    cm = multilabel_confusion_matrix(Y_test, y_pred)
    ac = accuracy_score(Y_test, y_pred)
    return cm, ac

def check_inference(model, actions, no_sequences):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            frame, results = mediapipe_detection(frame, holistic)
            draw_landmarks(frame, results)

            landmarks = extract_landmarks(results)
            sequence.append(landmarks)
            sequence = sequence[-no_sequences:]

            if len(sequence) == no_sequences:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0, 0), (640, 40), (255, 255, 0), -1)
            cv2.putText(frame, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame',frame)
            if cv2.waitKey(10) == ord('q'):
                break

if __name__ == '__main__':
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    file_path = 'data/'
    actions = ['book', 'look', 'sorry']
    no_sequences = 30
    sequence_length = 30

    '''cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
             ret, frame = cap.read()
             frame, results = mediapipe_detection(frame, holistic)
             draw_landmarks(frame, results)
             cv2.imshow('frame', frame)
             if cv2.waitKey(10)==ord('q'):
                 break
        cap.release()'''

    #create_action_dir(file_path, actions, no_sequences)
    #create_dataset(file_path, actions, no_sequences, sequence_length)
    #create_pickles()

    X = get_data_from_pickle('Sequences.pkl')
    y = get_data_from_pickle('labels.pkl')

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    #training Part
    '''log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = create_model(X, y)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics='categorical_accuracy')
    model.summary()
    model.fit(X_train, Y_train, epochs=1000, callbacks=[tb_callback])
    model.save('model.h5')'''

    #loading model
    model = keras.models.load_model('model.h5')

    cm, ac = get_metrics(model, X_test, Y_test)
    print('Confusion_matrix', cm)
    print('Accuracy_score', ac)

    check_inference(model, actions, no_sequences)
