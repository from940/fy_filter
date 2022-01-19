"""
뻐큐 탐지 후 모자이크
"""
import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1
# 제스처 목록
gesture = {
    0 : 'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손가락 탐지 모듈 초기화
hands = mp_hands.Hands(
    max_num_hands=max_num_hands, # 최대 몇 개 손 인식할 지
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train_fy.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

# knn 모델. 속도가 빠르다
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# 웹캠
cap = cv2.VideoCapture(0)

# 웹캠 한 프레임씩 읽어온다
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    # 거울처럼 뒤집음
    img = cv2.flip(img, 1)
    # opencv BGR 컬러 시스템
    # mediapipe는 RGB 컬러 시스템을 사용하기 때문에 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 전처리 및 모델 추론 함께 실행
    result = hands.process(img)

    # 이미지 컬러 다시 변환
    im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 손이 인식되지 않으면 False, 손이 인식되면 True
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:

            # joint 에 손가락 각 관절 위치 정보 저장
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :] # Chile joint

            # 관절 두 점의 위치 정보를 빼면 벡터를 구한다
            v = v2 - v1 # [20, 3]

            # Normalize v
            # 각 벡터의 길이로 normalize 유클리디안 거리를 구하는 공식
            # 길이로 나눠주면 eigen vector 고유벡터 나온다? 크기 1짜리 벡터?
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # arccos 코사인
            # np.einsum einsum 연산을 통하면 행렬 내적, 외적, 내적, 행렬곱 등 동일한 상태로 할 수 있다.
            # np.arccos 역삼각함수 아크코사인값 inverse trigonometric cosine
            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15, ]
            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            if idx==11:
                x1, y1 = tuple((joint.min(axis=0)[:2]*[img.shape[1], img.shape[0]]*0.95).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                fy_img = img[y1:y2, x1:x2].copy()
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                fy_img = cv2.resize(fy_img, dsize=(x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

                img[y1:y2, x1:x2] = fy_img

            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Filter', img)
    if cv2.waitKey(1)==ord('q'):
        break