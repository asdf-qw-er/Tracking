import cv2
import numpy as np

# YOLO 모델 로드
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# 객체 검출
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

# 객체 검출 결과 처리
def get_bounding_boxes(outs, height, width, conf_threshold=0.5, nms_threshold=0.4):
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [(boxes[i], confidences[i], class_ids[i]) for i in indices.flatten()]

# 트래커 초기화 및 객체 추적
def track_objects_from_webcam():
    net, classes, output_layers = load_yolo_model()
    cap = cv2.VideoCapture(0)  # 웹캠을 사용하여 비디오 캡처

    trackers = cv2.legacy.MultiTracker_create()
    frame_id = 0
    detection_interval = 30  # 객체 검출 주기 (프레임 단위)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        frame_id += 1

        if frame_id % detection_interval == 0:
            # 객체 검출
            outs = detect_objects(frame, net, output_layers)
            boxes = get_bounding_boxes(outs, height, width)

            trackers = cv2.legacy.MultiTracker_create()  # 트래커 초기화
            for box, confidence, class_id in boxes:
                tracker = cv2.legacy.TrackerCSRT_create()
                trackers.add(tracker, frame, tuple(box))

        # 객체 추적
        success, boxes = trackers.update(frame)
        for box in boxes:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 웹캠을 사용하여 객체 추적 시작
track_objects_from_webcam()