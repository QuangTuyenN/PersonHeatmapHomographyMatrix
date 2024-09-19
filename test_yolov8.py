import cv2
import imutils
from ultralytics import YOLO

model = YOLO('yolov8n.pt').to('cuda')

video_path = './ImageFolder/video_hall.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (tùy chọn)
    frame = imutils.resize(frame, width=800)

    results = model(frame)

    # Lấy tất cả bounding box của các đối tượng là "person"
    for result in results:
        for box in result.boxes:
            # Lấy tên class và toạ độ bounding box
            cls_name = result.names[int(box.cls)]
            if cls_name == 'person':  # Chỉ lấy bounding box của người
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Toạ độ bounding box
                conf = box.conf[0]  # Độ tin cậy của dự đoán

                # Vẽ bounding box trên frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
