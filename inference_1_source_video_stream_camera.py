### importing required libraries
import torch
import cv2
import time
import re
import numpy as np
from imutils.video import VideoStream
from ultralytics import YOLO


### -------------------------------------- function to run detection ---------------------------------------------------

def detectx(frame, model):
    frame = [frame]
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates


# --------------------------------------get plan view fuction-----------------------------------------------------------
# def get_plan_view(H,src, dst, results, frame, classes):
#     plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
#     labels, cord = results
#     n = len(labels)
#     x_shape, y_shape = frame.shape[1], frame.shape[0]
#     ### looping through the detections
#     dsttt = dst.copy()
#     for i in range(n):
#         row = cord[i]
#         if row[4] >= 0.55:  ### threshold value for detection. We are discarding everything below this value
#             x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
#                 row[3] * y_shape)  ## BBOx coordniates
#             text_d = classes[int(labels[i])]
#             if text_d == 'person':
#                 polygon = np.array([[0, 1216], [0, 1030], [700, 630], [1434, 660], [2150, 1130]], np.int32)
#                 polygon = polygon.reshape((-1, 1, 2))
#                 dist = cv2.pointPolygonTest(polygon, (int((x1 + x2)/2), y2), False)
#                 if dist == 1.0:
#                     pts = np.float32([int((x1 + x2)/2), y2]).reshape(-1, 1, 2)
#                     dstt = cv2.perspectiveTransform(pts, H)
#                     xx = int(dstt[0][0][0])
#                     yy = int(dstt[0][0][1])
#                     cv2.circle(dsttt, (xx, yy), 5, (0, 0, 255), -1)
#                     cv2.circle(plan_view, (xx, yy), 5, (0, 0, 255), -1)
#     return plan_view, dsttt


def get_plan_view(H,src, dst, results, frame):
    plan_view = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    # labels, cord = results
    # n = len(labels)
    # x_shape, y_shape = frame.shape[1], frame.shape[0]
    ### looping through the detections
    dsttt = dst.copy()
    for result in results:
        for box in result.boxes:
            # Lấy tên class và toạ độ bounding box
            cls_name = result.names[int(box.cls)]
            if cls_name == 'person':  # Chỉ lấy bounding box của người
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Toạ độ bounding box
                pts = np.float32([int((x1 + x2) / 2), y2]).reshape(-1, 1, 2)
                dstt = cv2.perspectiveTransform(pts, H)
                xx = int(dstt[0][0][0])
                yy = int(dstt[0][0][1])
                cv2.circle(dsttt, (xx, yy), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int((x1 + x2) / 2), y2), 5, (0, 0, 255), -1)
    # for i in range(n):
    #     row = cord[i]
    #     if row[4] >= 0.55:  ### threshold value for detection. We are discarding everything below this value
    #         x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
    #             row[3] * y_shape)  ## BBOx coordniates
    #         text_d = classes[int(labels[i])]
    #         if text_d == 'person':
    #             polygon = np.array([[0, 1216], [0, 1030], [700, 630], [1434, 660], [2150, 1130]], np.int32)
    #             polygon = polygon.reshape((-1, 1, 2))
    #             dist = cv2.pointPolygonTest(polygon, (int((x1 + x2)/2), y2), False)
    #             if dist == 1.0:
    #                 pts = np.float32([int((x1 + x2)/2), y2]).reshape(-1, 1, 2)
    #                 dstt = cv2.perspectiveTransform(pts, H)
    #                 xx = int(dstt[0][0][0])
    #                 yy = int(dstt[0][0][1])
    #                 cv2.circle(dsttt, (xx, yy), 5, (0, 0, 255), -1)
    #                 cv2.circle(plan_view, (xx, yy), 5, (0, 0, 255), -1)
    return frame, dsttt


# -------------------------------------- MAIN FUNCTION------------------------------------------------------------------

model = YOLO('yolov8n.pt').to('cuda')

video_path = './ImageFolder/video_hall.mp4'

cap = VideoStream(video_path).start()
# cap = cv2.VideoCapture(video_path)

# khai bao bien anh destination
dst = cv2.imread('./ImageFolder/dst.jpg', -1)


# khai bao ma tran homography H
H = [[1.27947143e+00, 1.43046366e+00, 5.10784175e+01], [3.23311622e-02, 2.41156586e+00, 6.29061599e+01], [3.65350962e-06, 2.31483382e-03, 1.00000000e+00]]
H = np.array(H)


while True:
    frame = cap.read()
    # try:
    frame = cap.read()
    frame = frame[1]
    # frame = cv2.resize(src=frame, dsize=None, fx=0.8, fy=0.8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)

    camera_view, des = get_plan_view(H, frame, dst, results, frame)
    cv2.imshow("camera view", camera_view)
    cv2.imshow("dst ", des)
    # except Exception as bug:
    #     print("bug: ", bug)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()





