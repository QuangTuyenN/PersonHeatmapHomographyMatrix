This project demo how to use opencv homography to build person heatmap.

It use: YOLOV8 to detect person, homography opencv to calculate homography matrix H and covert a point from src camera view to a point in top view

Uncomment cv2.imwite code in inference.py and run file  to save src.jpg 

Run calculate_homo_matrix.py to calculate H matrix: choose 1 point in src and 1 point in dst by click left mouse after that click 's' button to save point. You should choose >10 point to more exact.
Then click 'h' to create plan view and calculate H, click 'm' to merge 2 img src and dst.

Write homo H matrix to inference file and run inference to realtime draw heatmap.

