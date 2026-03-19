# Lane Detection Using Sliding Windows In Python Using OpenCV
# Author: Vương Hoài Nam
# Todo: 2025-10-27
# Flow
# Step-1: Reading Video
# Step-2: Making region of interest
# Step-3: Applying Perspective Transformation.
# Step-4: Lane Detection - Image Thresholding...
# Step-5: Lane Detection - Histogram > Sliding Windows


import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# 1. Reading video
cap = cv2.VideoCapture("LaneVideo.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Step-2: Making region of interest
    frame_reg = cv2.resize(frame,(640,480))
    ## Choosing point for perspective transformation
    tl = (222,387)
    bl = (70,472)
    tr = (400,380)
    br = (538,472)
    cv2.circle(frame_reg, tl, 5, (0,0,225), -1)
    cv2.circle(frame_reg, bl, 5, (0,0,225), -1)
    cv2.circle(frame_reg, tr, 5, (0,0,225), -1)
    cv2.circle(frame_reg, br, 5, (0,0,225), -1)

    # Step-3: Applying Perspective Transformation.
    ## Aplying rerspective transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

    # Matrix to warp the image for birdseye windows
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_frame = cv2.warpPerspective(frame_reg, matrix, (640,480))
    
    # Step-4: Lane Detection - Image Thresholding...
    ## Object Detection
    ### Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # Step-5: Lane Detection - Histogram > Sliding Windows
    ## Histogram
    ### Chung ta chia cho 2 de lay nua duoi cua histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    ## Sliding Windows
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y > 0:
        ## Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] !=0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx
       
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] !=0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(right_base-50 + cx)
                right_base = right_base-50 + cx

        cv2.rectangle(msk,(left_base-50,y),(left_base+50, y-40),(255,255,255), 2)
        cv2.rectangle(msk,(right_base-50,y),(right_base+50, y-40),(255,255,255), 2)
        y -= 40

    cv2.imshow("Original", frame_reg)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()