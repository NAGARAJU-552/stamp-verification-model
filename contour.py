
import numpy as np
import cv2

im = cv2.imread("F:/nagaraju/inputs/RIKMy.png")
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)

### Step #1
# Detect contours using both methods on the same image
contours1, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours2, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

### Step #2 - Reshape to 2D matrices
contours1 = contours1[0].reshape(-1,2)
contours2 = contours2[0].reshape(-1,2)

### Step #3 - Draw the points as individual circles in the image
img1 = im.copy()
img2 = im.copy()

for (x, y) in contours1:
    cv2.circle(img1, (x, y), 1, (255, 0, 0), 3)

for (x, y) in contours2:
    cv2.circle(img2, (x, y), 1, (255, 0, 0), 3)

### Step #4 - Stack the two images side by side and show it
out = np.hstack([img1])
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()