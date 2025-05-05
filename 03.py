# 使用opencv中的某一种图像去噪方法，实现对图像噪声处理
import cv2
img = cv2.imread("./input1.png")
median = cv2.medianBlur(img, ksize=5)
cv2.imshow("median", median)
cv2.waitKey(0)