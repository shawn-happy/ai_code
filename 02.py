# 使用opencv中直方图均衡化的方法对图像实现光照度均匀处理
import cv2
src = cv2.imread("./input2.jpg",0)
dst = cv2.equalizeHist(src)
cv2.imshow("dst", dst)
cv2.waitKey(0)