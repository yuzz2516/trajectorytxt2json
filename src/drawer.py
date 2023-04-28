import cv2
import numpy as np
import matplotlib.pyplot as plt

class FindPolygon():
    def __init__(self, filename):
        self.filename = filename

    def __call__(self):
        origin_img = cv2.imread(self.filename + '.png')
        h, w, c = origin_img.shape
        depthplot_img = cv2.imread('depthplot_' + self.filename + '.png')
        h2, w2, c2= depthplot_img.shape
        resized_img = cv2.resize(depthplot_img, None, fy=h/h2, fx=w/w2)
        print(resized_img.shape)

        # 色の範囲を抽出する
        mask = cv2.inRange(resized_img, (0, 255, 0), (0, 255, 0))
        # 輪郭を検出する
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭をポリゴンに変換する
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=False)
        approx = approx.astype(np.float64)
        approx_reshaped = approx.reshape((approx.shape[0], 2))

        # ポリゴンを描画する
        # output = np.zeros_like(origin_img)
        # cv2.polylines(origin_img, [approx], True, (0, 0, 255), 2)
        # cv2.imshow('image', origin_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return approx_reshaped

