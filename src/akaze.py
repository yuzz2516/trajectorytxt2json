import cv2

# 2つの画像を読み込む
img1 = cv2.imread("toyanobashi.png")
img2 = cv2.imread("toyanobashi3.png")

# AKAZE特徴量抽出器を初期化します
akaze = cv2.AKAZE_create()

# 特徴点と特徴ディスクリプタを抽出します
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

# Brute-Force Matcherを初期化します
bf = cv2.BFMatcher()

# 特徴点をマッチングします
matches = bf.match(des1, des2)

# 距離に基づいてマッチングをソートします
matches = sorted(matches, key = lambda x:x.distance)

# 上位30個のマッチングを描画します
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 結果を表示します
cv2.imshow("Matches", img3)
cv2.imwrite('matching_toyanobashi.png', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
