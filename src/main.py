import json
import numpy as np
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Polygon

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score

from drawer import FindPolygon

'''''''''''''''''''''''''''''''''''''''''''''
基本的にjson形式から新たなデータを起こすように設計する。
'''''''''''''''''''''''''''''''''''''''''''''

def main(filename:str) -> str:
    # jsonファイルの読み込み
    with open('./json/' + filename + '.json', 'r', encoding='shift_jis') as f:
        json_data = json.load(f)

    # 座標群を始点と終点に振り分ける
    s_list, e_list = split_coords(json_data)

    # ポリゴン領域内にある座標の内外判定をし、内側にあるものを返す関数
    polygon_points, start_inside_points, end_inside_points = polygon(s_list, e_list, filename)
    print(end_inside_points)

    # サポートベクターマシンにかける
    SVM(polygon_points, start_inside_points, end_inside_points, filename)
    #draw_image(start_inside_points, end_inside_points, polygon_points, filename)


def split_coords(json_data:str) -> list:
    # 各ID内のframeの最大値と最小値を取得する
    start_list = []
    end_list = []
    for vehicle_id, _ in json_data.items():
        pos_max_dict, pos_min_dict = get_pos_range(json_data, vehicle_id)
        # print(f"ID: {vehicle_id}")
        # print(f"Max pos: {pos_max_dict}")
        # print(f"Min pos: {pos_min_dict}")
        start_list.append(pos_min_dict)
        end_list.append(pos_max_dict)
    s_list = []
    e_list = []
    for s, e in zip(start_list, end_list):
        s_list.append(list(s.values())[0])
        e_list.append(list(e.values())[0])

    return s_list, e_list

# IDごとにフレーム値の最大値と最小値、および対応するposを見つける関数を定義
def get_pos_range(json_data:str, vehicle_id:int) -> dict:
    pos_dict = {}
    pos_max_dict = {}
    pos_min_dict = {}
    for d in json_data['{}'.format(vehicle_id)]['time_and_pos']:
        if json_data[vehicle_id].values() is None:
            continue
        else:
            pos_dict[d['frame']] = d['pos']
    pos_max = max(pos_dict)
    pos_min = min(pos_dict)
    pos_max_dict[pos_max] = pos_dict[pos_max]
    pos_min_dict[pos_min] = pos_dict[pos_min]
    return pos_max_dict, pos_min_dict

# 領域内に存在する始点、終点群の内外判定をし、内側にあるものをリストで返す
def polygon(s_list:list, e_list:list, filename:str) -> list:
    poly = FindPolygon(filename)
    polygon = poly()

    #polygon = get_clicked_points(filename + '.png')
    #polygon = [[512, 410], [520, 330], [588, 267], [621, 410], [1546, 1074], [1064, 1074]]

    poly_path = mpath.Path(polygon)

    start_inside_points = []
    for point in s_list:
        if poly_path.contains_point(point):
            start_inside_points.append(point)

    end_inside_points = []
    for point in e_list:
        if poly_path.contains_point(point):
            end_inside_points.append(point)

    # print('Inside points (Start)', start_inside_points)
    # print('Inside points (End)', end_inside_points)

    return polygon, start_inside_points, end_inside_points

def get_clicked_points(filename):
    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))

    image = cv2.imread(filename)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    return clicked_points

def draw_image(start_list:list, end_list:list, points:list, filename:str) -> str:
    # 画像の読み込み
    img = Image.open("{}.png".format(filename))
    fig, ax = plt.subplots()

    # start座標のプロット
    x_start = [coord[0] for coord in start_list]
    y_start = [coord[1] for coord in start_list]
    plt.scatter(x_start, y_start)

    # end座標のプロット
    x_end = [coord[0] for coord in end_list]
    y_end = [coord[1] for coord in end_list]
    plt.scatter(x_end, y_end)

    # Polygonのプロット
    polygon = Polygon(points, facecolor='0.9', edgecolor='r', fill=False)
    ax.add_patch(polygon)

    ax.imshow(img)
    plt.show()

def SVM(polygon_points:list ,start_inside_point:list, end_inside_point:list, filename:str):
    # 始点群と終点群の座標をSklearn用に対応させる
    # ラベルとして、0が始点群、1が終点群とする
    img = Image.open('{}.png'.format(filename))
    fig, ax = plt.subplots()
    ax.imshow(img, alpha=0.5)

    X = np.array(start_inside_point + end_inside_point)
    y = np.array([0] * len(start_inside_point) + [1] * len(end_inside_point))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 学習データと連結
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # SVCによる分類
    model = SVC(kernel='linear', random_state=None)
    model.fit(X_combined_std, y_combined)

    # 境界線の表示
    coef_scaled = model.coef_
    intercept_scaled = model.intercept_
    std = sc.scale_
    mean = sc.mean_
    coef = coef_scaled / std
    intercept = intercept_scaled - np.dot(coef, mean)

    # 分割境界線を可視化する
    w = coef[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 1920, 10)
    yy = a * xx - (intercept[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(coef ** 2))
    yy_down = yy - np.abs(a) * margin
    yy_up = yy + np.abs(a) * margin
    #plt.plot(xx, yy_down, 'k--')
    #plt.plot(xx, yy_up, 'k--')

    # トレーニングデータに対する精度
    pred_train = model.predict(X_train_std)
    accuracy_train = accuracy_score(y_train, pred_train)
    print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

    # テストデータに対する精度
    pred_test = model.predict(X_test_std)
    accuracy_test = accuracy_score(y_test, pred_test)
    print('テストデータに対する正解率： %.2f' % accuracy_test)
    
    # データと境界線のプロット
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma')
    plt.plot(xx, yy, 'k-')
    ax.set_xlim([0, 1920])
    ax.set_ylim([0, 1080])

    # ポリゴンの描画
    polygon = Polygon(polygon_points, facecolor='0.9', edgecolor='r', fill=False)
    ax.add_patch(polygon)

    # 始点群と終点群の座標をプロットする
    ax.invert_yaxis()
    plt.axis('tight')
    plt.savefig('./results/SVC/{}.png'.format(filename))

if __name__=='__main__':
    filename = 'toyanobashi'
    main(filename)