import matplotlib.pyplot as plt
import cv2

class PolygonDrawer(object):
    def polygon(s_list:list, e_list:list, filename:str) -> list:
        polygon = get_clicked_points(filename + '.png')
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

if __name__ == '__main__':
    fig, ax = plt.subplots()
    img = plt.imread('Jingubashi.png')
    ax.imshow(img)
    drawer = PolygonDrawer(ax)
    plt.show()
