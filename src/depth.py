import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def main(filename:str) -> str:
    # depth imageを開く
    img = cv2.imread('depth_' + filename + '.png') # depth image
    img2 = cv2.imread(filename + '.png')
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    fig, ax = plt.subplots()

    # jsonファイルの読み込み
    with open('./json/' + filename + '.json', 'r', encoding='shift_jis') as f:
        json_data = json.load(f)

    posx_lst = []
    posy_lst = []
    gray_value_lst = []
    for v, _ in json_data.items():
        for time_and_pos in json_data['{}'.format(v)]['time_and_pos']:
            pos = time_and_pos['pos']
            posx_lst.append(pos[0])
            posy_lst.append(pos[1])
            gray_value = get_grayscale(pos, img)
            gray_value_lst.append(gray_value)
            time_and_pos['depth'] = gray_value

    plt.scatter(posx_lst, posy_lst, s=gray_value_lst, c='#00ff00')
    ax.imshow(img2_bgr)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    plt.savefig('depthplot_'+ filename + '.png', bbox_inches='tight', pad_inches=0.0)

def get_grayscale(pos:list, img:str) -> str:
        px = pos[0]
        py = pos[1]
        gray_value = img[py-1, px-1, 0]  # 0はチャンネル指定（グレースケール画像の場合は0）
        #print(f"The grayscale value of pixel ({px}, {py}) is {gray_value}.")
        return gray_value

if __name__=='__main__':
    filename = 'toyanobashi'
    main(filename)