import csv
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Text2Json:
    def __init__(self, filename:str):
        self.filename = filename

    def read_txt(self):
        # txtファイルを読み込み、DataFrameを整形
        df = pd.read_table('./txt/{}.txt'.format(self.filename) , header=None, sep=' ')
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'n1', 'n2', 'n3', 'n4', 'n5', 'class', 'conf', 'n6']
        df = df.drop(columns=['n1', 'n2', 'n3', 'n4', 'n5', 'n6'])
        return df

    def calc_coords(self, x:int, y:int, w:int, h:int) -> int:
        # boundingboxの中央下部座標を算出する
        x_c = x + (w / 2)
        y_t = y + h
        x_c.to_numpy()
        y_t.to_numpy()
        x_c = x_c.astype(int)
        y_t = y_t.astype(int)

        return x_c, y_t

    def calc_x(self, row):
        return int(row['x'] + row['w'] / 2)
    
    def calc_y(self, row):
        return row['y'] + row['h']

    def text2json(self) -> str:
        df = self.read_txt()
        x = df['x']
        y = df['y']
        w = df['w']
        h = df['h']
        f = df['frame']
        cl = df['class']
        co = df['conf']
        df['x_c'] = df.apply(self.calc_x, axis='columns')
        df['y_t'] = df.apply(self.calc_y, axis='columns')
        print(df.head(5))

        # fig, ax = plt.subplots()
        # img = mpimg.imread('toyanobashi.png')
        # ax.imshow(img)
        # ax.scatter(x_c, y_t)
        # plt.show()
        groups = df.groupby('id')

        # JSONの生成と書き込み
        result = {}
        dict_id = {}
        for group_name, group_data in groups:
            time_and_pos = group_data[['frame', 'x_c', 'y_t', 'class', 'conf']].to_dict(orient='records')
            for item in time_and_pos:
                item['pos'] = [item.pop('x_c'), item.pop('y_t')]
                time_and_pos_dict = {'time_and_pos': time_and_pos}
                result[group_name] = time_and_pos_dict
            
        with open('./json/{}.json'.format(self.filename), 'w') as f:
            json.dump(result, f, indent=1, ensure_ascii=False)

def main(filename:str) -> str:
    t = Text2Json(filename)
    t.text2json()

if __name__=='__main__':
    filename='Jingubashi'
    main(filename)

