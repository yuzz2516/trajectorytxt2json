import csv
import pandas as pd
import json

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
        x_c = x + w / 2
        y_t = y + h
        x_c.to_numpy()
        y_t.to_numpy()
        x_c = x_c.astype(int)
        y_t = y_t.astype(int)

        return x_c, y_t

    def text2json(self) -> str:
        df = self.read_txt(self.filename)
        x = df['x']
        y = df['y']
        w = df['w']
        h = df['h']
        x_c, y_t = self.calc_coords(x, y, w, h)
        df = df.replace({'x': x_c, 'y': y_t})
        groups = df.groupby('id')

        # JSONの生成と書き込み
        result = {}
        for group_name, group_data in groups:
            time_and_pos = group_data[['frame', 'x', 'y', 'class', 'conf']].to_dict('records')
            for item in time_and_pos:
                item['pos'] = [item.pop('x'), item.pop('y')]
            
            result[group_name] = {'time_and_pos': time_and_pos}

        with open('./json/{}.json'.format(self.filename), 'w') as f:
            json.dump(result, f, indent=1, ensure_ascii=False)

def main(filename:str) -> str:
    t = Text2Json(filename)
    t.text2json()

if __name__=='__main__':
    filename='Jingubashi'
    main(filename)

