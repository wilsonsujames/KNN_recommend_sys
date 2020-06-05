import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

anime = pd.read_csv("anime.csv")
# 看資料結構
# print(anime.head())
# print(anime.isnull().sum())

# print(anime[anime['episodes']=='Unknown'].head(3))

# 把未知轉換成1
anime.loc[(anime["genre"]=="Hentai") & (anime["episodes"]=="Unknown"),"episodes"] = "1"
anime.loc[(anime["type"]=="OVA") & (anime["episodes"]=="Unknown"),"episodes"] = "1"
anime.loc[(anime["type"] == "Movie") & (anime["episodes"] == "Unknown")] = "1"


# 將Unknown即沒有填的資料換成平均值
anime["episodes"] = anime["episodes"].map(lambda x:np.nan if x=="Unknown" else x)
anime["episodes"].fillna(anime["episodes"].median(),inplace = True)

# print(pd.get_dummies(anime[["type"]]).head())


# rating特徵轉浮點數，空值補平均值
anime["rating"] = anime["rating"].astype(float)
anime["rating"].fillna(anime["rating"].median(),inplace = True)
# members特徵轉成浮點數
anime["members"] = anime["members"].astype(float)

# 將genre特徵轉換成onehot encoder
anime_features = pd.concat([anime["genre"].str.get_dummies(sep=","),
                            pd.get_dummies(anime[["type"]]),
                            anime[["rating"]],anime[["members"]],anime["episodes"]],axis=1)
anime["name"] = anime["name"].map(lambda name:re.sub('[^A-Za-z0-9]+', " ", name))
print(anime_features.head())

# 使用MinMaxScaler 幫助標準化 加速運算速度
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
anime_features = min_max_scaler.fit_transform(anime_features)
np.round(anime_features,decimals=2)

# print(anime_features)

# 引入最近鄰
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(anime_features)
distances, indices = nbrs.kneighbors(anime_features)
# indices 為一個[],第一個element 是動畫自己的ID，後面的element是最相似(推薦的)的動畫ID


# 設立function幫助查找
def get_index_from_name(name):
    return anime[anime["name"]==name].index.tolist()[0]


# 所有anime 的名稱
all_anime_names = list(anime.name.values) 
# print(all_anime_names)


def get_id_from_partial_name(partial):
    for name in all_anime_names:
        if partial in name:
            print(name,all_anime_names.index(name))


def print_similar_animes(query=None,id=None):
    if id:
        for id in indices[id][1:]:
            print(anime.loc[id]["name"])
    if query:
        found_id = get_index_from_name(query)
        for id in indices[found_id][1:]:
            print(anime.loc[id]["name"])            


print_similar_animes(query="Naruto")
# print_similar_animes("Mushishi")

# print_similar_animes("Fairy Tail")
# get_id_from_partial_name("Naruto")


# print_similar_animes(id=719)