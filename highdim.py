# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
def curse_of_dirmension(n,d,pdf,dist):
    '''
    次元の呪い
​
    Parameters
    ----------
    n: int
        生成するデータ数
​
    d: int
        データの次元数
    
    pdf: 
        データの分布
        norm: 正規分布
        unif: 一葉分布
    
    dist:
        距離計算法
        dist1: 原点からの距離(L2norm)
        dist2: i番目とi + 1番目の距離(L2norm)

    Returns
    -------
    mean_distance: 
        データ間の距離の平均
​
    std_distance:
        標準偏差
    '''

    if pdf == 'unif':
        # d次元一様分布からn個の点をサンプリングし、n行d列の行列を生成
        data = np.random.uniform(0,1,(n,d))
    elif pdf == 'norm':
         # d次元正規分布からn個の点をサンプリングし、n行d列の行列を生成
        data = np.random.randn(n, d)
    else:
        print('inadequate pdf')
        exit()

    distances = np.empty(0)

    if dist == 'dist1':
        #  原点からの距離(L2norm)を計算
        distances = np.linalg.norm(data - 0, ord=2, axis=1)
    elif dist == 'dist2':
        # (n-1)個の点について、i番目とi + 1番目の距離(L2norm)を計算
        for i in range(n-1):
            # L2ノルムを計算
            distance = np.linalg.norm((data[i] - data[i + 1]),ord=2)
            # 距離を配列に追加
            distances = np.append(distances, distance)
    else:
        print('inadequate dist')
        exit()

    mean_distance = np.mean(distances, axis=0)
    mean_distance = np.mean(distances)        
    std_distance = np.std(distances)
    return mean_distance, std_distance
    
if __name__ == "__main__":
# 次元数のリスト
    dimensions = [2, 4, 8, 16, 32, 64, 126, 256, 512,  1024]
# 発生させる点の数
    n = 10000
# データの分布　norm, unif
    pdf = 'norm'
#　距離計算法　dist1, dist2
    dist = 'dist1'

    mean_distances = []
    std_distances = []
    for d in dimensions:
        mean_distance, std_distance = curse_of_dirmension(n,d,pdf,dist)
        mean_distances.append(mean_distance)
        std_distances.append(std_distance)
        print("m,sd (d=%i) = %f, %f" %(d, mean_distance, std_distance))

    # グラフのプロット
    plt.figure()
    plt.title("Curse of Dimension (pdf=%s, n=%i, dist=%s)" %(pdf, n, dist))
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Eucledian Distance")
    # x軸、ｙ軸の対数目盛にするときは、下記を使う
    #plt.xscale('log')
    plt.yscale('log')
    plt.errorbar(dimensions, mean_distances, yerr=std_distances, color='blue', ecolor="red",capsize=3)
    plt.savefig("highdim37.png")
    plt.show()
    plt.close()

