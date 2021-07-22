# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
import numpy as np#科学技術計算ライブラリ
import matplotlib.pyplot as plt#グラフ描画ライブラリ
import pandas  as pd#DataFrameを使うためのライブラリ
from sklearn import datasets
from sklearn.decomposition import PCA#scikit-learnのPCAクラス
from sklearn.preprocessing import StandardScaler#標準化のクラス

# 変数にアヤメデータを入れる
iris = datasets.load_iris()
#アヤメデータをラベルと入力データに分離する
X = iris.data  
Y = iris.target
print(X.shape)#データ150×4
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

#PandsのDataFrameで特徴量を表示
iris_dataframe = pd.DataFrame(iris.data)

iris_target_dataframe = pd.DataFrame(iris.target)

md = pd.merge(iris_dataframe, iris_target_dataframe, left_index=True, right_index=True)
md.columns = ["sepal length", "sepal width", "petal length", "petal width", "target"]
print(md)

# sepal length vs sepal widthのグラフを作る
markers = ['o', '^', 'v']
for i in range(3):
    d = iris.data[iris.target == i, :]
    plt.plot(d[:,0], d[:,1], 'o', fillstyle='none', marker=markers[i])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend(iris.target_names)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#主成分分析する
pca = PCA(n_components = 2, whiten = False)
pca.fit(X)
X_pca = pca.fit_transform(X)
print(X_pca.shape)
# 分析結果を元にデータセットを主成分に変換する
transformed = pca.fit_transform(X)
#主成分をプロットする
for label in np.unique(Y):
    if label == 0:
        c = "red"
    elif label == 1:
        c = "blue"
    elif label == 2:
        c = "green"
    else:
        pass
    plt.scatter(transformed[Y == label, 0],
                transformed[Y == label, 1],
               c=c)
plt.title('principal component')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.ylim(-3, 3)
plt.legend(iris.target_names)
plt.show()
print(pca.get_covariance()) #分散共分散行列
print(pca.components_.T) #固有ベクトル
print(pca.explained_variance_) #固有値
#寄与率
print ("各次元の寄与率: {0}".format(pca.explained_variance_ratio_))
#累積寄与率
print("累積寄与率: {0}".format(np.cumsum(pca.explained_variance_ratio_)))

plt.clf()
#pcaを標準化
X_std = StandardScaler().fit_transform(X)
pca_std = pca.fit_transform(X_std)
for label in np.unique(Y):
    if label == 0:
        c = "pink"
    elif label == 1:
        c = "c"
    elif label == 2:
        c = "y"
    else:
        pass
    plt.scatter(pca_std[Y == label, 0],
                pca_std[Y == label, 1],
              c=c)
plt.title('principal component(std)')
plt.xlabel('pc1(std)')
plt.ylabel('pc2(std)')
plt.legend(iris.target_names)
plt.show()
print(pca.get_covariance()) #分散共分散行列
print(pca.components_.T) #固有ベクトル
print(pca.explained_variance_) #固有値

#寄与率
print ("各次元の寄与率: {0}".format(pca.explained_variance_ratio_))
#累積寄与率
print("累積寄与率: {0}".format(np.cumsum(pca.explained_variance_ratio_)))