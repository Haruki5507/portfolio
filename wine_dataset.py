from sklearn.datasets import load_wine
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns

wine = load_wine()

#決定木のモデルを描画するためのImport
from sklearn.tree import export_graphviz
from IPython.display import Image

#線形モデル(決定木)として測定器を作成する
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

X_all = pd.DataFrame(wine.data, columns = wine.feature_names)
y_all = pd.DataFrame(wine.target)
y_all = y_all.rename(columns={0:'class'})
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=0)

clf.fit(X_train, y_train)

#評価の実行
df = pd.DataFrame(clf.predict_proba(X_test))
df = df.rename(columns={0: 'class_0',1: 'class_1',2: 'class_2'})

#評価の実行（判定）
df = pd.DataFrame(clf.predict(X_test))
df = df.rename(columns={0: '判定'})

#混同行列
from sklearn.metrics import confusion_matrix
#print(X_train, X_test)

#標準化なしで学習
#PCAなし
lr1 = LogisticRegression(random_state=1)
lr1.fit(X_train, y_train)
print("標準化なしPCAなし")
print(X_train.shape, X_test.shape)
acc_train = accuracy_score(y_train, lr1.predict(X_train))
acc_test = accuracy_score(y_test, lr1.predict(X_test))
print(acc_test)
#混同行列
df5 = pd.DataFrame(confusion_matrix(y_test,lr1.predict(X_test).reshape(-1,1)))
df5 = df5.rename(columns={0: '予(class_0)',1: '予(class_1)',2: '予(class_2)'}, index={0: '実(class_0)',1: '実(class_1)',2: '実(class_2)'})
print(df5)

#PCAあり
pca1 = PCA(n_components=2)
X_train_pca1 = pca1.fit_transform(X_train)
X_test_pca1 = pca1.transform(X_test)
lr_pca1 = LogisticRegression(random_state=1)
lr_pca1.fit(X_train_pca1, y_train)
#lr_pca1.fit(X_test_pca1, y_test)
print("標準化なしPCAあり")
print(X_train_pca1.shape, X_test_pca1.shape)
acc_train_pca1 = accuracy_score(y_train, lr_pca1.predict(X_train_pca1))
acc_test_pca1 = accuracy_score(y_test, lr_pca1.predict(X_test_pca1))
print(acc_test_pca1)
#混同行列
df2 = pd.DataFrame(confusion_matrix(y_test,lr_pca1.predict(X_test_pca1).reshape(-1,1)))
df2 = df2.rename(columns={0: '予(class_0)',1: '予(class_1)',2: '予(class_2)'}, index={0: '実(class_0)',1: '実(class_1)',2: '実(class_2)'})
print(df2)

#標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#PCAを施さずに学習
lr2 = LogisticRegression(random_state=1)
lr2.fit(X_train_std, y_train)
acc_train_std = accuracy_score(y_train, lr2.predict(X_train_std))
acc_test_std = accuracy_score(y_test, lr2.predict(X_test_std))
print("標準化ありPCAを施さずに学習")
print(X_train_std.shape, X_test_std.shape)
print(acc_train_std, acc_test_std)
#混同行列
df3 = pd.DataFrame(confusion_matrix(y_test,lr2.predict(X_test_std).reshape(-1,1)))
df3 = df3.rename(columns={0: '予(class_0)',1: '予(class_1)',2: '予(class_2)'}, index={0: '実(class_0)',1: '実(class_1)',2: '実(class_2)'})
print(df3)

#PCAを施して学習
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr_pca2 = LogisticRegression(random_state=1)
lr_pca2.fit(X_train_pca, y_train)
print("標準化ありPCAを施して学習")
print(X_train_pca.shape, X_test_pca.shape)
acc_train_pca = accuracy_score(y_train, lr_pca2.predict(X_train_pca))
acc_test_pca = accuracy_score(y_test, lr_pca2.predict(X_test_pca))
print(acc_train_pca, acc_test_pca)
#混同行列
df4 = pd.DataFrame(confusion_matrix(y_test,lr_pca2.predict(X_test_pca).reshape(-1,1)))
df4 = df4.rename(columns={0: '予(class_0)',1: '予(class_1)',2: '予(class_2)'}, index={0: '実(class_0)',1: '実(class_1)',2: '実(class_2)'})
print(df4)

#正規化ありで学習
mms = preprocessing.MinMaxScaler()
X_train_ms = mms.fit_transform(X_train)
X_test_ms = mms.transform(X_test)
#PCAを施さずに学習
lr2 = LogisticRegression(random_state=1)
lr2.fit(X_train_ms, y_train)
acc_train_ms = accuracy_score(y_train, lr2.predict(X_train_ms))
acc_test_ms = accuracy_score(y_test, lr2.predict(X_test_ms))
print("正規化ありPCAを施さずに学習")
print(X_train_ms.shape, X_test_ms.shape)
print(acc_train, acc_test)
#混同行列
df5 = pd.DataFrame(confusion_matrix(y_test,lr2.predict(X_test_ms).reshape(-1,1)))
df5 = df5.rename(columns={0: '予(class_0)',1: '予(class_1)',2: '予(class_2)'}, index={0: '実(class_0)',1: '実(class_1)',2: '実(class_2)'})
sns.heatmap(df5, square=True, annot=True, cbar=True, fmt='d', cmap='RdPu')
plt.title("normalization")
plt.xlabel("predicted class")
plt.ylabel("true class")
print(df5)

#PCAを施して学習
pca = PCA(n_components=2)
X_train_ms_pca = pca.fit_transform(X_train_ms)
X_test_ms_pca = pca.transform(X_test_ms)
lr_pca2 = LogisticRegression(random_state=1)
lr_pca2.fit(X_train_ms_pca, y_train)
print("正規化ありPCAを施して学習")
print(X_train_ms_pca.shape, X_test_ms_pca.shape)
acc_train_ms_pca = accuracy_score(y_train, lr_pca2.predict(X_train_ms_pca))
acc_test_ms_pca = accuracy_score(y_test, lr_pca2.predict(X_test_ms_pca))
print(acc_train_ms_pca, acc_test_ms_pca)
#混同行列
df6 = pd.DataFrame(confusion_matrix(y_test,lr_pca2.predict(X_test_ms_pca).reshape(-1,1)))
df6 = df6.rename(columns={0: '予(class_0)',1: '予(class_1)',2: '予(class_2)'}, index={0: '実(class_0)',1: '実(class_1)',2: '実(class_2)'})
print(df6)