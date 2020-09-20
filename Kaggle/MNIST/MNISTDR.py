#MNIST Problem by dimensional reduction

import pandas as pd


# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.iloc[:,1:]
y = train.iloc[:,0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,stratify= y,random_state=1)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
test_std = sc.transform(test)

#With PCA
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# initializing the PCA transformer and
# logistic regression estimator:
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr',random_state=1,solver='lbfgs')
# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
test_pca = pca.transform(test_std)


#PIPELINE

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')

pipe_kn = make_pipeline(StandardScaler(),PCA(n_components=2), clf3)

pipe_kn.fit(X_train_pca, y_train)
y_pred_kn = pipe_kn.predict(test_pca)

pipe_dt = make_pipeline(StandardScaler(),PCA(n_components=2), clf2)

pipe_dt.fit(X_train_pca, y_train)
y_pred_dt = pipe_dt.predict(test_pca)

print(pipe_dt.score(X_train_pca,y_train))
print(pipe_kn.score(X_train_pca,y_train))

print(y_pred_kn)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
rfc.fit(X_train_pca, y_train)
rfc.predict(test_pca)


#Predict result


result = rfc.predict(test_pca)
submission = pd.DataFrame({ 'ImageId' : list(range(1,len(result)+1)),
             'Label': result})

submission.to_csv('Submission.csv',index=False,header = True)
