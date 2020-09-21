#MNIST Problem by dimensional reduction

import pandas as pd
import numpy as np

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.decomposition import PCA
# initializing the PCA transformer and
# logistic regression estimator:
pca = PCA(n_components=250)
lda = LDA(n_components=8)
# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_train_lda = lda.fit_transform(X_train_std, y_train)

test_pca = pca.transform(test_std)
test_lda=lda.transform(test_std)
print(np.sum(pca.explained_variance_ratio_))
print(np.sum(lda.explained_variance_ratio_))

#PIPELINE


from sklearn.ensemble import RandomForestClassifier

#Predict result PCA
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
rfc.fit(X_train_pca, y_train)
rfc.predict(test_pca)

result = rfc.predict(test_pca)
submission = pd.DataFrame({ 'ImageId' : list(range(1,len(result)+1)),
             'Label': result})

submission.to_csv('Submission.csv',index=False,header = True)


#Predict result LDA

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)
rfc.fit(X_train_lda, y_train)
rfc.predict(test_lda)

result = rfc.predict(test_lda)
submission = pd.DataFrame({ 'ImageId' : list(range(1,len(result)+1)),
             'Label': result})

submission.to_csv('Submission.csv',index=False,header = True)
