#!/usr/bin/env python
# coding: utf-8


import pandas as pd



nps = pd.read_csv('data.csv')


# define features and target
X = nps[['a','b','c']]
y = nps[['target']]


# In[43]:


X.info()


# data split
from sklearn.cross_validation import train_test_split
# define size of training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state = 33)


# use scikit-learn.feature_extraction to transform feature
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print (vec.feature_names_)

X_test = vec.transform(X_test.to_dict(orient='record'))



# import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
# use default set
dtc = DecisionTreeClassifier(criterion='entropy',max_depth = 3,min_impurity_decrease=0.03)
# use training data to training model
dtc.fit(X_train, y_train)
# use model to predict
# y_predict = dtc.predict(X_test)



# import classification_report。
from sklearn.metrics import classification_report
# print predict accuracy
print (dtc.score(X_test, y_test))
print (classification_report(y_predict, y_test, target_names = ['0', '1']))


# In[51]:



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
feature_name = ['a','b','c']
target_name = ['0','1']
with open("local.dot", 'w') as f:
    f = tree.export_graphviz(dtc,out_file=f,feature_names=feature_name,
                     class_names=target_name,filled=True,rounded=True,
                     special_characters=True)
###
#transform dot to pdf,make the result more visualizable
#input in terminal: dot -T pdf input.dot -o convertoutput.pdf


# In[52]:


# optimize algorithm
import pandas as pd
import numpy as np
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    return(clf.score(X_train, y_train), clf.score(X_test, y_test))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
depths = np.arange(1,10)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]

# find highest score cross validation index
tr_best_index = np.argmax(tr_scores)
te_best_index = np.argmax(te_scores)

print("bestdepth:", te_best_index+1, " bestdepth_score:", te_scores[te_best_index], '\n')


# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def minsplit_score(val):
    clf = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=val)
    clf.fit(X_train, y_train)
    return (clf.score(X_train, y_train), clf.score(X_test, y_test), )

# calculate prediction score

vals = np.linspace(0, 0.2, 100)
scores = [minsplit_score(v) for v in vals]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]

bestmin_index = np.argmax(te_scores)
bestscore = te_scores[bestmin_index]
print("bestmin:", vals[bestmin_index])
print("bestscore:", bestscore)

plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel("min_impurity_decrease")
plt.ylabel("Scores")
plt.plot(vals, te_scores, label='test_scores')
plt.plot(vals, tr_scores, label='train_scores')

plt.legend()

#Generate result
