
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import sklearn


# In[2]:

train_v = pd.read_csv("C:\\Users\\nitesh.garg\\Desktop\\Python\\cancer problem\\training_variants.csv")
test_v = pd.read_csv("C:\\Users\\nitesh.garg\\Desktop\\Python\\cancer problem\\test_variants.csv")
train_v.head()


# In[3]:

train_t = pd.read_csv("C:\\Users\\nitesh.garg\\Desktop\\Python\\cancer problem\\training_text", sep = "\|\|",header= None, skiprows =1,names=["ID","Text"], engine = 'python')
test_t = pd.read_csv("C:\\Users\\nitesh.garg\\Desktop\\Python\\cancer problem\\test_text", sep = "\|\|",header= None, skiprows =1,names=["ID","Text"], engine = 'python')
train_t.head()


# In[4]:

train = pd.merge(train_v, train_t, how = 'left', on = 'ID')


# train.head()
train.describe(include= ['object', np.number])
# x = train.groupby(['Text']).count()
# x.sort_values(['ID'], ascending = 0)

# plt.figure(figsize=(12,8))
# sns.countplot(x = 'Class', data = train,palette="Blues_d")
# plt.show()

# x = train.groupby(['Gene'])['Gene'].count().sort_values(ascending = False)[:15]

# In[5]:

x_train = train_t['Text']
x_test = test_t['Text']


# In[6]:

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
vect = TfidfVectorizer(min_df=5, max_features=16000, token_pattern=r'\w+', strip_accents='unicode',lowercase =True,
analyzer='word', use_idf=True, 
smooth_idf=True, sublinear_tf=True, stop_words = 'english')
vect.fit(x_train)


# In[7]:

x_train_df1 = vect.transform(x_train)
x_test_df1 = vect.transform(x_test)


# In[8]:

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(200)
svd1 = svd.fit(x_train_df1)
x_train_df = svd.transform(x_train_df1)
x_test_df = svd.transform(x_test_df1)
sorted(svd.explained_variance_ratio_)


# In[9]:

train_variants_df = train_v.drop(['ID','Class'], axis=1)
y = train_v['Class']
test_variants_df = test_v.drop(['ID'], axis=1)
data = train_variants_df.append(test_variants_df)
x_data = pd.get_dummies(data).values
x = x_data[:train_v.shape[0]]
x_test = x_data[train_v.shape[0]:]


# In[10]:

x.shape


# In[11]:

import scipy.sparse as ssp
x = ssp.hstack([x,pd.DataFrame(x_train_df)],format = 'csr')
x_test = ssp.hstack([x_test,pd.DataFrame(x_test_df)],format = 'csr')


# In[23]:

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state  = 500)


# In[25]:

y_pred = clf.fit(x,y)
y_pred


# In[26]:

y_pred = clf.predict(x)
sklearn.metrics.accuracy_score(y,y_pred)


# In[31]:

yfinal = clf.predict_proba(x_test)
clf.classes_


# In[39]:

data_pred = pd.DataFrame(yfinal, columns = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9'])


# In[40]:

data_pred.index.name = 'ID'


# In[42]:

#cols = ['ID','Class1','Class2', 'Class3','Class4','Class5','Class6', 'Class7','Class8','Class9']
#sub3.reindex_axis(cols,axis =1)
data_pred.to_csv("C:\\Users\\nitesh.garg\\Desktop\\Python\\cancer problem\\submission3.csv",index= True)


# In[ ]:




# In[17]:

sub1 = pd.DataFrame(yfinal,columns = ['Class'])
sub1.index.name = 'ID'
sub1.Class.astype(int)
sub1.iloc[0,0]


# In[79]:

for i in range(sub1.shape[0]):
    for j in range(9):
        sub1.loc[i,'Class' +  str(j+1)] = 0 
    sub1.loc[i,'Class' +  str(sub1.iloc[i,0])] = 1 


# In[80]:

sub1


# In[81]:

sub2  = sub1.drop(['Class'], axis = 1)
#sub3 = sub2.rename(columns={"index":'ID'})


# In[82]:

sub2


# In[83]:

#cols = ['ID','Class1','Class2', 'Class3','Class4','Class5','Class6', 'Class7','Class8','Class9']
#sub3.reindex_axis(cols,axis =1)
sub2.to_csv("C:\\Users\\nitesh.garg\\Desktop\\Python\\cancer problem\\submission2.csv",index= True)


# In[84]:

plt.figure(figsize=(12,8))
sns.countplot(x = 'Class',data = sub1, palette="Blues_d")
plt.show()

