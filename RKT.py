import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso,SGDRegressor,PassiveAggressiveRegressor,ElasticNet,LinearRegression

path = '/Users/zhuxiaoxiansheng/Desktop/Animals_with_Attributes2/'

classname = pd.read_csv(path+'classes.txt',header=None,sep = '\t')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
    
def make_test_attributetable():
    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous-01.txt',header=None,sep = ',')
    test_classes = pd.read_csv(path+'testclasses.txt',header=None)
    test_classes_flag = []
    for item in test_classes.iloc[:,0].values.tolist():
        test_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[test_classes_flag,:]

def make_train_attributetable():
    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous-01.txt',header=None,sep = ',')
    train_classes = pd.read_csv(path+'trainclasses.txt',header=None)
    train_classes_flag = []
    for item in train_classes.iloc[:,0].values.tolist():
        train_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[train_classes_flag,:]

def construct_Y(label_onehot):
    for i in range(label_onehot.shape[0]):
        for j in range(label_onehot.shape[1]):
            if label_onehot[i][j] == 0:
                label_onehot[i][j] = -1
    return np.mat(label_onehot)

def generate_data(data_mean,data_std,attribute_table,num):
    class_num = data_mean.shape[0]
    feature_num = data_mean.shape[1]
    data_list = []
    label_list = []
    for i in range(class_num):
        data = []
        for j in range(feature_num):
            data.append(list(np.random.normal(data_mean[i,j],np.abs(data_std[i,j]),num)))
        data = np.row_stack(data).T
        data_list.append(data)   
        label_list+=[test_attributetable.iloc[i,:].values]*num
    return np.row_stack(data_list),np.row_stack(label_list)

trainlabel = np.load(path+'AWA2_trainlabel.npy')
train_attributelabel = np.load(path+'AWA2_train_continuous_01_attributelabel.npy')

testlabel = np.load(path+'AWA2_testlabel.npy')
test_attributelabel = np.load(path+'AWA2_test_continuous_01_attributelabel.npy')

enc1 = OneHotEncoder()
enc1.fit(np.mat(trainlabel).T)
trainlabel_onehot = enc1.transform(np.mat(trainlabel).T).toarray()

enc2 = OneHotEncoder()
enc2.fit(np.mat(testlabel).T)
testlabel_onehot = enc2.transform(np.mat(testlabel).T).toarray()

trainfeatures = np.load(path+'resnet101_trainfeatures.npy')
testfeatures = np.load(path+'resnet101_testfeatures.npy')

print(trainfeatures.shape,trainlabel.shape,train_attributelabel.shape,trainlabel_onehot.shape)
print(testfeatures.shape,testlabel.shape,test_attributelabel.shape,testlabel_onehot.shape)


train_attributetable = make_train_attributetable()
test_attributetable = make_test_attributetable()



trainfeatures_tabel = pd.DataFrame(trainfeatures)
trainfeatures_tabel['label'] = trainlabel

trainfeature_mean = np.mat(trainfeatures_tabel.groupby('label').mean().values).T
trainfeature_std = np.mat(trainfeatures_tabel.groupby('label').std().values).T


clf = Lasso(alpha=0.01)
clf.fit(np.mat(train_attributetable.values).T,np.mat(test_attributetable.values).T)
W = clf.coef_.T


virtual_testfeature_mean = (trainfeature_mean*W).T
virtual_testfeature_std = np.ones(virtual_testfeature_mean.shape)*0.3

virtual_testfeature,virtual_test_attributelabel = generate_data(virtual_testfeature_mean,virtual_testfeature_std,test_attributetable,50)
print(virtual_testfeature.shape,virtual_test_attributelabel.shape)

rand_index = np.random.choice(virtual_testfeature.shape[0],virtual_testfeature.shape[0],replace=False)
virtual_testfeature = virtual_testfeature[rand_index]
virtual_test_attributelabel = virtual_test_attributelabel[rand_index]


res_list = []
for i in range(virtual_test_attributelabel.shape[1]):
    print("{} th classifier is training".format(i+1))
    clf = LinearRegression()
    clf.fit(virtual_testfeature,virtual_test_attributelabel[:,i])
    res = clf.predict(testfeatures)
    res_list.append(list(res))

test_pre_attribute = np.mat(np.row_stack(res_list)).T
print(test_pre_attribute.shape)
test_attributetable = make_test_attributetable()

label_lis = []
for i in range(test_pre_attribute.shape[0]):
    pre_res = test_pre_attribute[i,:]
    loc = np.sum(np.square(test_attributetable.values - pre_res),axis=1).argmin()
    label_lis.append(test_attributetable.index[loc])

print(accuracy_score(list(testlabel),label_lis))













