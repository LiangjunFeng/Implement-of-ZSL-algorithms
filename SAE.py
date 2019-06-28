import scipy.io as sio
data = sio.loadmat('/Volumes/TOSHIBA EXT/TIZSL/ZSL/AwA/awa_demo_data.mat')

trainfeatures = data['X_tr']
train_attributelabel = data['S_tr']

testfeatures = data['X_te']

train_attributetable = data['S_te_pro']
test_attributetable = data['S_te_gt']

testlabel = data['param'][0][0][2]
        

def normalize(fea):
    nSmp,mFea = fea.shape
    feaNorm = np.sqrt(np.sum(np.square(fea),1))
    fea = fea/np.mat(feaNorm).T
    return fea

    
trainfeatures = normalize(trainfeatures.T).T
#testfeatures = normalize(testfeatures.T).T

lam = 500000

S = np.mat(train_attributelabel).T
X = np.mat(trainfeatures).T

A = S*S.T
B = lam*X*X.T
C = (1+lam)*S*X.T
W = solve_sylvester(A,B,C)

W = normalize(W)

test_pre_attribute =  testfeatures.dot(W.T)
test_attributetable = normalize(test_attributetable.T).T


print(test_pre_attribute.shape)

from sklearn.metrics.pairwise import cosine_similarity

dist = 1-cosine_similarity(test_pre_attribute,test_attributetable)

lis = data['param'][0][0][1]

label_lis = []
for i in range(dist.shape[0]):
    loc = dist[i].argmin()
    label_lis.append(lis[loc])

print(accuracy_score(list(testlabel),label_lis))


















