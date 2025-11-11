from flask import Flask, render_template, url_for, request, redirect

import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.neighbors as knn
import sklearn.metrics as met

import pandas as pd
import numpy as np
import os

app = Flask(__name__)
root = os.getcwd()
path = os.path.join(root, ) #'mysite'
app.secret_key = 'apisekar2111017'

def dataPreprocessing(Xtrain, Xtest):
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(Xtrain)
    X_train = scl.transform(Xtrain)
    X_test = scl.transform(Xtest)
    return X_train, X_test

def persentaseAkurasi(Xtrain, Xtest, y_train, y_test):
    a,b,pa = [0,1,4, 5,2,3], [2,3,5, 3,4,1], []
    klm1 = [Xtrain.axes[1][i] for i in a]
    klm2 = [Xtrain.axes[1][i] for i in b]
    model = knn.KNeighborsClassifier(n_neighbors=2, metric='euclidean')
    for i in range(0, len(Xtrain.axes[1])):
        X_train = Xtrain[[klm1[i], klm2[i]]]
        X_test = Xtest[[klm1[i], klm2[i]]]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        score = model.score(X_test, y_test)
        pa.append((y_test==y_predict))
        # endfor
    paTrue = list(filter(lambda x:x==True, [x for x in pa]))
    paFalse = list(filter(lambda x:x==False, [x for x in pa]))
    paMax = 16.5 * len(paTrue)
    paMin = 16.5 * len(paFalse)
    paMax = (paMax + 0.1) if (paMax == 0.0) else paMax
    paMin = (paMin + 0.1) if (paMin == 0.0) else paMin
    paMax = (paMax + 1.0) if (paMax == paMin) else paMax
    print('=== autentikasi_tandaTangan:',y_test, 'asli',paMax, 'palsu',paMin)
    return y_test, paMax, paMin

@app.route('/')
def home():
    return str(app.secret_key)+'.pythonanywhere.com/fiturIMG'

@app.route('/<fiturImg>')
def klasifikasiKNN_fitur(fiturImg):
    df = pd.read_csv(os.path.join(path, '..', 'fiturImg.csv')) # .. dihapus
    Xtrain, y_train = df.drop(['filename','label'], axis=1), df['label']
    f = []
    for i in fiturImg.split('-'):
        try:
            f.append(int(i))
        except Exception as e:
            f.append(float(i))

    # print(f, type(f))
    Xtest = pd.DataFrame(np.array([f]), columns=Xtrain.columns)
    # print(Xtest)
    X_train, X_test = dataPreprocessing(Xtrain, Xtest)
    model = knn.KNeighborsClassifier(n_neighbors=2, metric='euclidean')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    r_y, r_paMax, r_paMin = persentaseAkurasi(Xtrain, Xtest, y_train, y_predict)
    return str(r_y)+'_'+str(r_paMax)+'_'+str(r_paMin)

if __name__=='__main__':
    app.run(debug=True)
