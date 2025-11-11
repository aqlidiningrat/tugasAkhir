from flask import Flask
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
root = os.getcwd()
path = os.path.join(root) # 'mysite'

def r_default():
    return 'wanDinulAqli.pythonanywhere.com/07/chanel/youtube/Dr.fahrudinFaiz/jalaludinRumi/biarkanHatimuPecahAgarIaTerbuka...'

def cariDF(pathDS, namaDS):
    fldrMin2 = os.path.join(os.getcwd(), pathDS)
    for fldr in os.listdir(fldrMin2):
        fldrMin1 = os.path.join(fldrMin2, fldr)
        for fname in os.listdir(fldrMin1):
            if (fname == namaDS):
                pathDtrain = os.path.join(fldrMin1, fname)
                break
                # endif
            # endfor
        # endfor
    df = pd.read_excel(pathDtrain)
    return df

def dataPreprocessing(Xtrain, Xtest):
    import sklearn.preprocessing as pp
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(Xtrain)
    X_train = scl.transform(Xtrain)
    X_test = scl.transform(Xtest)
    return X_train, X_test

def dataModel(X_train, y_train, X_test, nk, dismet):
    import sklearn.neighbors as knn
    model = knn.KNeighborsClassifier(n_neighbors=nk, metric=dismet)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    h_predict = 'Laki-Laki' if y_predict == [0] else 'Perempuan'
    # print('*h_predict:',h_predict)
    return h_predict

def indexColumns(bIn, modulo):
    iCols = []
    for i in range(0, bIn):
        if (bIn%2==modulo)and(i<int(round(bIn/2)))and(int(i*2)!=int(bIn-1)):
            iCols.append(i*2)
            continue
            # endif
        # endfor
    # print('iCols:', iCols, len(iCols), '\n')
    return iCols

def singleColumn(Xtrain, y_train, Xtest, bIn, nk, dismet):
    X_train, X_test = Xtrain[[str(Xtrain.axes[1][int(bIn-1)])]], Xtest[[str(Xtest.axes[1][int(bIn-1)])]]
    return (dataModel(X_train, y_train, X_test, nk, dismet))

def persentase2(y_predict, perKlas):
    seperSepuluhKlas = round(float(100/perKlas), 1)
    seperSeratusKlas = round(float(seperSepuluhKlas*perKlas), 1)
    perSetengahMin = round(float(seperSeratusKlas/2), 1)
    perSetengahMax = round(float(seperSeratusKlas/2), 1)

    perSepuluhMax = round(float(seperSeratusKlas-perSetengahMin), 1)
    perSepuluhMin = round(float(seperSeratusKlas-perSetengahMax), 1)
    # kompensasi
    perSepuluhMax += 1
    perSepuluhMin -= 1
    # print('*perSepuluhMax:',perSepuluhMax, '*perSepuluhMin:',perSepuluhMin)

    perhL = perSepuluhMax if (y_predict=='Laki-Laki') else perSepuluhMin
    perhP = perSepuluhMax if (y_predict=='Perempuan') else perSepuluhMin
    # print('*y_predict:',y_predict, '*persentaseLaki:',perhL, '*persentasePerempuan:',perhP)
    return y_predict, perhL, perhP, perSepuluhMax

def persentase3(pa, y_predict):
    # print('*pa:',pa, '*len(pa):',len(pa))
    perKlas = len(pa)
    seperSepuluhKlas = round(float(100/perKlas), 1)
    seperSeratusKlas = round(float(seperSepuluhKlas*perKlas), 1)
    # print('*seperSepuluhKlas:',seperSepuluhKlas, '*seperSeratusKlas:',seperSeratusKlas)
    paL, paP = [], []
    for value in pa:
        paL.append(value) if (value=='Laki-Laki') else paP.append(value)
        # endfor
    perSatuL, perSatuP = float(seperSepuluhKlas*len(paL)), float(seperSepuluhKlas*len(paP))
    # print('*perSatuL:',perSatuL, '*perSatuP:',perSatuP)
    if (perSatuL == perSatuP):
        return persentase2(y_predict, perKlas)
        # endif
    perSatuMax = perSatuL if (perSatuL > perSatuP) else perSatuP
    perSatuMin = perSatuP if (perSatuP < perSatuL) else perSatuL

    perSepuluhMax = round(float(seperSeratusKlas-perSatuMin), 1)
    perSepuluhMin = round(float(seperSeratusKlas-perSatuMax), 1)
    # kompensasi
    perSepuluhMax -= 1
    perSepuluhMin += 1
    # print('*perSepuluhMax:',perSepuluhMax, '*perSepuluhMin:',perSepuluhMin)

    perhL = perSepuluhMax if (y_predict=='Laki-Laki') else perSepuluhMin
    perhP = perSepuluhMax if (y_predict=='Perempuan') else perSepuluhMin
    # print('*y_predict:',y_predict, '*persentaseLaki:',perhL, '*persentasePerempuan:',perhP)
    return y_predict, perhL, perhP, perSepuluhMax

def persentase(X_train, y_train, X_test, bIn, nk, dismet, y_predict):
    clm, pa = ['fhog'+str(i) for i in range(0, bIn)], []
    Xtrain = pd.DataFrame(X_train, columns=clm)
    Xtest = pd.DataFrame(X_test, columns=clm)
    modulo = 0 if (bIn%2==0) else 1
    for i in indexColumns(bIn, modulo):
        X_train = Xtrain[[str(Xtrain.axes[1][i]), str(Xtrain.axes[1][i+1])]]
        X_test = Xtest[[str(Xtest.axes[1][i]), str(Xtest.axes[1][i+1])]]
        pa.append(dataModel(X_train, y_train, X_test, nk, dismet))
        # endfor
    if (modulo==1):
        pa.append(singleColumn(Xtrain, y_train, Xtest, bIn, nk, dismet))
        # endif
    return persentase3(pa, y_predict)

def api_klasifikasiKNN(bIn, nk, dismet, namaApps, namaDS, fitur):
    if (namaApps == 't3-wan'):
        df = cariDF('dataFitur/citraIdeal', namaDS)
        bIn, nk, dismet = int(bIn), int(nk), str(dismet)
    elif (namaApps == 't5-adnarim'):
        df = cariDF('dataFiturLowBrightnes', namaDS)
        bIn, nk, dismet = int(bIn), int(nk), str(dismet)
    else:
        return r_default()
        # endif
    # klasifikasiKNN
    X_train, y_train = df.drop(['fname','label'], axis=1), df['label']
    X_test = pd.DataFrame(np.array([[int(i) for i in fitur.split('-')]]), columns=X_train.columns)
    X_train, X_test = dataPreprocessing(X_train, X_test)
    # print('*',X_train.shape, y_train.shape, X_test.shape)
    y_predict = dataModel(X_train, y_train, X_test, nk, dismet)
    return persentase(X_train, y_train, X_test, bIn, nk, dismet, y_predict)

@app.route('/')
def home():
    return r_default()

@app.route('/<bIn>/<nk>/<dismet>/<namaApps>/<namaDS>/<fitur>')
def responAPI(bIn, nk, dismet, namaApps, namaDS, fitur):
    h_predict, paL, paP, paMax = api_klasifikasiKNN(bIn, nk, dismet, namaApps, namaDS, fitur)
    respon = h_predict+'_'+str(paL)+'_'+str(paP)+'_'+str(paMax)
    return respon

if __name__=='__main__':
    app.run(debug=True)
