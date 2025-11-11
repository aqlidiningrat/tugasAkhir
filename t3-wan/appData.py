def get_pixel(img, center, x, y):
    new_value = 0
    try:
        # if local neighborhod pixel, value is greater than or equal, to center pixel values then, set it to 1
        if img[x][y] >= center:
            new_value = 1

    except Exception as e:
        # Exception is required when, neighborhod value of a center, pixel value is null i.e. values, present at boundaries
        pass

    return new_value

# feature for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))
    # top
    val_ar.append(get_pixel(img, center, x-1, y))
    # top_right
    val_ar.append(get_pixel(img, center, x-1, y+1))
    # right
    val_ar.append(get_pixel(img, center, x, y+1))
    # button_right
    val_ar.append(get_pixel(img, center, x+1, y+1))
    # button
    val_ar.append(get_pixel(img, center, x+1, y))
    # button_left
    val_ar.append(get_pixel(img, center, x+1, y-1))
    # left
    val_ar.append(get_pixel(img, center, x, y-1))

    # how we need to convert binary values to decimal
    power_val = [1,2,4,8,16,32,64,128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val

def ekstraksiFiturLBP(img, bIn):
    print('==> ekstraksiFiturLBP')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(img.shape)
    height, width = img.shape
    img_lbp = np.zeros((height, width), np.uint8)
    npHist = []
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i,j] = lbp_calculated_pixel(img, i, j)
            npHist.append(img_lbp[i,j])

    # preview_img
    print(img_lbp)

    # featureLBP (transform to hist_lbp)
    hist, bins = np.histogram(npHist, bins=bIn)
    print('bins:',bins)
    print('hist_lbp:',hist)
    return hist

def ekstraksiFiturHOG(img, bIn):
    print('==> ekstraksiFiturHOG')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(img.shape)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang / (2*np.pi))
    bin_cells = []
    mag_cells = []
    cellx, celly = 8,8

    for i in range(0, int(img.shape[0] / celly)):
        for j in range(0, int(img.shape[1] / cellx)):
            bin_cells.append(bin[i*celly: i*celly+celly, j+cellx: j*cellx+cellx])
            mag_cells.append(mag[i*celly: i*celly+celly, j+cellx: j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # preview_img
    img_hog = np.array(hists, 'uint8')
    print(img_hog)

    # fiturHOG (transform to hellinger kernel and hist_hog)
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps
    hist, bins = np.histogram(hist, bins=bIn)
    print('bins:',bins)
    print('hist_hog:', hist)
    return hist

def persiapanData(path, eksFitur, bIn):
    lisar = []
    print(os.listdir(path))
    for label_folder in os.listdir(path):
        print('\n ==>label_folder:',label_folder)
        label = 0 if label_folder == 'man' else 1 # ternary-operator
        in_folder_label = path+'/'+label_folder
        i = 0
        for file in os.listdir(in_folder_label):
            i += 1
            path_img = in_folder_label+'/'+file
            print('img:',path_img, i, 'dari', len(os.listdir(in_folder_label)))
            img = cv.imread(path_img)
            img_resize = cv.resize(img, (128,128))

            if (eksFitur=='lbp'):
                fiturLBP = ekstraksiFiturLBP(img_resize, bIn)
                listLBP = [int(fiturLBP[v]) for v in range(0, len(fiturLBP))]
                listLBP.insert(0, file), listLBP.append(label)
                lisar.append(listLBP)

                clm = [eksFitur+str(i) for i in range(0, len(fiturLBP))]
                clm.insert(0, 'fname'), clm.append('label')
                binFname = len(fiturLBP)

            elif (eksFitur=='hog'):
                fiturHOG = ekstraksiFiturHOG(img_resize, bIn)
                listHOG = [int(fiturHOG[v]) for v in range(0, len(fiturHOG))]
                listHOG.insert(0, file), listHOG.append(label)
                lisar.append(listHOG)

                clm = [eksFitur+str(i) for i in range(0, len(fiturHOG))]
                clm.insert(0, 'fname'), clm.append('label')
                binFname = len(fiturHOG)

            else:
                fiturLBP = ekstraksiFiturLBP(img_resize, bIn)
                fiturHOG = ekstraksiFiturHOG(img_resize, bIn)
                listLBP = [int(fiturLBP[v]) for v in range(0, len(fiturLBP))]
                listHOG = [int(fiturHOG[v]) for v in range(0, len(fiturHOG))]
                listFC = listLBP+listHOG
                listFC.insert(0, file), listFC.append(label)
                lisar.append(listFC)

                clm = [eksFitur+str(i) for i in range(0, int(len(fiturHOG)+len(fiturLBP)))]
                clm.insert(0, 'fname'), clm.append('label')
                binFname = int(len(fiturHOG)+len(fiturLBP))
            # endfor
        # endfor
    data = pd.DataFrame(lisar, columns=clm)
    print(data)
    save_filename = path.split('/')[-2]+'_'+path.split('/')[-1]+'_'+str(binFname)+'bin_'+eksFitur
    print(save_filename)
    data.to_excel('dataFitur/'+path.split('/')[-2]+'/'+eksFitur+'/'+save_filename+'.xlsx', sheet_name=save_filename, index=False)
    # data.to_excel(path.split('/')[-2]+'_'+eksFitur+'.xlsx', sheet_name=save_filename, index=False)

def dataPreprocessing(Xtrain, Xtest):
    import sklearn.preprocessing as pp
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(Xtrain)
    X_train = scl.transform(Xtrain)
    X_test = scl.transform(Xtest)
    return X_train, X_test

def dataModel(X_train, y_train, X_test, n_k, metrik_jarak):
    import sklearn.neighbors as knn
    model = knn.KNeighborsClassifier(n_neighbors=n_k, metric=metrik_jarak)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict

def reportKlasifikasi(y_test, y_predict):
    import sklearn.metrics as met
    accuracy = met.accuracy_score(y_test, y_predict)
    confusionmatrix = met.confusion_matrix(y_test, y_predict)
    precision = met.precision_score(y_test, y_predict)
    sensitifity = met.recall_score(y_test, y_predict)
    report = met.classification_report(y_test, y_predict)

    print('accuracy:',round(accuracy, 2))
    print(confusionmatrix)
    print('precision:',round(precision, 2))
    print('sensitifity:',round(sensitifity, 2))
    print(report)

def akurasiModel(dftrain, dftest, n_k, metrik_jarak):
    print('\n *',dftrain.split('/')[-1],'||',dftest.split('/')[-1], '|| k:',n_k, 'metric:',metrik_jarak)
    df = pd.read_excel(dftrain)
    dfman, dfwoman = df.loc[df['label'] == 0], df.loc[df['label'] == 1]

    X_train = df.drop(['fname','label'], axis=1)
    y_train = df['label']
    print(X_train.shape, 'dftrain man:',len(dfman), 'dftrain woman:',len(dfwoman))

    df2 = pd.read_excel(dftest)
    df2man, df2woman = df2.loc[df2['label'] == 0], df2.loc[df2['label'] == 1]

    X_test = df2.drop(['fname','label'], axis=1)
    y_test = df2['label']
    print(X_test.shape, 'dftest man:',len(df2man), 'dftest woman:',len(df2woman))
    print('total man:', len(dfman)+len(df2man), 'total woman:', len(dfwoman)+len(df2woman))

    X_train, X_test = dataPreprocessing(X_train, X_test)
    y_predict = dataModel(X_train, y_train, X_test, n_k, metrik_jarak)
    reportKlasifikasi(y_test, y_predict)

if __name__=='__main__':
    import cv2 as cv
    import pandas as pd
    import numpy as np
    import os

    # for kualitasCitra in ['citraIdeal','citraTidakIdeal','citraGabungan']:
    #     for i in range(2,10):
    #         persiapanData(path='d:/dataset/train_test_balance/'+kualitasCitra+'/train', eksFitur='lbp', bIn=i)
    #         persiapanData(path='d:/dataset/train_test_balance/'+kualitasCitra+'/test', eksFitur='lbp', bIn=i)
    #         persiapanData(path='d:/dataset/train_test_balance/'+kualitasCitra+'/train', eksFitur='hog', bIn=i)
    #         persiapanData(path='d:/dataset/train_test_balance/'+kualitasCitra+'/test', eksFitur='hog', bIn=i)
    #         persiapanData(path='d:/dataset/train_test_balance/'+kualitasCitra+'/train', eksFitur='fc', bIn=i)
    #         persiapanData(path='d:/dataset/train_test_balance/'+kualitasCitra+'/test', eksFitur='fc', bIn=i)

    # persiapanData(path='d:/dataset/train_test_balance/citraIdeal/test', eksFitur='fc', bIn=2)

    akurasiModel(dftrain='dataFitur/citraIdeal/fc/citraIdeal_train_18bin_fc.xlsx', dftest='dataFitur/citraIdeal/fc/citraIdeal_test_18bin_fc.xlsx', n_k=31, metrik_jarak='manhattan')
    # akurasiModel(dftrain='dataFitur/citraTidakIdeal/fc/citraTidakIdeal_train_18bin_fc.xlsx', dftest='dataFitur/citraTidakIdeal/fc/citraTidakIdeal_test_18bin_fc.xlsx', n_k=21, metrik_jarak='minkowski')
    # akurasiModel(dftrain='dataFitur/citraGabungan/fc/citraGabungan_train_18bin_fc.xlsx', dftest='dataFitur/citraGabungan/fc/citraGabungan_test_18bin_fc.xlsx', n_k=31, metrik_jarak='minkowski')

    # dftrain, dftest = 'dataFitur/citraIdeal/fc/citraIdeal_train_18bin_fc.xlsx', 'test_fc.xlsx'
    # for bbnn in ['18']:
    #     for mtrjk in ['l2','cityblock','sqeuclidean','l1','euclidean','nan_euclidean','manhattan','minkowski']:
    #         for nnkk in [11,19,21,31]:
    #             akurasiModel(dftrain=dftrain, dftest=dftest, n_k=nnkk, metrik_jarak=mtrjk)
