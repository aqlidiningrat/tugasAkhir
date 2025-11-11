def show_plt(caption, img):
    cv.imshow(caption, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite(os.path.join('d:/', caption), img)

def buatFolder(new_path):
    if (not os.path.exists(new_path)):
        os.makedirs(new_path)
    else:
        pass
    return new_path

def low_brightnes(img, alpha):
    img_normal = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    img_alpha = img_normal * alpha
    img_low_brightnes = img_alpha.astype(np.uint8)
    img_low_brightnes[img_low_brightnes < 0] = 0
    img_low_brightnes[img_low_brightnes > 255] = 255
    return img_low_brightnes

def dataFaceLowBrightnes():
    root = os.getcwd()
    path = os.path.join(root, 'dataFace')
    for alpha in ['3', '2', '1']:
        for train_or_test in os.listdir(path):
            for man_or_woman in os.listdir(os.path.join(path, train_or_test)):
                iplus, in_path_gender = 0, os.path.join(path, train_or_test, man_or_woman)
                new_path = buatFolder(os.path.join(root, 'dataFaceLowBrightnes', 'low_brightnes_0'+alpha,
                                                    train_or_test, man_or_woman))
                for fname in os.listdir(in_path_gender):
                    img = cv.imread(os.path.join(in_path_gender, fname))
                    img_low_brightnes = low_brightnes(img, float('0.'+alpha))
                    cv.imwrite(os.path.join(new_path, '0'+alpha+'_'+fname), cv.resize(img_low_brightnes, (128,128)))
                    iplus += 1
                    print('low_brightnes_0'+alpha, train_or_test, man_or_woman, img.shape,
                            iplus,':',len(os.listdir(in_path_gender)))
                    # endfor fname
                # endfor man_or_woman
            # endfor train_or_test
        # endfor alpha
    print('===================',len(os.listdir(new_path)) == iplus)

def __levelGray(lv01=False, lv02=False, lv03=False):
    root = os.getcwd()
    path = os.path.join(root, 'dataFaceLowBrightnes')
    lisar_levelGray = []
    for alpha in ['03','02','01']:
        in_path = os.path.join(path, 'low_brightnes_'+alpha)
        for train_or_test in os.listdir(in_path):
            for man_or_woman in os.listdir(os.path.join(in_path, train_or_test)):
                iplus, in_path_gender = 0, os.path.join(in_path, train_or_test, man_or_woman)
                for fname in os.listdir(in_path_gender):
                    img = cv.resize(cv.imread(os.path.join(in_path_gender, fname)), (128,128))
                    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    lisar_levelGray.append([round(img_gray.mean(),1), fname.split('_')[0], fname])
                    iplus += 1
                    # endfor fname
                # endfor man_or_woman
            # endfor train_or_test
        # endfor alpha
        df_levelGray = pd.DataFrame(lisar_levelGray, columns=['meanPixel','levelGray','fname'])
        max_low = df_levelGray.loc[df_levelGray['levelGray'] == alpha]
        print(max_low.loc[max_low['meanPixel'] == max_low['meanPixel'].max()])
    # print(df_levelGray, df_levelGray['levelGray'].value_counts())

def ekstraksiFiturHOG(img, bIn, fname, label):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gx = cv.Sobel(img_gray, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img_gray, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang / (2*np.pi))
    bin_cells = []
    mag_cells = []
    cellx, celly = 8,8
    for i in range(0, int(img_gray.shape[0] / celly)):
        for j in range(0, int(img_gray.shape[1] / cellx)):
            bin_cells.append(bin[i*celly: i*celly+celly, j+cellx: j*cellx+cellx])
            mag_cells.append(mag[i*celly: i*celly+celly, j+cellx: j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    # preview_img
    arr_hog = np.array(hists, 'uint8')
    img_hog = cv.resize(arr_hog, (img_gray.shape))
    # print('img_hog', img_hog)
    # fiturHOG (transform to hellinger kernel and hist_hog)
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps
    hist, bins = np.histogram(hist, bins=bIn)
    listHOG = [int(hist[v]) for v in range(0, len(hist))]
    listHOG.insert(0, fname), listHOG.append(label)
    return img_gray.shape, listHOG

def ekstraksiFiturSLIC(img, bIn, fname, label_):
    slic = cv.ximgproc.createSuperpixelSLIC(img) # region_size=25, ruler=10.0
    slic.iterate() # 10
    labels = slic.getLabels()
    segmented_img = np.zeros_like(img)
    for label in np.unique(labels):
        mask = labels == label
        segmented_img[mask] = np.mean(img[mask], axis=0)

    # preview_img
    slic_img = segmented_img.astype(np.uint8)
    # print('slic_img', slic_img)
    slic_img_gray = cv.cvtColor(slic_img, cv.COLOR_BGR2GRAY)
    hist, bins = np.histogram(slic_img_gray, bins=bIn)
    listSLIC = [int(hist[i]) for i in range(0, len(hist))]
    listSLIC.insert(0, fname), listSLIC.append(label_)
    return slic_img_gray.shape, listSLIC

def pca_img(img, bIn, fname, label):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # create a PCA object with 4 principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=bIn)
    # pca implementation on img
    pca_attributes = pca.fit_transform(img_gray)
    pca_variance = pca.explained_variance_ratio_
    np_pcaVariance = np.array(pca_variance, 'float64')
    listPCA = [round(float(np_pcaVariance[i]), 3) for i in range(0, len(np_pcaVariance))]
    listPCA.insert(0, fname), listPCA.append(label)
    # see the variance of each attribute
    return img_gray.shape, listPCA

def gabunganFeature(img, bIn, fname, label):
    _, h = ekstraksiFiturHOG(img, bIn, fname, label)
    _, s = ekstraksiFiturSLIC(img, bIn, fname, label)
    _, p = pca_img(img, bIn, fname, label)
    del h[-1]
    del s[0]
    del s[-1]
    del p[0]
    fg = h+s+p
    _ = (int(len(h)-1), int(len(s)), int(len(p)-1))
    return _, fg

def clm_df(cname, bIn):
    clm = [cname+str(i) for i in range(0, bIn)]
    clm.insert(0, 'fname'), clm.append('label')
    return clm

def save_df(df, fnm1, fnm2):
    root = os.getcwd()
    path = os.path.join(root,'dataFiturLowBrightnes',fnm1, fnm2)
    df.to_excel(path, sheet_name=fnm1, index=False)
    return print(df), print(df['label'].value_counts())

def dataFiturLowBrightnes(bIn, fg=False):
    root = os.getcwd()
    path = os.path.join(root, 'dataFaceLowBrightnes')
    for alpha in ['03','02','01']:
        lvbrightnes = 'low_brightnes_'+alpha
        in_path = os.path.join(path, 'low_brightnes_'+alpha)
        new_path = buatFolder(os.path.join(root, 'dataFiturLowBrightnes', lvbrightnes))
        for train_or_test in os.listdir(in_path):
            lisarHOG, lisarSLIC, lisarPCA, lisarGabungan = [], [], [], []
            for man_or_woman in os.listdir(os.path.join(in_path, train_or_test)):
                iplus, in_path_gender = 0, os.path.join(in_path, train_or_test, man_or_woman)
                label = 0 if (man_or_woman=='man') else 1
                for fname in os.listdir(in_path_gender):
                    img = cv.resize(cv.imread(os.path.join(in_path_gender, fname)), (128,128))
                    _,featureHOG = ekstraksiFiturHOG(img, bIn, fname, label)
                    lisarHOG.append(featureHOG)
                    _,featureSLIC = ekstraksiFiturSLIC(img, bIn, fname, label)
                    lisarSLIC.append(featureSLIC)
                    _,featurePCA = pca_img(img, bIn, fname, label)
                    lisarPCA.append(featurePCA)
                    print('*HOG feature', _, str(bIn)+'_bin', featureHOG)
                    print('*SLIC feature', _, str(bIn)+'_bin', featureSLIC)
                    print('*PCA feature', _, str(bIn)+'_bin', featurePCA)
                    if fg:
                        _,featureGabungan = gabunganFeature(img, bIn, fname, label)
                        lisarGabungan.append(featureGabungan)
                        print('*featureGabungan', _, str(bIn*3)+'_bin', featureGabungan)
                        # endif
                    iplus += 1
                    print('===========in_path: dataFaceLowBrightnes/low_brightnes_'+alpha, train_or_test, man_or_woman,
                            '-', iplus,':',len(os.listdir(in_path_gender)),'\n')
                    # endfor fname
                # endfor man_or_woman
            # endfor train_or_test
            save_df(df=pd.DataFrame(lisarHOG, columns=clm_df(cname='fhog', bIn=bIn)), fnm1=lvbrightnes,
                    fnm2=str(bIn)+'binHOG_'+train_or_test+'_lv'+alpha+'.xlsx')
            save_df(df=pd.DataFrame(lisarSLIC, columns=clm_df(cname='fslic', bIn=bIn)), fnm1=lvbrightnes,
                    fnm2=str(bIn)+'binSLIC_'+train_or_test+'_lv'+alpha+'.xlsx')
            save_df(df=pd.DataFrame(lisarPCA, columns=clm_df(cname='fpca', bIn=bIn)), fnm1=lvbrightnes,
                    fnm2=str(bIn)+'binPCA_'+train_or_test+'_lv'+alpha+'.xlsx')
            if fg:
                save_df(df=pd.DataFrame(lisarGabungan, columns=clm_df(cname='fg', bIn=bIn*3)), fnm1=lvbrightnes,
                    fnm2=str(bIn*3)+'binFG_'+train_or_test+'_lv'+alpha+'.xlsx')
        # endfor alpha
    print('===================',len(os.listdir(in_path_gender)) == iplus)

def dataPreprocessing(Xtrain, Xtest):
    import sklearn.preprocessing as pp
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(Xtrain)
    X_train = scl.transform(Xtrain)
    X_test = scl.transform(Xtest)
    return X_train, X_test

def machineLearningModel_KNN(X_train, y_train, X_test, nk, distance_metric):
    import sklearn.neighbors as knn
    model = knn.KNeighborsClassifier(n_neighbors=nk, metric=distance_metric)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict

def reportKlasifikasi(y_test, y_predict):
    import sklearn.metrics as met
    accuracy = met.accuracy_score(y_test, y_predict)
    confusionmatrix = met.confusion_matrix(y_test, y_predict)
    # print(confusionmatrix)
    precision = met.precision_score(y_test, y_predict)
    sensitifity = met.recall_score(y_test, y_predict)
    report = met.classification_report(y_test, y_predict)
    # print(report)
    return round(accuracy, 2), round(precision, 2), round(sensitifity, 2)

def modelClassifier(pathDtrain, pathDtest, nk, distance_metric):
    # __pathDtrain
    df = pd.read_excel(pathDtrain)
    dfman, dfwoman = df.loc[df['label'] == 0], df.loc[df['label'] == 1]
    X_train, y_train = df.drop(['fname','label'], axis=1), df['label']
    # __pathDtest
    df2 = pd.read_excel(pathDtest)
    df2man, df2woman = df2.loc[df2['label'] == 0], df2.loc[df2['label'] == 1]
    X_test, y_test = df2.drop(['fname','label'], axis=1), df2['label']

    # print('*',pathDtrain.split('\\')[-1],'||',pathDtest.split('\\')[-1],'|| k:',nk,'metric:',distance_metric)
    # print(X_train.shape, 'pathDtrain man:',len(dfman), 'pathDtrain woman:',len(dfwoman))
    # print(X_test.shape, 'pathDtest man:',len(df2man), 'pathDtest woman:',len(df2woman))
    # print('total man:', len(dfman)+len(df2man), 'total woman:', len(dfwoman)+len(df2woman))

    X_train, X_test = dataPreprocessing(X_train, X_test)
    y_predict = machineLearningModel_KNN(X_train, y_train, X_test, nk, distance_metric)
    accuracy, precision, recall = reportKlasifikasi(y_test, y_predict)
    pdtr, pdts = pathDtrain.split('\\')[-1], pathDtest.split('\\')[-1]
    _ = ('accuracy:',accuracy, 'precision:',precision, 'recall:',recall, 'k:',nk, 'metric:',distance_metric, pdtr, pdts)
    return accuracy, _

def akurasiTertinggi(bn, lv, dismet='euclidean', fg=False):
    print('===================Mulai low_brightnes_'+lv+', bin:'+str(bn)+', metric:'+dismet+', KNeighborsClassifier', 'fg =',fg)
    pathDs = os.path.join(os.getcwd(), 'dataFiturLowBrightnes', 'low_brightnes_'+lv)
    pdtr = os.path.join(pathDs, str(bn)+'binHOG_train_lv'+lv+'.xlsx')
    pdts = os.path.join(pathDs, str(bn)+'binHOG_test_lv'+lv+'.xlsx')
    pdtr2 = os.path.join(pathDs, str(bn)+'binSLIC_train_lv'+lv+'.xlsx')
    pdts2 = os.path.join(pathDs, str(bn)+'binSLIC_test_lv'+lv+'.xlsx')
    pdtr3 = os.path.join(pathDs, str(bn)+'binPCA_train_lv'+lv+'.xlsx')
    pdts3 = os.path.join(pathDs, str(bn)+'binPCA_test_lv'+lv+'.xlsx')
    if fg:
        pdtr4 = os.path.join(pathDs, str(bn*3)+'binFG_train_lv'+lv+'.xlsx')
        pdts4 = os.path.join(pathDs, str(bn*3)+'binFG_test_lv'+lv+'.xlsx')
        # endif
    for k in [15,17,35]:
        if (k%2 == 0):
            pass
        else:
            accuracyHOG, _ = modelClassifier(pathDtrain=pdtr, pathDtest=pdts, nk=k, distance_metric=dismet)
            accuracySLIC, _2 = modelClassifier(pathDtrain=pdtr2, pathDtest=pdts2, nk=k, distance_metric=dismet)
            accuracyPCA, _3 = modelClassifier(pathDtrain=pdtr3, pathDtest=pdts3, nk=k, distance_metric=dismet)
            if fg:
                accuracyFG, _4 = modelClassifier(pathDtrain=pdtr4, pathDtest=pdts4, nk=k, distance_metric=dismet)
                # endif
            if (accuracyHOG >= 0.70):
                if fg:
                    print(_), print(_2), print(_3), print(_4), print('')
                else:
                    print(_), print(_2), print(_3), print('')
                # endif
            # endif
        # endfor

def dataModel(X_train, y_train, X_test, nk):
    import sklearn.neighbors as knn
    model = knn.KNeighborsClassifier(n_neighbors=nk, metric='euclidean')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    h_predict = 'Laki-Laki' if y_predict == [0] else 'Perempuan'
    print('*h_predict:',h_predict)
    return h_predict

def indexColumns(bIn, modulo):
    iCols = []
    for i in range(0, bIn):
        if (bIn%2==modulo)and(i<int(round(bIn/2)))and(int(i*2)!=int(bIn-1)):
            iCols.append(i*2)
            continue
            # endif
        # endfor
    print('*iCols:',iCols, len(iCols), '\n')
    return iCols

def singleColumn(Xtrain, y_train, Xtest, bIn, nk):
    X_train, X_test = Xtrain[[str(Xtrain.axes[1][int(bIn-1)])]], Xtest[[str(Xtest.axes[1][int(bIn-1)])]]
    return (dataModel(X_train, y_train, X_test, nk))

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
    print('*perSepuluhMax:',perSepuluhMax, '*perSepuluhMin:',perSepuluhMin)

    perhL = perSepuluhMax if (y_predict=='Laki-Laki') else perSepuluhMin
    perhP = perSepuluhMax if (y_predict=='Perempuan') else perSepuluhMin
    print('*y_predict:',y_predict, '*persentaseLaki:',perhL, '*persentasePerempuan:',perhP)
    return y_predict, perhL, perhP, perSepuluhMax

def persentase3(pa, y_predict):
    print('*pa:',pa, '*len(pa):',len(pa))
    perKlas = len(pa)
    seperSepuluhKlas = round(float(100/perKlas), 1)
    seperSeratusKlas = round(float(seperSepuluhKlas*perKlas), 1)
    # seperSeratusKlas -= 1
    print('*seperSepuluhKlas:',seperSepuluhKlas, '*seperSeratusKlas:',seperSeratusKlas)
    paL, paP = [], []
    for value in pa:
        paL.append(value) if (value=='Laki-Laki') else paP.append(value)
        # endfor
    perSatuL, perSatuP = float(seperSepuluhKlas*len(paL)), float(seperSepuluhKlas*len(paP))
    print('*perSatuL:',perSatuL, '*perSatuP:',perSatuP)
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
    print('*perSepuluhMax:',perSepuluhMax, '*perSepuluhMin:',perSepuluhMin)

    perhL = perSepuluhMax if (y_predict=='Laki-Laki') else perSepuluhMin
    perhP = perSepuluhMax if (y_predict=='Perempuan') else perSepuluhMin
    print('*y_predict:',y_predict, '*persentaseLaki:',perhL, '*persentasePerempuan:',perhP)
    return y_predict, perhL, perhP, perSepuluhMax

def persentase(X_train, y_train, X_test, bIn, nk, y_predict):
    clm, pa = ['fhog'+str(i) for i in range(0, bIn)], []
    Xtrain = pd.DataFrame(X_train, columns=clm)
    Xtest = pd.DataFrame(X_test, columns=clm)
    modulo = 0 if (bIn%2==0) else 1
    for i in indexColumns(bIn, modulo):
        X_train = Xtrain[[str(Xtrain.axes[1][i]), str(Xtrain.axes[1][i+1])]]
        X_test = Xtest[[str(Xtest.axes[1][i]), str(Xtest.axes[1][i+1])]]
        pa.append(dataModel(X_train, y_train, X_test, nk))
        # endfor
    if (modulo==1):
        pa.append(singleColumn(Xtrain, y_train, Xtest, bIn, nk))
        # endif
    return persentase3(pa, y_predict)

def persentaseAkurasi(bIn, nk):
    pxtr = os.path.join(os.getcwd(), 'dataFiturLowBrightnes', 'low_brightnes_03', str(bIn)+'binHOG_train_lv03.xlsx')
    pxts = os.path.join(os.getcwd(), 'dataFiturLowBrightnes', 'low_brightnes_03', str(bIn)+'binHOG_test_lv03.xlsx')
    Xtrain, Xtest = pd.read_excel(pxtr), pd.read_excel(pxtr)
    X_train, X_test = Xtrain.drop(['fname','label'], axis=1), Xtest.drop(['fname','label'], axis=1)
    y_train = Xtrain['label']
    X_test = X_test.head(1)
    print(X_train.shape, y_train.shape, X_test.shape)
    print('*bIn:',bIn, '\n')
    X_train, X_test = dataPreprocessing(X_train, X_test)
    y_predict = dataModel(X_train, y_train, X_test, nk)
    h_predict, perhL, perhP, paMax = persentase(X_train, y_train, X_test, bIn, nk, y_predict)

def machineLearningModel_SVM(c, dfnm, lv):
    path_dtr = os.path.join(os.getcwd(), 'dataFiturLowBrightnes', 'low_brightnes_'+lv, dfnm+'_train_lv'+lv+'.xlsx')
    path_dts = os.path.join(os.getcwd(), 'dataFiturLowBrightnes', 'low_brightnes_'+lv, dfnm+'_test_lv'+lv+'.xlsx')
    dftrain = pd.read_excel(path_dtr)
    dftest = pd.read_excel(path_dts)
    X_train = dftrain.drop(['fname','label'], axis=1)
    y_train = dftrain['label']
    X_test = dftest.drop(['fname','label'], axis=1)
    y_test = dftest['label']
    import sklearn.svm as clfsvm
    model = clfsvm.SVC(C=float(c))
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy, precision, recall = reportKlasifikasi(y_test, y_predict)
    hasil = ('accuracy:',accuracy, 'precision:',precision, 'recall:',recall)
    print('*', path_dtr.split('\\')[-1], '||', path_dts.split('\\')[-1])
    return str(c),hasil

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import cv2 as cv
    import random
    import os

    # =======keperluan mendadak
    dt = pd.read_excel(os.path.join(os.getcwd(), 'Book2.xlsx'))
    # print(dt.info())

    # untuk pca
    df = dt
    # df = dt.drop(dt.axes[1][-1], axis=1)
    df['jarak'] = dt[dt.axes[1][-1]].round().astype('int')
    # print(df.info())

    k = int(input('K: '))
    dt2 = df.sort_values(by=df.axes[1][-1], ascending=True)
    dt2['nearest'] = [i if(i<=k) else i for i in range(1, len(dt2)+1)]
    dt3 = dt2.sort_index(axis=0)
    # print(dt3.info())

    dt4 = dt3.loc[dt3['nearest'] <= k]
    # print(dt4['label'].value_counts())
    dt3['label_inNearest'] = ['True' if(i in dt4.index) else 'False' for i in dt3.index]

    for newClm in dt4['label'].unique():
        dtClm = dt4.loc[dt4['label'] == newClm]
        dt3['sumLabel_'+str(newClm)] = [len(dtClm) for i in range(0, len(dt3))]
        # endfor
    dt3['k'] = [k for i in range(0, len(dt3))]

    # save
    dt3.to_excel('fix-nearest.xlsx', sheet_name='fix-nearest', index=False)
    print(dt3.info())
    print('*selesai..')

    # print('========================================',int(191.5*2))
    # dataFaceLowBrightnes()
    # __levelGray(lv01=19.2, lv02=39.0, lv03=58.6)

    # print('========================================',int(193.3*2))
    # dataFiturLowBrightnes(bIn=9)
    # dataFiturLowBrightnes(bIn=8)
    # dataFiturLowBrightnes(bIn=18)
    # dataFiturLowBrightnes(bIn=9, fg=True)

    # print('========================================',int(196.5*2))
    # akurasiTertinggi(bn=8, lv='01')
    # akurasiTertinggi(bn=18, lv='02')
    # akurasiTertinggi(bn=9, lv='03')

    # # =========klasifikasiSVM
    # print('\n ======================================= fg=True')
    # akurasiTertinggi(bn=9, lv='01', fg=True)
    # akurasiTertinggi(bn=9, lv='02', fg=True)
    # akurasiTertinggi(bn=9, lv='03', fg=True)

    # print('======================================== (persentaseAntarKelas)')
    # persentaseAkurasi(bIn=9, nk=35)

    # # =========klasifikasiSVM
    # lv = '01'
    # for bn in ['8bin','18bin','9bin']:
    #     for ef in ['HOG','SLIC','PCA']:
    #         for c in ['0.1', '0.5', '0.9']:
    #             dfnm = bn+ef
    #             c, hasil = machineLearningModel_SVM(float(c), dfnm, lv)
    #             print(' SVM_c='+c+':',hasil)
    #             # endfor
    #     slv = lv[1]
    #     ilv = int(slv) + 1
    #     lv = '0'+str(ilv)
    #     print('')
    # print('================fg=True')
    # for lv in ['01','02','03']:
    #     for c in ['0.1','0.5','0.9']:
    #         _, hasil = machineLearningModel_SVM(float(c), '27binFG', lv)
    #         print('SVM_c='+_+':',hasil)
    #     print('')
    #     # endfor

    # # ====================citraWajah-benar-salah
    # def modelKNN(X_train, y_train, X_test, y_test, nk=35):
    #     import sklearn.neighbors as knn
    #     model = knn.KNeighborsClassifier(n_neighbors=nk, metric='euclidean')
    #     model.fit(X_train, y_train)
    #     y_predict = model.predict(X_test)
    #     return y_predict, model.score(X_test, y_test)
    #
    # def modelSVM(X_train, y_train, X_test, y_test, nc=0.9):
    #     import sklearn.svm as clfsvm
    #     model = clfsvm.SVC(C=float(nc))
    #     model.fit(X_train, y_train)
    #     y_predict = model.predict(X_test)
    #     return y_predict, model.score(X_test, y_test)
    #
    # path = os.path.join(os.getcwd(),'dataFiturLowBrightnes')
    # pathlv = os.listdir(path)
    # bin = ['8bin','18bin','9bin']
    # lv = ['lv01','lv02','lv03']
    # for i in range(0, len(lv)):
    #     pathDataTrain = os.path.join(path, pathlv[i], str(bin[i]+'HOG'+'_train_'+lv[i]+'.xlsx'))
    #     pathDataTest = os.path.join(path, pathlv[i], str(bin[i]+'HOG'+'_test_'+lv[i]+'.xlsx'))
    #     dtrain = pd.read_excel(pathDataTrain)
    #     dtest = pd.read_excel(pathDataTest)
    #     X_train = dtrain.drop(['fname','label'], axis=1)
    #     X_test = dtest.drop(['fname','label'], axis=1)
    #     y_train = dtrain['label']
    #     y_test = dtest['label']
    #     X_train, X_test = dataPreprocessing(X_train, X_test)
    #     y_predict, score = modelKNN(X_train, y_train, X_test, y_test)
    #     # y_predict = modelSVM(X_train, y_train, X_test)
    #     dtest['y_predict'] = y_predict
    #     positive = dtest[dtest['label']==0]
    #     negative = dtest[dtest['label']==1]
    #     tp = positive['fname'].loc[positive['y_predict']==0]
    #     fp = positive['fname'].loc[positive['y_predict']==1]
    #     tn = negative['fname'].loc[negative['y_predict']==1]
    #     fn = negative['fname'].loc[negative['y_predict']==0]
    #     print('*',pathDataTrain.split('\\')[-1],pathDataTest.split('\\')[-1], '*score:',round(score, 2))
    #     print('tp:',len(tp), 'fp:',len(fp), 'tn:',len(tn), 'fn:',len(fn))
    #
    #
    #     pathtp = 'dataFaceHasilKlasifikasi/'+pathlv[i]+'/TP'
    #     os.makedirs(pathtp, exist_ok=True)
    #     for fnameImg in tp:
    #         pathFnameImg = os.path.join(os.getcwd(),'dataFaceLowBrightnes',pathlv[i],'test','man',fnameImg)
    #         img = cv.imread(pathFnameImg)
    #         cv.imwrite(os.path.join(pathtp, fnameImg), img)
    #         # endofr
    #     print(len(os.listdir(pathtp)))
    #
    #     pathfp = 'dataFaceHasilKlasifikasi/'+pathlv[i]+'/FP'
    #     os.makedirs(pathfp, exist_ok=True)
    #     for fnameImg in fp:
    #         pathFnameImg = os.path.join(os.getcwd(),'dataFaceLowBrightnes',pathlv[i],'test','man',fnameImg)
    #         img = cv.imread(pathFnameImg)
    #         cv.imwrite(os.path.join(pathfp, fnameImg), img)
    #         # endofr
    #     print(len(os.listdir(pathfp)))
    #
    #     pathtn = 'dataFaceHasilKlasifikasi/'+pathlv[i]+'/TN'
    #     os.makedirs(pathtn, exist_ok=True)
    #     for fnameImg in tn:
    #         pathFnameImg = os.path.join(os.getcwd(),'dataFaceLowBrightnes',pathlv[i],'test','woman',fnameImg)
    #         img = cv.imread(pathFnameImg)
    #         cv.imwrite(os.path.join(pathtn, fnameImg), img)
    #         # endofr
    #     print(len(os.listdir(pathtn)))
    #
    #     pathfn = 'dataFaceHasilKlasifikasi/'+pathlv[i]+'/FN'
    #     os.makedirs(pathfn, exist_ok=True)
    #     for fnameImg in fn:
    #         pathFnameImg = os.path.join(os.getcwd(),'dataFaceLowBrightnes',pathlv[i],'test','woman',fnameImg)
    #         img = cv.imread(pathFnameImg)
    #         cv.imwrite(os.path.join(pathfn, fnameImg), img)
    #         # endofr
    #     print(len(os.listdir(pathfn)))
