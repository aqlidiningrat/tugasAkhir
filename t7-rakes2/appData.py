def contours(pathImg, img):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,2))
    # plt.figtext(0.5, 0.08, 'Tugas Akhir Sekar Lucia Ningrum 2111004',
    #                 ha='center', va='center', fontsize='small', style='oblique')
    plt.suptitle('fitur Contour Image', fontsize='small', weight='extra bold')

    # plt.subplot(2,3,1)
    # plt.title('imgInvert', fontsize=7, style='oblique')
    # plt.axis('off')
    # plt.imshow(img, cmap='gray')

    img_resize = cv.resize(img, (400,200))
    img = (255-img_resize)
    height,width = img.shape
    scale = 4
    heightScale = int(scale*height)
    widthScale = int(scale*width)
    img = cv.resize(img, (widthScale, heightScale))

    _,thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # plt.subplot(2,3,2)
    # plt.title('imgThreshold', fontsize=7, style='oblique')
    # plt.axis('off')
    # plt.imshow(thresh, cmap='binary')

    kernel = np.ones((90,90), np.uint8)
    thresh = cv.dilate(thresh,kernel)
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [contours[0]]
    # cv.drawContours(img, contours,-1,(0, 0, 255), 3)
    # plt.subplot(2,3,3)
    # plt.title('imgContour', fontsize=7, style='oblique')
    # plt.axis('off')
    # plt.imshow(img, cmap='binary')

    M = cv.moments(contours[0])
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])
    # plt.subplot(2,3,4)
    plt.subplot(1,2,1)
    plt.title('solidity, equiDIa, jumlahContour', fontsize=7, style='oblique')
    # plt.axis('off')
    plt.imshow(img, cmap='binary')
    plt.plot(Cx, Cy, 'r*')

    area = cv.contourArea(contours[0])
    perimeter = cv.arcLength(contours[0], True)
    epsilon = .01*perimeter
    approx = cv.approxPolyDP(contours[0], epsilon, True)
    approx = np.array(approx)
    approx = np.concatenate((approx, approx[:1]), axis=0)
    plt.plot(approx[:,0,0], approx[:,0,1])

    hull = cv.convexHull(contours[0])
    hull = hull[:,0,:]
    hull = np.concatenate((hull,hull[:1]))
    # plt.subplot(2,3,5)
    plt.subplot(1,2,2)
    plt.title('aspectRatio, extent, angle', fontsize=7, style='oblique')
    plt.imshow(img, cmap='binary')
    # plt.axis('off')
    plt.plot(hull[:,0], hull[:,1], 'r-')

    x,y,w,h = cv.boundingRect(contours[0])
    cv.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
    # imgBoundingRect = img[y:y+h, x:x+w]
    # plt.subplot(2,3,6)
    # plt.title('imgContourBoundingRect', fontsize=7, style='oblique')
    # plt.imshow(img, cmap='binary')
    # plt.axis('off')

    aspectRatio = round(w/h, 2)
    extent = round(w*h/area, 2)
    solidity = round(area/cv.contourArea(hull), 2)
    equiDIa = round(int(np.sqrt(4*area/np.pi)), 2)

    if (len(contours[0]) <= 100):
        _angle = round(len(contours[0])/100, 2)
        ketImg = False
    else:
        _,_,_angle = cv.fitEllipse(contours[0])
        _angle = round(_angle, 2)
        ketImg = True

    plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(),'static','plt_figure.jpg'))
    plt.show()

    # _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # img = (255-img)
    # cv.imshow('imgContour', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    print('*pathImg:', pathImg, ketImg)
    # {'aspectRatio':'hubunganProporsionalAntaraLebarDanTinggiGambar',
    #     'extent':'jangkauanLuasAtauSejauhManaSesuatuMeluas',
    #     'solidity':'jangkauanTepi', 'equiDIa':'istilahMenggambarkanJarakYangSamaPadaSuatuTitikData',
    #     '_angle':'sudut', 'len_contours':'jumlahContours'}
    df = pd.DataFrame({'aspectRatio':[aspectRatio], 'extent':[extent], 'solidity':[solidity],
                        'equiDIa':[equiDIa], 'angle':[_angle], 'len_contours':[len(contours[0])]})
    print(df.to_string(index=False))
    return [aspectRatio, extent, solidity, equiDIa, _angle, len(contours[0])]

def fiturImg():
    pathImg = os.path.join(os.getcwd(), 'img')
    listFitur = []
    for label in os.listdir(pathImg):
        pathImgLabel = os.path.join(pathImg, label)
        for fnameImg in os.listdir(pathImgLabel):
            filename = os.path.join(pathImgLabel, fnameImg)
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            # img_resize = cv.resize(img, (400, 200))
            # cv.imwrite(filename, img_resize)
            # continue
            fiturImg = contours(filename, img)
            fiturImg.insert(0, fnameImg)
            fiturImg.append(int(label))
            listFitur.append(fiturImg)
            # endfor
        # endofor
    df = pd.DataFrame(listFitur, columns=['filename','aspectRatio','extent','solidity',
                                            'equiDIa','angle','len_contours','label'])
    df.to_csv('fiturImg.csv', index=False)
    data_csv = pd.read_csv('fiturImg.csv')
    print(data_csv)

def dataPreprocessing(Xtrain, Xtest):
    import sklearn.preprocessing as pp
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(Xtrain)
    X_train = scl.transform(Xtrain)
    X_test = scl.transform(Xtest)
    return X_train, X_test

def reportKlasifikasi(y_test, y_predict):
    import sklearn.metrics as met
    accuracy = met.accuracy_score(y_test, y_predict)
    print(accuracy)
    confusionmatrix = met.confusion_matrix(y_test, y_predict)
    print(confusionmatrix)
    # precision = met.precision_score(y_test, y_predict)
    # sensitifity = met.recall_score(y_test, y_predict)
    report = met.classification_report(y_test, y_predict)
    print(report)

def modelClassifier(k, X_train, X_test, y_train, y_test):
    import sklearn.neighbors as knn
    model = knn.KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = model.score(X_test, y_test)
    return y_predict, score

def klasifikasiKNN():
    df = pd.read_csv(os.path.join(os.getcwd(), 'fiturImg.csv'))
    X, y = df.drop(['filename','label'], axis=1), df['label']
    import sklearn.model_selection as ms
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.1, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    ax = X_test.index
    X_train, X_test = dataPreprocessing(X_train, X_test)

    # multiple k
    for k in list(filter(lambda x: x%2==0, [x for x in range(10, 40)])):
        print('=== k:',k)
        y_predict, score = modelClassifier(k, X_train, X_test, y_train, y_test)
        print(score)
        # endfor
    # exit()

    y_predict, score = modelClassifier(12, X_train, X_test, y_train, y_test)
    show_df_test = df.loc[ax]
    show_df_test['y_predict'] = y_predict
    show_df_miss = show_df_test[show_df_test['label'] != show_df_test['y_predict']]
    print(show_df_miss)
    reportKlasifikasi(y_test, y_predict)

def persentaseAkurasi(Xtrain, Xtest, y_train, y_test):
    print('====================== persentaseAkurasi')
    print(Xtrain.columns)
    a,b = [0,1,4, 5,2,3],[2,3,5, 3,4,1]
    klm1 = [Xtrain.axes[1][i] for i in a]
    klm2 = [Xtrain.axes[1][i] for i in b]
    for k in list(filter(lambda x: x%2==0, [x for x in range(2,5)])):
        print('\n ===== k:',k)
        pa = []
        for i in range(0, len(Xtrain.axes[1])):
            X_train = Xtrain[[klm1[i], klm2[i]]]
            X_test = Xtest[[klm1[i], klm2[i]]]
            y_predict, score = modelClassifier(k, X_train, X_test, y_train, y_test)
            print(X_train.axes[1], X_test.axes[1], y_train.shape)
            print(y_test, y_predict, (y_test==y_predict))
            pa.append((y_test==y_predict))
            # endfor
        paTrue = list(filter(lambda x:x==True, [x for x in pa]))
        paFalse = list(filter(lambda x:x==False, [x for x in pa]))
        paMax = 16.5 * len(paTrue)
        paMin = 16.5 * len(paFalse)
        paMax = (paMax + 0.1) if (paMax == 0.0) else paMax
        paMin = (paMin - 0.1) if (paMin == paMax) else paMin
        print(pa, len(paTrue), len(paFalse))
        print(paMax, paMin)

def klasifikasiKNN_fitur(fiturImg, ytest):
    df = pd.read_csv(os.path.join(os.getcwd(), 'fiturImg.csv'))
    Xtrain, y_train = df.drop(['filename','label'], axis=1), df['label']
    Xtest, y_test = pd.DataFrame(np.array([fiturImg]), columns=Xtrain.columns), pd.Series(ytest)
    print(Xtrain.shape, Xtest.shape, y_train.shape, y_test.shape)
    X_train, X_test = dataPreprocessing(Xtrain, Xtest)
    y_predict, score = modelClassifier(2, X_train, X_test, y_train, y_test)
    print('*y_predict:',y_predict, '*ytest:',ytest, '*accuracy:',score)
    # multiple k
    for k in list(filter(lambda x: x%2==0, [x for x in range(2, 5)])):
        print('=== k:',k)
        y_predict, score = modelClassifier(2, X_train, X_test, y_train, y_test)
        print(score)
    persentaseAkurasi(Xtrain, Xtest, y_train, y_predict)
    return y_predict, round(score, 2)

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import cv2 as cv
    import os

    # === testTTD bukdiah2, pak wanda4, bukdara5
    # root = os.getcwd()
    # pathImg = os.path.join(root, 'img', '0')
    # fnameImg = os.listdir(pathImg)[0]
    # filename = os.path.join(pathImg, fnameImg)

    # 'ttdPakMuttaqin', ttdBukUlfa2, ttdBukUlfa, ttdBukUlfa_noise, ttdWan, t1, t2, t3, t4
    fnameImg = 'ttdSekar.PNG'
    filename = os.path.join('d:/', fnameImg)
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    fiturImg = contours(fnameImg, img)
    # y_predict, accuracy = klasifikasiKNN_fitur(fiturImg, [2])

    # fiturImg()
    # klasifikasiKNN()
