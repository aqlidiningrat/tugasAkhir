from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2 as cv
# import requests
import os, random
import pandas as pd

app = Flask(__name__)
root = os.getcwd()
path = os.path.join(root) # mysite

# app.config
app.secret_key = 'TA_miranda_2111017'
app.config['UPLOAD_FOLDER'] = os.path.join(path,'static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #batas ukuran gambar 16mb
ALLOWED_EXTENSIONS = set(['jpeg','jpg','png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def ekstraksiFiturHOG(img, bIn):
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
            # endfor
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    # preview_img
    img_hog = np.array(hists, 'uint8')
    # print('img_hog', img_hog)
    # fiturHOG (transform to hellinger kernel and hist_hog)
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps
    hist, bins = np.histogram(hist, bins=bIn)
    listHOG = [int(hist[v]) for v in range(0, len(hist))]
    return listHOG

def draw_rectangles(face, h_predict, filename, paMax):
    # draw rectangles around the faces
    height, width, _ = face.shape
    start_point, end_point = (0,0), (width, height)
    colorRectandText = (242,51,70) if h_predict=='Laki-Laki' else (132,22,254)

    cv.rectangle(face, start_point, end_point, colorRectandText, 5)
    cv.putText(face, h_predict+' '+str(paMax)+'%', (5,12), cv.FONT_HERSHEY_DUPLEX, 0.3, colorRectandText, 1)
    cv.putText(face, 'TA_miranda_2111017', (5,height-5), cv.FONT_HERSHEY_DUPLEX, 0.3, (255,243,220), 1)
    # save the outputImage
    nameOutputImage = 'hpredict_'+filename
    cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'outputImage/'+nameOutputImage), face)
    return nameOutputImage

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
    print('iCols:', iCols, len(iCols), '\n')
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

def klasifikasiKNN(filename):
    # read the image
    face = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename))
    face_resize = cv.resize(face, (128,128))
    face_gray = cv.cvtColor(face_resize, cv.COLOR_BGR2GRAY)
    if (round(face_gray.mean(), 1) <= 19.2):
        bIn, nk, lv, dismet = 8, 17, '1', 'euclidean'
    elif (round(face_gray.mean(),1) <= 39.0):
        bIn, nk, lv, dismet = 18, 15, '2', 'euclidean'
    else:
        bIn, nk, lv, dismet = 9, 35, '3', 'euclidean'
        # endif
    print('*face_gray.mean():', round(face_gray.mean(), 1), 'bIn:',bIn, 'k:',nk, 'lv:'+lv)
    fiturHOG = ekstraksiFiturHOG(face_resize, bIn=bIn)
    clm = ['fhog'+str(i) for i in range(0, len(fiturHOG))]
    X_test = pd.DataFrame(np.array([fiturHOG]), columns=clm)
    pdtr = 'dataFiturLowBrightnes/low_brightnes_0'+lv+'/'+str(bIn)+'binHOG_train_lv0'+lv+'.xlsx'
    dftrain = pd.read_excel(os.path.join(path, pdtr))
    X_train, y_train = dftrain.drop(['fname','label'], axis=1), dftrain['label']
    X_train, X_test = dataPreprocessing(X_train, X_test)
    print('*',X_train.shape, y_train.shape, X_test.shape)
    y_predict = dataModel(X_train, y_train, X_test, nk, dismet)
    h_predict, paL, paP, paMax = persentase(X_train, y_train, X_test, bIn, nk, dismet, y_predict)
    return h_predict, paL, paP, draw_rectangles(face_resize, h_predict, filename, paMax)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home2():
    # no name file in form html
    if 'file' not in request.files:
        return render_template('index.html')
    file = request.files['file']
    # no file chosen
    if file.filename == '':
        return render_template('index.html')
    # inputImage
    if file and allowed_file(file.filename):
        # save the inputImages
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename))
        h_predict, paL, paP, filenameOutputImage = klasifikasiKNN(filename)
        return render_template('index.html', paL=paL, paP=paP, h_predict=h_predict,
                                filenameOutputImage=filenameOutputImage)

    else: # yangDiupload bukanFileGambar
        return render_template('index.html', filenameOutputImage='yangDiupload bukanFileGambar', filename=file.filename)

@app.route('/displayOutputImage/<filename>')
def displayOutputImage(filename):
    return redirect(url_for('static', filename='outputImage/'+filename), code=301)

@app.route('/downloadOutputImage/<filename>')
def downloadOutputImage(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'],
                        'outputImage/'+filename), as_attachment=True)

@app.route('/hapusImageTA', methods=['GET'])
def hapus():
    for ffolder in ['inputImage','outputImage']:
        for fname in os.listdir(os.path.join(path, 'static', ffolder)):
            os.remove(os.path.join(path, 'static', ffolder, fname))
            # dataBerhasilDihapusBoss..
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
