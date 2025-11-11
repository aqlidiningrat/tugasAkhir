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
app.secret_key = 'TA_wanDinulAqli_2011042'
app.config['UPLOAD_FOLDER'] = os.path.join(path,'static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #batas ukuran gambar 16mb
ALLOWED_EXTENSIONS = set(['jpeg','jpg','png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def get_pixel(img, center, x, y):
    new_value = 0
    # if local neighborhod pixel, value is greater than or equal,
    # to center pixel values then, set it to 1
    try:
        if img[x][y] >= center:
            new_value = 1
    # Exception is required when, neighborhod value of a center,
    # pixel value is null i.e. values, present at boundaries
    except Exception as e:
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

def ekstraksiFiturLBP(img):
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

    # featureLBP (transform to hist_lbp)
    hist, bin = np.histogram(npHist, bins=9)
    print('lenHist:', len(npHist))
    print('hist_lbp:',hist)
    return hist

def ekstraksiFiturHOG(img):
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

    # fiturHOG (transform to hellinger kernel and hist_hog)
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    print('lenHist:', len(hist))
    hist /= np.linalg.norm(hist) + eps
    hist, bins = np.histogram(hist, bins=9)
    print('hist_hog:', hist)
    return hist

def draw_rectangles(face, h_predict, filename):
    # draw rectangles around the faces
    height, width, _ = face.shape
    start_point, end_point = (0,0), (width, height)

    colorRectandText = (242,51,70) if h_predict=='Laki-laki' else (132,22,254)
    cv.rectangle(face, start_point, end_point, colorRectandText, 5)
    cv.putText(face, h_predict, (5,12), cv.FONT_HERSHEY_DUPLEX, 0.3, colorRectandText, 1)
    cv.putText(face, 'TA_wanDInulAqli', (5,height-5), cv.FONT_HERSHEY_DUPLEX, 0.3, (255,243,220), 1)

    # save the outputImage
    nameOutputImage = 'wajah_'+filename
    cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'outputImage/'+nameOutputImage), face)
    return nameOutputImage

def dataPreprocessing(X_train, fiturGabungan):
    import sklearn.preprocessing as pp
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(fiturGabungan)
    return X_train, X_test

def dataModel(X_train, y_train, X_test):
    import sklearn.neighbors as knn
    # citraIdeal-31-'manhattan' || citraTidakIdeal-21-'minkowski' || citraGabungan-31-'minkowski'
    model = knn.KNeighborsClassifier(n_neighbors=11, metric='manhattan')
    model.fit(X_train, y_train)
    h_predict = model.predict(X_test)
    h_predict = 'Laki-laki' if h_predict == [0] else 'Perempuan'
    print(h_predict)
    return h_predict

def persentase(X_train, y_train, X_test):
    pa = []
    clm1, clm2 = [0,2,4,6,8,10,12,14,16], [1,3,5,7,9,11,13,15,17]
    for i in range(0,9):
        dftrain = X_train[[X_train.columns[clm1[i]], X_train.columns[clm2[i]]]]
        dftest = X_test[[X_test.columns[clm1[i]], X_test.columns[clm2[i]]]]
        Xtrain, Xtest = dataPreprocessing(dftrain, dftest)
        h_predict = dataModel(Xtrain, y_train, Xtest)
        pa.append(h_predict)

    pa = pd.Series(pa)
    paL, paP = [],[]
    for value in pa:
        if (value=='Laki-laki'):
            paL.append(value)
        else:
            paP.append(value)

    print('persentaseLaki-laki:',float(len(paL)*11.1), 'persentasePerempuan:',float(len(paP)*11.1))
    h_predict = 'Laki-laki' if (float(len(paL)*11.1) > float(len(paP)*11.1)) else 'Perempuan'
    return h_predict, round(float(len(paL)*11.1), 2), round(float(len(paP)*11.1), 2)

def klasifikasiKNN(filename):
    # read the image
    face = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename))
    face = cv.resize(face, (128,128))
    fiturLBP = ekstraksiFiturLBP(face)
    fiturHOG = ekstraksiFiturHOG(face)
    fiturGabungan = np.array([fiturLBP[0], fiturLBP[1], fiturLBP[2], fiturLBP[3], fiturLBP[4],
                                fiturLBP[5], fiturLBP[6], fiturLBP[7], fiturLBP[8],
                                fiturHOG[0], fiturHOG[1], fiturHOG[2], fiturHOG[3], fiturHOG[4],
                                fiturHOG[5], fiturHOG[6], fiturHOG[7], fiturHOG[8]])

    clm = ['fc'+str(i) for i in range(0, (len(fiturLBP)+len(fiturHOG)))]
    fiturGabungan = pd.DataFrame(np.array([fiturGabungan]), columns=clm)
    print('==>fiturGabungan:')
    print(fiturGabungan)

    dftrain = pd.read_excel(os.path.join(path, 'dataFitur/citraIdeal/fc/citraIdeal_train_18bin_fc.xlsx'))
    X_train = dftrain.drop(['fname','label'], axis=1)
    y_train = dftrain['label']
    h_predict, paL, paP = persentase(X_train, y_train, fiturGabungan)
    X_train, X_test = dataPreprocessing(X_train, fiturGabungan)
    # h_predict = dataModel(X_train, y_train, X_test)
    return paL, paP, h_predict, draw_rectangles(face, h_predict, filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home2():
    if 'file' not in request.files:
        return render_template('index.html') # no name file in form html

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html') # no file chosen

    if file and allowed_file(file.filename):
        print(file)
        exit()
        # save the inputImages
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename))
        paL, paP, h_predict, filenameOutputImage = klasifikasiKNN(filename)
        return render_template('index.html', paL=paL, paP=paP, h_predict=h_predict,
                                filenameOutputImage=filenameOutputImage)
    else:
        return render_template('index.html', filenameOutputImage='yangDiupload bukanFileGambar',
                                filename=file.filename)

@app.route('/displayOutputImage/<filename>', methods=['GET'])
def displayOutputImage(filename):
    return redirect(url_for('static', filename='outputImage/'+filename), code=301)

@app.route('/downloadOutputImage/<filename>', methods=['GET'])
def downloadOutputImage(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'outputImage/'+filename), as_attachment=True)

# kritik dan saran
@app.route('/<pesan>', methods=['GET'])
def kritik(pesan):
    return render_template('index_saran.html')

@app.route('/<pesan>', methods=['POST'])
def saran(pesan):
    nma, gmn = request.form.get('nma'), request.form.get('gmn')
    krg, rtg = request.form.get('krg'), request.form.get('rtg')
    # try:
    #     file_txt = open(os.path.join(app.config['UPLOAD_FOLDER'],'pesan', nma+'.txt'), 'a')
    # except Exception as e:
    #     file_txt = open(os.path.join(app.config['UPLOAD_FOLDER'],'pesan', nma+'.txt'), 'a')
    # # create file.txt
    # klmtnya1, klmtnya2 = ' *atasaNama: ', '\n *gimanaAplikasinyaBagus: '
    # klmtnya3, klmtnya4 = '\n *apaYangKurangDariAplikasinya: ', '\n *berikanRating1-10: '
    # file_txt.write(klmtnya1+nma+klmtnya2+gmn+klmtnya3+krg+klmtnya4+rtg)
    # file_txt.close()
    return render_template('index_saran.html', udah=nma)

@app.route('/hapusImageTA', methods=['GET'])
def hapus():
    for ffolder in ['inputImage','outputImage']:
        for fname in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], ffolder)):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], ffolder, fname))
            # imageBerhasilDihapusBoss..
    return redirect(url_for('home'))

@app.route('/hapusPesanTA', methods=['GET'])
def hapus2():
    for fname in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'pesan')):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'pesan', fname))
        # pesanBerhasilDihapusBoss..
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
