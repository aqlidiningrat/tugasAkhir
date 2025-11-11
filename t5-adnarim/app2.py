from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import sklearn.preprocessing as pp
import sklearn.neighbors as knn
import pandas as pd
import numpy as np
import cv2 as cv
import sqlite3
import random
import base64
import os, io

app = Flask(__name__)
path = os.path.join(os.getcwd(), ) # mysite
app.secret_key = 'TA_miranda_2111017'
app.config['UPLOAD_FOLDER'] = os.path.join(path, 'static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # limits to 16 megabytes

ALLOWED_EXTENSIONS = set(['jpeg','jpg','png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def insertTB_mirandaDB(namaTable, filename, buffer_file):
    db = sqlite3.connect('miranda.db')
    cursor = db.cursor()
    cursor.execute('''create table if not exists '''+namaTable+'''(
                    id integer primary key, fname text not null, img blob)''')
    idDB = random.randint(000000000,999999999)
    cursor.execute('''insert into '''+namaTable+'''(
                    id, fname, img) values (?, ?, ?)''', (idDB, filename, buffer_file))
    db.commit(), db.close()
    return idDB

def insertTB_mirandaDB_withIndex(namaTable, idDB, filename, buffer_file):
    db = sqlite3.connect('miranda.db')
    cursor = db.cursor()
    cursor.execute('''create table if not exists '''+namaTable+'''(
                    id integer primary key, fname text not null, img blob)''')
    cursor.execute('''insert into '''+namaTable+'''(
                    id, fname, img) values (?, ?, ?)''', (idDB, filename, buffer_file))
    db.commit(), db.close()

def readTB_mirandaDB(namaTable, idDB):
    db = sqlite3.connect('miranda.db')
    cursor = db.cursor()
    # readTB_mirandaDB
    cursor.execute('''select * from '''+namaTable+''' where id=?''', (idDB,))
    # retrivied_bytes = cursor.fetchone()[-1]
    retrivied_bytes = cursor.fetchall()
    img_buffer = retrivied_bytes[0][-1]
    db.close()
    np_buffer = np.frombuffer(img_buffer, np.uint8) # create nparray from buffer bytes
    img = cv.imdecode(np_buffer, cv.IMREAD_COLOR) # read buffer bytes
    return img

def decode_utf8(img):
    _, img_stream = cv.imencode('.jpg', img) # create stream img
    encode_img = base64.b64encode(img_stream) # encode b64 from stream img
    img_decode = encode_img.decode('utf8') # decode utf8 from b64
    return img_decode

def low_brightnes(img, alpha):
    if (alpha == float('0.0')):
        return img
    img_normal = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    img_alpha = img_normal * float(alpha)
    img_low_brightnes = img_alpha.astype(np.uint8)
    img_low_brightnes[img_low_brightnes < 0] = 0
    img_low_brightnes[img_low_brightnes > 255] = 255
    return img_low_brightnes

def ekstraksiFiturHOG(img, bIn):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_resize = cv.resize(img_gray, (128,128))
    gx = cv.Sobel(img_resize, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img_resize, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang / (2*np.pi))
    bin_cells = []
    mag_cells = []
    cellx, celly = 8,8
    for i in range(0, int(img_resize.shape[0] / celly)):
        for j in range(0, int(img_resize.shape[1] / cellx)):
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

def dataPreprocessing(Xtrain, Xtest):
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(Xtrain)
    X_train = scl.transform(Xtrain)
    X_test = scl.transform(Xtest)
    return X_train, X_test

def dataModel(X_train, y_train, X_test, nk, dismet):
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
    print('*y_predict:',y_predict, '*persentaseLaki:',perhL, '*persentasePerempuan:',perhP)
    return y_predict, perhL, perhP, perSepuluhMax

def persentase3(pa, y_predict):
    # print('*pa:',pa, '*len(pa):',len(pa))
    perKlas = len(pa)
    seperSepuluhKlas = round(float(100/perKlas), 1)
    seperSeratusKlas = round(float(seperSepuluhKlas*perKlas), 1)
    # seperSeratusKlas -= 1
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

def draw_rectangles(face_lb, h_predict, paMax):
    # draw rectangles around the faces
    height, width, _ = face_lb.shape
    start_point, end_point = (0,0), (width, height)
    colorRectandText = (242,51,70) if h_predict=='Laki-Laki' else (132,22,254)
    # cv.rectangle(face_lb, start_point, end_point, colorRectandText, 5)
    cv.putText(face_lb, h_predict+' '+str(paMax)+'%', (5,10), cv.FONT_HERSHEY_DUPLEX, 0.3, colorRectandText, 1)
    cv.putText(face_lb, 'TA_miranda_2111017', (5,height-5), cv.FONT_HERSHEY_DUPLEX, 0.2, (255,243,220), 1)
    _, img_stream = cv.imencode('.jpg', face_lb) # create stream img
    filename = h_predict+'_'+str(paMax)+'%'
    idDBEnd = insertTB_mirandaDB('hpredictImg', filename, img_stream)
    img = readTB_mirandaDB('hpredictImg', idDBEnd)
    return img, idDBEnd

def klasifikasiKNN(face_lb, lv):
    bIn = 8 if(lv=='0.1') else 18 if(lv=='0.2') else 9
    nk = 17 if(lv=='0.1') else 15 if(lv=='0.2') else 35
    lv = '01' if(lv=='0.1') else '02' if(lv=='0.2') else '03'
    dismet = 'euclidean'
    print('*level_low_brightnes:'+str(lv), '*bIn:'+str(bIn), '*nk:'+str(nk), '*dismet:'+str(dismet))

    # pada pengaplikasiannya menggunakan ekstraksiFiturHOG, karena HOG menghasilkan akurasi tertinggi
    # yaitu sebesar 0.83% dibanding SLIC dan PCA
    fiturHOG = ekstraksiFiturHOG(face_lb, bIn=bIn)
    clm = ['fhog'+str(i) for i in range(0, len(fiturHOG))]
    Xtest = pd.DataFrame(np.array([fiturHOG]), columns=clm)
    pdtr = 'dataFiturLowBrightnes/low_brightnes_'+lv+'/'+str(bIn)+'binHOG_train_lv'+lv+'.xlsx'
    dftrain = pd.read_excel(os.path.join(path, pdtr))
    Xtrain, y_train = dftrain.drop(['fname','label'], axis=1), dftrain['label']
    X_train, X_test = dataPreprocessing(Xtrain, Xtest)
    # print('*',X_train.shape, y_train.shape, X_test.shape)
    y_predict = dataModel(X_train, y_train, X_test, nk, dismet)
    h_predict, paL, paP, paMax = persentase(X_train, y_train, X_test, bIn, nk, dismet, y_predict)
    img_hpredict, idDBEnd = draw_rectangles(face_lb, h_predict, paMax)
    return h_predict, paL, paP, img_hpredict, idDBEnd

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ambilGambarWebcam')
def ambilGambarWebcam():
    idWebcam = random.randint(000000000,999999999)
    return render_template('index.html', render='ambilGambarWebcam', idWebcam=idWebcam)

@app.route('/tangkapLayar/<idWebcam>', methods=['POST'])
def capture(idWebcam):
    file = request.files['imageBlob']
    filename = secure_filename(file.filename)
    buffer_file = file.read()
    insertTB_mirandaDB_withIndex('ambilGambarWebcam', idWebcam, filename, buffer_file)
    return 'Gambar berhasil disimpan didatabase...'

@app.route('/pilihGambar')
def pilihGambar():
    return render_template('index.html', render='pilihGambar')

@app.route('/pilihGambar', methods=['POST'])
def pilihGambar2():
    file = request.files['pilihGambar']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        buffer_file = file.read()
        idDB = insertTB_mirandaDB('pilihGambar', filename, buffer_file)
        return redirect(url_for('potongGambar', dari='pilihGambar', idDB=idDB))
    else: # yangDiupload bukanFileGambar
        return render_template('index.html', render='yangDiupload bukanFileGambar', filename=file.filename)

@app.route('/<dari>/<idDB>/potongGambar')
def potongGambar(dari, idDB):
    img = readTB_mirandaDB(dari, idDB)
    img_decode = decode_utf8(img)
    idDBNew = random.randint(000000000,999999999)
    return render_template('index-crop.html', render='potongGambar', dari=dari, idDB=idDB, idDBNew=idDBNew,
                            img_decode=img_decode)

@app.route('/potongGambar/<idDBNew>', methods=['POST'])
def potongGambar2(idDBNew):
    file = request.files['croppedImage']
    filename = secure_filename(file.filename)
    buffer_file = file.read()
    insertTB_mirandaDB_withIndex('potongGambar', idDBNew, filename, buffer_file)
    return {'message':'Save cropped_image.jpg successfully', 'filename':filename}

@app.route('/<dari>/<idDB>/potongGambar/<idDBNew>/aturKecerahanGambar')
def aturKecerahanGambar(dari, idDB, idDBNew):
    img = readTB_mirandaDB('potongGambar', idDBNew)
    img_decode = decode_utf8(img)
    print('ukuranPotonganGambar:',img.shape,'.pixel')
    return render_template('index.html', render='aturKecerahanGambar', dari=dari, idDB=idDB, idDBNew=idDBNew,
                            lv='Normal', img_decode=img_decode)

@app.route('/<dari>/<idDB>/potongGambar/<idDBNew>/aturKecerahanGambar/<lv>')
def aturKecerahanGambar_lv(dari, idDB, idDBNew, lv):
    img = readTB_mirandaDB('potongGambar', idDBNew)
    img_low_brightnes = low_brightnes(img, float(lv))
    img_decode = decode_utf8(img_low_brightnes)
    return render_template('index.html', render='aturKecerahanGambar', dari=dari, idDB=idDB, idDBNew=idDBNew,
                            lv=lv, img_decode=img_decode)

@app.route('/<dari>/<idDB>/potongGambar/<idDBNew>/aturKecerahanGambar/<lv>/mulaiKlasifikasi')
def mulaiKlasifikasi(dari, idDB, idDBNew, lv):
    img = readTB_mirandaDB('potongGambar', idDBNew)
    img_low_brightnes = low_brightnes(img, float(lv)) if (lv != 'Normal') else img
    h_predict, paL, paP, img_hpredict, idDBEnd = klasifikasiKNN(img_low_brightnes, lv)
    img_decode = decode_utf8(img_hpredict)
    return render_template('index.html', render='mulaiKlasifikasi', dari=dari, idDB=idDB, idDBNew=idDBNew,
                            idDBEnd=idDBEnd, lv=lv, h_predict=h_predict, paL=paL, paP=paP, img_decode=img_decode)

@app.route('/downloadGambarHasil/<lv>/<idDBEnd>')
def downloadGambarHasil(lv, idDBEnd):
    img = readTB_mirandaDB('hpredictImg', idDBEnd)
    _, img_stream = cv.imencode('.jpg',img) #create img buffer
    byte_io = io.BytesIO(img_stream) #create BytesIO from img_stream (img_buffer)
    byte_io.seek(0) #read to first buffer file
    return send_file(byte_io, as_attachment=True, download_name='TA_miranda_2111017_lowBrightnes'+lv+'.jpg', mimetype='image/png')

@app.route('/lihatData/<namaTable>')
def lihatData(namaTable):
    db = sqlite3.connect('miranda.db')
    df = pd.read_sql('select * from '+namaTable, db)
    db.close()
    print(len(df))
    return df[['id','fname']].to_html()

if __name__ == '__main__':
    app.run(debug=True)
