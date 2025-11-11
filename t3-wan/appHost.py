from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2 as cv
import requests
import os, random
import sqlite3
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
        # endif
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
        # endfor
    return val

def ekstraksiFiturLBP(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(img.shape)
    height, width = img.shape
    img_lbp = np.zeros((height, width), np.uint8)
    npHist = []
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i,j] = lbp_calculated_pixel(img, i, j)
            npHist.append(img_lbp[i,j])
            # endfor
    # featureLBP (transform to hist_lbp)
    hist, bin = np.histogram(npHist, bins=9)
    return hist

def ekstraksiFiturHOG(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(img.shape)
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
    hist /= np.linalg.norm(hist) + eps
    hist, bins = np.histogram(hist, bins=9)
    return hist

def draw_rectangles(face, h_predict, filename, paMax):
    # draw rectangles around the faces
    height, width, _ = face.shape
    start_point, end_point = (0,0), (width, height)
    colorRectandText = (242,51,70) if h_predict=='Laki-Laki' else (132,22,254)

    cv.rectangle(face, start_point, end_point, colorRectandText, 5)
    cv.putText(face, h_predict+' '+str(paMax)+'%', (5,12), cv.FONT_HERSHEY_DUPLEX, 0.3, colorRectandText, 1)
    cv.putText(face, 'TA_wanDInulAqli', (5,height-5), cv.FONT_HERSHEY_DUPLEX, 0.3, (255,243,220), 1)
    # save the outputImage
    nameOutputImage = 'hpredict_'+filename
    cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'outputImage/'+nameOutputImage), face)
    return nameOutputImage

def api_apps(fiturGabungan):
    # copy-paste-url: http://localhost:5000/bIn/nk/metric/t3-wan/namaDS/fiturGabungan
    appApiEndPoint = 'https://wanDinulAqli.pythonanywhere.com'
    bIn, nk, dismet, namaApps, namaDS = str(18), str(11), 'manhattan', 't3-wan', 'citraIdeal_train_18bin_fc.xlsx'
    strFeature = '-'.join([str(i) for i in fiturGabungan])
    urlParams = appApiEndPoint+'/'+bIn+'/'+nk+'/'+dismet+'/'+namaApps+'/'+namaDS+'/'+strFeature
    responAPI = requests.get(urlParams)
    if (responAPI.status_code != 200):
        return 'h_predict', 'paL', 'paP', 'paMax'
        # endif
    dictResponAPI = responAPI.text
    print(type(urlParams), len(urlParams), responAPI.status_code, dictResponAPI)
    h_predict, paL, paP = dictResponAPI.split('_')[0], dictResponAPI.split('_')[1], dictResponAPI.split('_')[2]
    paMax = dictResponAPI.split('_')[3]
    return h_predict, paL, paP, paMax

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

    h_predict, paL, paP, paMax = api_apps(fiturGabungan)
    return h_predict, paL, paP, draw_rectangles(face, h_predict, filename, paMax)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home2():
    # no name file in form html
    if 'file' not in request.files:
        return render_template('index.html') # no name file in form html
    file = request.files['file']
    # no file chosen
    if file.filename == '':
        return render_template('index.html') # no file chosen
    # inputImage
    if file and allowed_file(file.filename):
        # save the inputImages
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename))
        h_predict, paL, paP, filenameOutputImage = klasifikasiKNN(filename)
        return render_template('index.html', h_predict=h_predict, paL=paL, paP=paP,
                                filenameOutputImage=filenameOutputImage)

    else: # yangDiupload bukanFileGambar
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

    # wan.db
    db = sqlite3.connect(os.path.join(path, 'wan.db'))
    cursor = db.cursor()
    # createTable
    cursor.execute('''create table if not exists wanTable(id integer primary key autoincrement,
                    atasaNama text not null, bagusAplikasinya text not null,
                    apaYangKurang text not null, rating integer not null)''')
    dataDB = (nma, gmn, krg, rtg)
    cursor.execute('''insert into wanTable(atasaNama, bagusAplikasinya,
                            apaYangKurang, rating) values(?,?,?,?)''', dataDB)
    db.commit()
    db.close()
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

@app.route('/lihatPesanDB', methods=['GET'])
def dbPesan():
    db = sqlite3.connect(os.path.join(path, 'wan.db'))
    df = pd.read_sql("select * from wanTable", db)
    return df.to_html()

if __name__ == '__main__':
    app.run(debug=True)
