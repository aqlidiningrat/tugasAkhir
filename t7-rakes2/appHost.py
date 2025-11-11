from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import os, io, base64, random, sqlite3

from datetime import datetime
import pandas as pd
import numpy as np
import cv2 as cv

import requests

app = Flask(__name__)
root = os.getcwd()
path = os.path.join(root, ) #'mysite'
app.secret_key = 'sekar2111004'
app.config['UPLOAD_FOLDER'] = os.path.join(path, 'static')
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024 #limits to 16mb
ALLOWED_EXTENSIONS = set(['jpeg','jpg','png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def insertTB_sekarDB(namaTable, idDB, filename, buffer_file):
    db = sqlite3.connect('sekar.db')
    currentTime = datetime.now()
    strfTime = currentTime.strftime('%d-%B-%Y %H:%M')
    cursor = db.cursor()
    cursor.execute('''create table if not exists '''+namaTable+'''(
                    idImg integer primary key, strfTime text not null,
                    fnameImg text not null, imgBuffer blob)''')
    cursor.execute('''insert into '''+namaTable+'''(
                    idImg, strfTime, fnameImg, imgBuffer) values (?, ?, ?, ?)''',
                    (idDB, strfTime, filename, buffer_file))
    db.commit(), db.close()

def readTB_sekarDB(namaTable, idDB):
    db = sqlite3.connect('sekar.db')
    cursor = db.cursor()
    # readTB_mirandaDB
    cursor.execute('''select * from '''+namaTable+''' where idImg=?''', (idDB,))
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

def contourImage(idDB, img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_resize = cv.resize(img, (400, 200))
    img = (255-img_resize)

    height,width = img.shape
    scale = 4
    heightScale = int(scale*height)
    widthScale = int(scale*width)
    img = cv.resize(img, (widthScale, heightScale))

    _,thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((90,90), np.uint8)
    thresh = cv.dilate(thresh,kernel)
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [contours[0]]
    cv.drawContours(img, contours,-1,(0, 0, 255), 3)

    M = cv.moments(contours[0])
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])

    area = cv.contourArea(contours[0])
    perimeter = cv.arcLength(contours[0], True)
    epsilon = .01*perimeter
    approx = cv.approxPolyDP(contours[0], epsilon, True)
    approx = np.array(approx)
    approx = np.concatenate((approx, approx[:1]), axis=0)

    hull = cv.convexHull(contours[0])
    hull = hull[:,0,:]
    hull = np.concatenate((hull,hull[:1]))

    x,y,w,h = cv.boundingRect(contours[0])
    cv.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
    # imgBoundingRect = img[y:y+h, x:x+w]

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
        # endif
    _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    img = (255-img)
    img_resize = cv.resize(img, (400, 200))
    _,img_stream = cv.imencode('.jpg', img_resize)
    filename = 'TA_Sekar_2111004.jpg'
    insertTB_sekarDB('autentikasiTandaTangan', idDB, filename, img_stream)
    img = readTB_sekarDB('autentikasiTandaTangan', idDB)
    return ketImg, [aspectRatio, extent, solidity, equiDIa, _angle, len(contours[0])]

def api_apps(fiturImg):
    appApiEndPoint = 'https://apisekar2111017.pythonanywhere.com/'
    strFitur = '-'.join([str(i) for i in fiturImg])
    # import requests
    responAPI = requests.get(appApiEndPoint+strFitur)
    if (responAPI.status_code != 200):
        return str(responAPI.status_code)
        # endif
    print(responAPI.status_code, responAPI.text)
    y_predict = responAPI.text.split('_')[0]
    paMax = responAPI.text.split('_')[1]
    paMin =   responAPI.text.split('_')[2]
    return y_predict, paMax, paMin

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tulisTandaTangan/')
def tulisTandaTanganGet():
    namaTable = 'signaturePad'
    idDB = random.randint(000000000,999999999)
    return render_template('signaturePad.html', namaTable=namaTable, idDB=idDB)

@app.route('/tulisTandaTangan/<namaTable>/<idDB>/', methods=['POST'])
def tulisTandaTanganPost(namaTable, idDB):
    data_url = request.json['image_data']
    filename = 'captureSignaturePad.jpg'
    # convert base64 image data to opencv format
    image_data = base64.b64decode(data_url.split(',')[1])
    np_buffer = np.frombuffer(image_data, np.uint8)
    img = cv.imdecode(np_buffer, cv.IMREAD_COLOR)
    insertTB_sekarDB(namaTable, idDB, filename, image_data)
    return jsonify({'message': 'captured Signature successfully..'})

@app.route('/uploadGambarTandaTangan/')
def uploadGambarTandaTanganGet():
    return render_template('uploadGambar.html')

@app.route('/uploadGambarTandaTangan/', methods=['POST'])
def uploadGambarTandaTanganPost():
    file = request.files['imageSignature']
    if file and allowed_file(file.filename):
        buffer_file = file.read()
        np_buffer = np.frombuffer(buffer_file, np.uint8)
        img = cv.imdecode(np_buffer, cv.IMREAD_COLOR)
        img_resize = cv.resize(img, (400, 200))
        _, buffer_file = cv.imencode('.jpg', img_resize)
        namaTable = 'uploadGambar'
        filename = secure_filename(file.filename)
        idDB = random.randint(000000000, 999999999)
        insertTB_sekarDB(namaTable, idDB, filename, buffer_file)
        img = readTB_sekarDB(namaTable, idDB)
        img_decode = decode_utf8(img)
    return render_template('uploadGambar.html', img_decode=img_decode,
                            namaTable=namaTable, idDB=idDB)

@app.route('/klasifikasiKNN/<namaTable>/<idDB>/')
def preprocessingCitra(namaTable, idDB):
    img = readTB_sekarDB(namaTable, idDB)
    if (img == 255).all()or(img == 250).all()or(img == 0).all():
        if (namaTable=='signaturePad'):
            pindahkanURL = 'tulisTandaTanganGet'
        elif (namaTable=='uploadGambar'):
            pindahkanURL = 'uploadGambarTandaTanganGet'
            # endif
        return redirect(url_for(pindahkanURL))
        # endif
    ketImg, fiturImg = contourImage(idDB, img)
    y_predict, paMax, paMin = api_apps(fiturImg)
    return redirect(url_for('autentikasiTandaTangan', namaTable=namaTable, idDB=idDB,
                                ketImg=ketImg, y_predict=y_predict, paMax=paMax, paMin=paMin))

@app.route('/autentikasiTandaTangan/<namaTable>/<idDB>/<ketImg>/<y_predict>/<paMax>/<paMin>/')
def autentikasiTandaTangan(namaTable, idDB, ketImg, y_predict, paMax, paMin):
    if (ketImg == 'True'):
        if (y_predict == '[0]'):
            ttd = 'Bapak Muttaqin, S.T., M.Cs'
        elif (y_predict == '[1]'):
            ttd = 'Ibu Ulfa Nadia, S.Kom., M.IT'
        elif (y_predict == '[2]'):
            ttd = 'Bapak Muttaqin, S.T., M.Cs'
        elif (y_predict == '[3]'):
            ttd = 'Bapak Irwanda Syahputra, S.T., M.Kom'
        elif (y_predict == '[4]'):
            ttd = 'Ibu Dara Thursina, S.Kom., M.Kom'
        else:
            ttd = 'tidakDikenali . . .'
            # endif
        y_predict = 'Terdeteksi tandaTangan: '+ttd
        accuracy = 'autentikasi tandaTangan: asli '+paMax+'%   palsu '+paMin+'%'
        color = 'green' if (float(paMax) >= 49.5) else 'red'
    else:
        y_predict = 'perhatikan KotakPersegi padaGambar tandaTangan..'
        accuracy = 'tandaTangan tidakTerdeksi . . .'
        color = 'blue'
        # endif
    img = readTB_sekarDB('autentikasiTandaTangan', idDB)
    img_decode = decode_utf8(img)
    return render_template('index.html', img_decode=img_decode,
                            y_predict=y_predict, accuracy=accuracy, color=color)

@app.route('/lihatDB/')
def lihatDB():
    return render_template('lihatDB.html')

@app.route('/lihatDB/<namaTable>')
def lihatTB(namaTable):
    db = sqlite3.connect('sekar.db')
    df = pd.read_sql('select * from '+namaTable, db)
    db.close()
    for i in df.index:
        id_img = df[df.axes[1][0]][i]
        strfTime = df[df.axes[1][1]][i]
        fname_img = df[df.axes[1][2]][i]
        img_buffer = df[df.axes[1][-1]][i]
        np_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv.imdecode(np_buffer, cv.IMREAD_COLOR)
        img_decode = decode_utf8(img)
        df.loc[i] = [id_img, strfTime, fname_img, img_decode]
        # endfor
    return render_template('lihatDB.html', df=df, namaTable=namaTable)

if __name__=='__main__':
    app.run(debug=True)
