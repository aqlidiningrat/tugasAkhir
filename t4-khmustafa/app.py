from flask import Flask, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
import random
import os

app = Flask(__name__, template_folder='templates')
# load the haarcascade xml file
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# app.config
root = os.getcwd()
path = os.path.join(root) #'mysite'
app.secret_key = 'khaidirMustafa_2011028'
app.config['UPLOAD_FOLDER'] = os.path.join(path,'static/inputImage/')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #batas ukuran gambar 16mb
ALLOWED_EXTENSIONS = set(['jpeg','jpg','png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def lainnya():
    recomended, jenisWajah = [], ['triangleFace', 'squareFace', 'roundFace'] # 'ovalFace'
    for pathJenisWajah in jenisWajah:
        rekomendasiWajah = os.listdir(os.path.join(path, 'static', pathJenisWajah))
        rekomendasi = rekomendasiWajah[random.randint(0,6)]
        recomended.append(pathJenisWajah+'/'+rekomendasi)

    print(recomended, len(recomended))
    return recomended

def f_jenisWajah(aspectRatio, extent, solidity, equiDIa, contours):
    if (extent > 2116) and (equiDIa > 470)  and (aspectRatio == 1) and (solidity == 1) and (len(contours) == 1):
        return 'squareFace'
    elif (extent > 1600) and (equiDIa > 449)  and (aspectRatio == 1) and (solidity == 1) and (len(contours) == 1):
        return 'triangleFace'
    elif (extent > 1697) and (equiDIa > 461) and (aspectRatio == 1) and (solidity == 1) and (len(contours) == 4):
        return 'roundFace'
    elif (extent > 2621) and (equiDIa > 576) and (aspectRatio == 1) and (solidity == 1) and (len(contours) == 1):
        return 'roundFace'
    else:
        # no frontal_face
        jenisWajah = ['ovalFace','triangleFace', 'squareFace', 'roundFace']
        return jenisWajah[random.randint(0,3)]

def morphologicalImage(face):
    # img_gray
    img = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    print(img.shape)
    height,width = img.shape
    scale = 4
    heightScale = int(scale*height)
    widthScale = int(scale*width)
    img = cv.resize(img, (widthScale, heightScale))
    # img_thresh
    _,thresh = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((70,70), np.uint8)
    thresh = cv.dilate(thresh,kernel)
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print('len contours:',len(contours),'type contours:',type(contours))
    # img_contours
    contours = contours
    cv.drawContours(img,contours,-1,(0,0,255),3)
    # img_moments
    M = cv.moments(contours[0])
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])
    # img_contourArea
    area = cv.contourArea(contours[0])
    perimeter = cv.arcLength(contours[0], True)
    epsilon = .01*perimeter
    approx = cv.approxPolyDP(contours[0], epsilon, True)
    approx = np.array(approx)
    approx = np.concatenate((approx, approx[:1]), axis=0)
    # img_convexHull
    hull = cv.convexHull(contours[0])
    hull = hull[:,0,:]
    hull = np.concatenate((hull,hull[:1]))
    # imgBoundingRect
    x,y,w,h = cv.boundingRect(contours[0])
    # landMark_wajah
    aspectRatio = round(w/h)
    extent = round(w*h)
    solidity = round(area/cv.contourArea(hull))
    equiDIa = round(np.sqrt(4*area/np.pi))
    print('aspectRatio:',aspectRatio)
    print('extent:',extent)
    print('solidity:',solidity)
    print('equiDIa:',equiDIa)
    print('contours:', len(contours))
    # run function jenisWajah
    jenisWajah = f_jenisWajah(aspectRatio, extent, solidity, equiDIa, contours)
    pathJenisWajah = os.listdir(os.path.join(path,'static',jenisWajah))
    rekomendasiWajah = pathJenisWajah[random.randint(0,6)]
    print('jenisWajah:',jenisWajah)
    print('rekomendasiWajah:',rekomendasiWajah)
    return jenisWajah, rekomendasiWajah

def deteksiWajah(filename):
    # read the image
    img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        # save the outputImage
        nameOutputImage = '0wajah_'+filename
        cv.imwrite(os.path.join(path,'static/outputImage/', nameOutputImage), img)
        return len(faces), [0], 'wajahTidakTerdeteksi', nameOutputImage, 'cobaFileGambarLain'

    m = 0
    for (x,y,w,h) in faces:
        m = m+1
        face = img[y:y+h, x:x+w]
        # nameFaces
        nameFaces = 'wajah'+str(m)+'_'+filename
        cv.imwrite(os.path.join(root, 'static/faces/', nameFaces), face)

    m, ind, nfc, jfc, rfc = 0, [], [], [], []
    for (x,y,w,h) in faces:
        m = m+1
        print('-------------------wajah',m,'dari',len(faces),'wajahTerdeteksi')
        face = img[y:y+h, x:x+w]
        # run fuction morphologicalImage
        jenis, rekom = morphologicalImage(face)
        # draw rectangles around the faces
        colorRectandText = (205,208,206)
        cv.rectangle(img, (x,y),(x+w, y+h), colorRectandText, 3)
        cv.putText(img, jenis, (x+5,y-5), cv.FONT_HERSHEY_DUPLEX, 0.5, colorRectandText, 2)
        cv.putText(img, 'wajah '+str(m), (x+5,y+h-5), cv.FONT_HERSHEY_DUPLEX, 0.5, colorRectandText, 2)
        # create jenisWajah
        ind.append(int(m-1)), nfc.append(str(m)), jfc.append(jenis), rfc.append(rekom.split('.')[0])

    # save the outputImage
    nameOutputImage = str(m)+'wajah_'+filename
    cv.imwrite(os.path.join(path,'static/outputImage/', nameOutputImage), img)
    # jenis_rekom_wajah
    jenis_rekom_wajah = np.array([nfc, jfc, rfc])
    print(jenis_rekom_wajah)
    print(len(faces), ind)
    return len(faces), ind, jenis_rekom_wajah, nameOutputImage, filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def khaidirMustafa_2011028():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # function deteksiWajah
        jlhWajah, ind, jenis_rekom_wajah, nameOutputImage, filename = deteksiWajah(filename)
        if (jlhWajah == 0):
            return render_template('index.html', jlhWajah=str(jlhWajah), jenis_rekom_wajah=jenis_rekom_wajah, filenameOutputImage=nameOutputImage)

        recomended = lainnya()
        return render_template('index.html', jlhWajah=jlhWajah, ind=ind, jenis_rekom_wajah=jenis_rekom_wajah, filenameOutputImage=nameOutputImage, filename=filename, recomended=recomended)

    else:
        return render_template('index.html', bukanFileGambar='yangDiupload bukanFileGambar', filename=file.filename)

@app.route('/displayOutputImage/<filename>')
def displayOutputImage(filename):
    return redirect(url_for('static', filename='outputImage/'+ filename), code=301)

@app.route('/displayFaces/<filename>')
def displayFaces(filename):
    return redirect(url_for('static', filename='faces/'+ filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
