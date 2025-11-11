from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2 as cv
import os

app = Flask(__name__)
root = os.getcwd()
path = os.path.join(root) #'mysite'

# app.config
app.secret_key = 'TA_sekar_2111004'
app.config['UPLOAD_FOLDER'] = os.path.join(path,'static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #batas ukuran gambar 16mb
ALLOWED_EXTENSIONS = set(['jpeg','jpg','png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def deleteGambar():
    for ffolder in ['inputImage','outputImage']:
        for fname in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], ffolder)):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], ffolder, fname))
            # endfor
        # endfor
    return 'dataBerhasilDihapusBoss..'

def listImage():
    listOutputImage = [fname for fname in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'outputImage'))]
    if (len(listOutputImage) < 11):
        listFix = sorted([listOutputImage[i] for i in range(0,len(listOutputImage))], reverse=True)
        # endif
    elif (len(listOutputImage) > 10)and(len(listOutputImage) < 21):
        a = sorted([listOutputImage[i] for i in range(0,10)], reverse=True)
        b = sorted([listOutputImage[i] for i in range(10,len(listOutputImage))], reverse=True)
        listFix = a+b
        # endif
    elif (len(listOutputImage) > 20)and(len(listOutputImage) < 31):
        a = sorted([listOutputImage[i] for i in range(0,10)], reverse=True)
        b = sorted([listOutputImage[i] for i in range(10,20)], reverse=True)
        c = sorted([listOutputImage[i] for i in range(20,len(listOutputImage))], reverse=True)
        listFix = a+b+c
        # endif
    else:
        deleteGambar()
        listFix = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'outputImage'))
        # endif
    print(listOutputImage)
    return listFix

def draw_rectangles(image, filename):
    # draw rectangles around the faces
    height, width, _ = image.shape
    start_point, end_point = (0,0), (width, height)
    cv.putText(image, 'Inpainting Image Metode Fast Marching', (5,12), cv.FONT_HERSHEY_DUPLEX, 0.3, (255,243,220), 1)
    cv.putText(image, 'TugasAkhir_SekarLuciaNingrum_2111004', (5,height-5), cv.FONT_HERSHEY_DUPLEX, 0.3, (255,243,220), 1)
    # save the outputImage
    nameOutputImage = '0'+str(len(listImage()))+'_'+filename if (len(listImage()) < 10) else str(len(listImage()))+'_'+filename
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],'outputImage/'+nameOutputImage), img_rgb)
    return nameOutputImage

def inpaintedImage(filename):
    # readImage
    image = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'inputImage', filename))
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img_resize = cv.resize(img_rgb, (1024, 1024))
    # edgePreservingFiltering
    filtered = cv.bilateralFilter(img_resize, d=9, sigmaColor=75, sigmaSpace=75)
    mask = cv.inRange(filtered, (0,0,0), (50,50,50))
    inpainted = cv.inpaint(filtered, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA) #cv.INPAINT_NS
    return draw_rectangles(inpainted, filename)

@app.route('/', methods=['GET'])
def home():
    return  render_template('index.html', listOutputImage=listImage())

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
        filenameOutputImage = inpaintedImage(filename)
        return render_template('index.html', listOutputImage=listImage(), filenameOutputImage=filenameOutputImage)

    else: # yangDiupload bukanFileGambar
        return render_template('index.html', filenameOutputImage='yangDiupload bukanFileGambar', filename=file.filename)

@app.route('/displayOutputImage/<filename>', methods=['GET'])
def displayOutputImage(filename):
    return redirect(url_for('static', filename='outputImage/'+filename), code=301)

@app.route('/downloadOutputImage/<filename>', methods=['GET'])
def downloadOutputImage(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'outputImage/'+filename), as_attachment=True)

@app.route('/hapusImageTA', methods=['GET'])
def hapus():
    deleteGambar()
    return redirect(url_for('home'))

if __name__=='__main__':
    app.run(debug=True)
