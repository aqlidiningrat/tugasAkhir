import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random
root = os.getcwd()

def contours(faceImageGray, m, root):
    # imgPath = path
    # img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    # img = img[2680:2860, 1490:1720] # cropImage_punyaKevinWood
    img = faceImageGray
    print(img.shape)

    plt.figure(figsize=(5,5))
    # plt.suptitle('kevinWood Contours', fontsize='xx-large', weight='extra bold')
    # plt.subplot(2,3,1)
    # plt.title('imgGray')
    # plt.imshow(img, cmap='gray')

    height,width = img.shape
    scale = 4
    heightScale = int(scale*height)
    widthScale = int(scale*width)
    img = cv.resize(img, (widthScale, heightScale))

    _,thresh = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((70,70), np.uint8)
    thresh = cv.dilate(thresh,kernel)
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print('pathFile:',path)
    print('len contours:',len(contours),'type contours:',type(contours))
    contours = contours
    cv.drawContours(img,contours,-1,(0,0,255),3)

    # plt.subplot(2,3,2)
    # plt.title('imgThresh')
    # plt.imshow(thresh, cmap='gray')
    # plt.subplot(2,3,3)
    # plt.title('imgcontours')
    # plt.imshow(img, cmap='gray')

    M = cv.moments(contours[0])
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])

    # plt.subplot(2,3,4)
    # plt.title('imgMoments')
    # plt.imshow(img, cmap='gray')
    # plt.plot(Cx, Cy, 'r*')

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
    plt.subplot(1,1,1)
    # plt.title('imgConvexHull')
    plt.imshow(img, cmap='gray')
    plt.plot(hull[:,0], hull[:,1], 'r-')

    x,y,w,h = cv.boundingRect(contours[0])
    # plt.subplot(2,3,6)
    # cv.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
    # plt.title('imgBoundingRect')
    # plt.imshow(img, cmap='gray')

    aspectRatio = w/h
    # extent = area(w*h)
    extent = w*h
    solidity = area/cv.contourArea(hull)
    equiDIa = np.sqrt(4*area/np.pi)

    jenisWajah = ['oval face','triangle face', 'square face', 'round face']

    try:
        _,_,_angle = cv.fitEllipse(contours[0])
    except Exception as e:
        cv.imwrite(os.path.join(root, 'static/faces/','wajah'+str(m)+'.png'), img)
        # cv.imshow('imgConvexHull',img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        jenisWajah = jenisWajah[random.randint(0,3)]
        return jenisWajah

    print('aspectRatio:',aspectRatio)
    print('extent:',extent)
    print('solidity:',solidity)
    print('equiDIa:',equiDIa)
    print('_angle:', _angle)

    if (len(contours) <= 2):
        jenisWajah = jenisWajah[0]
    elif (len(contours) <= 4):
        jenisWajah = jenisWajah[1]
    elif (len(contours) <= 6):
        jenisWajah = jenisWajah[2]
    elif (len(contours) <= 8):
        jenisWajah = jenisWajah[3]
    else:
        jenisWajah = jenisWajah[random.randint(0,3)]

    # plt.suptitle('wajah '+str(m)+' '+jenisWajah, fontsize='xx-large', weight='extra bold')
    # plt.show()

    cv.imwrite(os.path.join(root, 'static/faces/','wajah'+str(m)+'.png'), img)
    # cv.imshow('imgConvexHull',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return jenisWajah

def faceImage(path, root):
    # load the haarcascade xml file
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # read the image
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(gray.shape)

    # cv.imshow('imgGray', gray)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
    # draw rectangles around the faces
    m = 0
    for (x,y,w,h) in faces:
        m = m+1
        print('x,y,w,h:',x,w,y,h)
        faceImageGray = gray[y:y+h, x:x+w]
        print(faceImageGray.shape)
        cv.imshow('imgResize1', faceImageGray)
        cv.waitKey(0)
        cv.destroyAllWindows()

        print('\n _____wajah '+str(m))
        jenisWajah = contours(faceImageGray, m, root)
        print('jenisWajah:', jenisWajah)

        xC,yC,wC,hC = x-10, y-25, w+10, h+35
        print('xC,yC,wC,hC:',x,y,w,h)
        faceImageGray2 = gray[y:(y-25)+(h+35), x:(x-10)+(w+10)]

        # saveWajahnya
        # cv.imwrite(os.path.join(root, 'static/faces/', 'wajah'+str(m)+'.png'), faceImageGray2)

        cv.imshow('imgResize2', faceImageGray2)
        cv.waitKey(0)
        cv.destroyAllWindows()

        cv.rectangle(img, (xC,yC),(xC+wC, yC+hC), (236,233,234), 2)
        cv.putText(img, jenisWajah, (xC+5,yC-5), cv.FONT_HERSHEY_DUPLEX, 0.7, (236,233,234),2)
        cv.putText(img, 'wajah '+str(m), (xC+5,yC+hC-5), cv.FONT_HERSHEY_DUPLEX, 0.5, (236,233,234),1)

    # display the output
    cv.imshow(str(path), img)
    cv.imwrite(os.path.join(root,'static/outputImage/','outputImage_'+str(m)+'wajah'+'.png'), img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # return img

# def plotFaceImage():
#     images = ['khaidir.jpg','labelleFerronniere.jpg','eviDariati.jpg','monalisa.jpg','salvatorMundi.jpg', 'ginevraDeBenci.jpg']
#
#     plt.figure(figsize=(10,7))
#     plt.suptitle('plotImages and faceDetectionImage', fontsize='xx-large', weight='extra bold')
#
#     plt.subplot(2,3,1)
#     image1 = faceImage(os.path.join('d:/images/', images[0]))
#     plt.title(images[0].split('.')[0])
#     plt.imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
#
#     plt.subplot(2,3,2)
#     image2 = faceImage(os.path.join('d:/images/', images[1]))
#     plt.title(images[1].split('.')[0])
#     plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
#
#     plt.subplot(2,3,3)
#     image3 = faceImage(os.path.join('d:/images/', images[2]))
#     plt.title(images[2].split('.')[0])
#     plt.imshow(cv.cvtColor(image3, cv.COLOR_BGR2RGB))
#
#     plt.subplot(2,3,4)
#     image4 = faceImage(os.path.join('d:/images/', images[3]))
#     plt.title(images[3].split('.')[0])
#     plt.imshow(cv.cvtColor(image4, cv.COLOR_BGR2RGB))
#
#     plt.subplot(2,3,5)
#     image5 = faceImage(os.path.join('d:/images/', images[4]))
#     plt.title(images[4].split('.')[0])
#     plt.imshow(cv.cvtColor(image5, cv.COLOR_BGR2RGB))
#
#     plt.subplot(2,3,6)
#     image6 = faceImage(os.path.join('d:/images/', images[5]))
#     plt.title(images[5].split('.')[0])
#     plt.imshow(cv.cvtColor(image6, cv.COLOR_BGR2RGB))
#
#     plt.savefig('d:/images/plotImages and faceDetectionImage')
#     plt.show()

if __name__ == '__main__':
    faceImage(os.path.join('d:/images/', 'ft.jpg'),root)
    # plotFaceImage()
