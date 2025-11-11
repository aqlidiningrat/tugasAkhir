def perbandinganTrainTest():
    kelompokCitra = ['citraIdeal', 'citraTidakIdeal', 'citraGabungan']
    d1 = [462, 532, 994]
    d2 = [198, 228, 426]
    fig = plt.figure(figsize=(10,5))
    plt.title('Perbandingan Data Training dan Data Testing kelompokCitra')
    # plt.xlabel('jumlah')
    # plt.ylabel('kelompokCitra')
    chart = plt.subplot()
    chart.plot(kelompokCitra, d1, marker='.')
    chart.plot(d2, marker='.')
    plt.legend(['dataTraining','dataTesting'])
    plt.savefig('d:/skripsi/wan/'+'perbandinganTraining-Testing'+'.jpg')
    plt.show()

def barHasil():
    metriks = ['Minkowski','L2','Euclidean','Nan_euclidean','Sqeuclidean']
    akurasi = [83,83,83,83,83]
    # plt.figure(figsize=(10,5))
    plt.bar(x=metriks, height=akurasi)
    plt.savefig('d:/skripsi/wan/'+'barHasil'+'.jpg')
    plt.show()

def lineHasilAkhirPengujian():
    kelompokCitra = ['citraIdeal', 'citraTidakIdeal', 'citraGabungan']
    d1 = [74, 66, 68]
    d2 = [83, 72, 76]
    d3 = [84, 72, 77]
    fig = plt.figure(figsize=(10,5))
    plt.title('Perbandingan Akurasi kelompokCitra dan ekstraksiCiriCitra')
    # plt.xlabel('jumlah')
    plt.ylabel('Akurasi')
    chart = plt.subplot()
    chart.plot(kelompokCitra, d1, marker='.')
    chart.plot(d2, marker='.')
    chart.plot(d3, marker='.')
    plt.legend(['akurasiCiriLBP','akurasiCiriHOG','akurasiCiriGabungan'])
    plt.savefig('d:/skripsi/wan/'+'akurasiKelompokCitra_dan_ekstraksiCiriCitra'+'.jpg')
    plt.show()

def prosesEkstraksiCiri(path):
    img = cv.cvtColor(cv.resize(cv.imread(path), (128,128)), cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(cv.resize(cv.imread(path), (128,128)), cv.COLOR_BGR2GRAY)
    img_HOG = cv.cvtColor(cv.resize(cv.imread('d:/images/faceHOG.jpg'), (128,128)), cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('citraInput')
    plt.subplot(1,3,2)
    plt.imshow(img_gray, 'gray')
    plt.title('citraGray')
    plt.subplot(1,3,3)
    plt.imshow(img_HOG)
    plt.title('citraHOG')
    # plt.suptitle('Citra Preprocessing HOG', fontsize='xx-large', weight='bold')
    # print(img_gray, img_gray.shape)
    # print(img_HOG, img_HOG.shape)

    # for i in range(0,8):
    #     arr = img_8x8[i]
    #     print(arr[0:8])

    plt.show()

def gambarGelap(img_normal, alpha):
    img_gelap = img_normal * alpha
    img_gelap = img_gelap.astype(np.uint8)
    img_gelap[img_gelap < 0] = 0
    img_gelap[img_gelap > 255] = 255
    return img_gelap

def low_brightnes():
    img = cv.imread('d:/images/test8.png')
    img_normal = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    hasil1 = gambarGelap(img_normal, 0.1)
    hasil2 = gambarGelap(img_normal, 0.2)
    hasil3 = gambarGelap(img_normal, 0.3)


    cv.imshow('gambarAsli', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('gambarHasil', hasil1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('gambarHasil', hasil2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('gambarHasil', hasil3)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite('d:/images/faces/low_brightnes_monalisa.jpg', img_gelap)

if __name__=='__main__':
    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt

    # prosesEkstraksiCiri('d:/images/faces/face_monalisa.jpg')
    low_brightnes()
