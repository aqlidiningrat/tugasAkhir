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

def barHasil():
    metriks = ['Minkowski','L2','Euclidean','Nan_euclidean','Sqeuclidean']
    akurasi = [83,83,83,83,83]
    # plt.figure(figsize=(10,5))
    plt.bar(x=metriks, height=akurasi)
    plt.savefig('d:/skripsi/wan/'+'barHasil'+'.jpg')

def lineHasilAkhirPengujian():
    kelompokCitra = ['Akurasi', 'Precision', 'Recall']
    d1 = [0.88, 0.75, 1.0]
    d2 = [0.78, 0.68, 1.0]
    fig = plt.figure(figsize=(10,5))
    plt.title('Perbandingan Parameter Decision Tree Max Depth 10')
    # plt.xlabel('jumlah')
    # plt.ylabel('Akurasi')
    chart = plt.subplot()
    chart.plot(kelompokCitra, d1, marker='.')
    chart.plot(d2, marker='.')
    plt.legend(['Entorpy 10','Gini Index 10'])
    # plt.savefig('d:/skripsi/wan/'+'akurasiKelompokCitra_dan_ekstraksiCiriCitra'+'.jpg')

if __name__=='__main__':
    import matplotlib.pyplot as plt

    # perbandinganTrainTest()
    # barHasil()
    lineHasilAkhirPengujian()

    plt.show()
