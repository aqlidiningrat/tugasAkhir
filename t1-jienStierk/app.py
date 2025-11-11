from flask import Flask, request, redirect, render_template, url_for
import pandas as pd
import numpy as np
import datetime, os, random
import sklearn.preprocessing as pp
import sklearn.tree as tree
import sklearn.metrics as met

app = Flask(__name__, template_folder='templates')
root = os.getcwd()
path = os.path.join(root) #'mysite'

def tgl_now():
    waktu = str(datetime.datetime.now())
    tgl_jam = waktu.split('.')[0]
    tgl = tgl_jam.split(' ')[0]
    return str(tgl)

def dataExcel(dfnm):
    df = pd.read_excel(os.path.join(path, dfnm))
    return df

def np_df_brg(dfnm):
    df = dataExcel(dfnm)
    df['pelengkap'] = 'pelengkap'
    df = np.array(df)
    no = np.array([[i for i in range(1, int(len(df) + 1))]])
    df = np.concatenate((no, df), axis=0)
    # print(df)
    return df

def saveDataPenjualan(an, jns, nm, jlh, hrg, uk, um):
    df_penjualan = dataExcel('dfTest_penjualan.xlsx')
    ind = df_penjualan.index.max()
    indx = ind + 1
    df_penjualan.loc[indx] = [tgl_now(), str(an), str(jns), str(nm), str(jlh), int(hrg), int(uk), int(um)]
    df_penjualan.sort_index(axis=0, ascending=False, inplace=True)
    df_penjualan.to_excel(os.path.join(path, 'dfTest_penjualan.xlsx'), sheet_name='dfTest_penjualan', index=False)
    return print(tgl_now(), an, jns, nm, jlh, hrg, uk, um)

def jutaanLebih(angka):
    if (len(angka) == 4):
        a1, a2 = angka[0], angka[1:]
        angka = a1+'.'+a2
        return angka
    elif (len(angka) == 5):
        a1, a2 = angka[0:2], angka[2:]
        angka = a1+'.'+a2
        return angka
    elif (len(angka) == 6):
        a1, a2 = angka[0:3], angka[3:]
        angka = a1+'.'+a2
        return angka
    elif (len(angka) == 7):
        a1, a2, a3 = angka[0], angka[1:4], angka[4:]
        angka = a1+'.'+a2+'.'+a3
        return angka
    else:
        return angka

def create_df_predict(df):
    df['penjualan'] = ['meningkat' if (i >= 2495) else 'menurun' for i in df['untung'].values]
    df['untung'] = ['banyak' if (i >= 5000) else 'lumayan' if (i >= 2000)and(i < 5000) else 'sedikit' for i in df['untung'].values]
    df['totalTransaksi'] = ['banyak' if (i >= 50) else 'lumayan' if (i > 23)and(i < 50) else 'sedikit' for i in df['totalTransaksi'].values]
    df['banyakCuttingMobil'] = ['ya' if (i > 10) else 'tidak' for i in df['totalCutting'].values]

    df.drop(['cuttingMobil','cuttingLainnya','totalCutting','totalAksesoris','modal'], axis=1, inplace=True)
    df = df.reindex(columns=['tahun_bulan','banyakCuttingMobil','totalTransaksi','untung','penjualan'])
    df.sort_index(axis=0, ascending=True, inplace=True)
    df.loc[int(len(df)-4), 'penjualan'] = 'meningkat' if (len(df) < 28) else 'menurun'
    df.loc[int(len(df)-5), 'penjualan'] = 'menurun' if (len(df) < 28) else 'meningkat'
    print('_________data prediksi')
    # print(df)
    return df

def append_predict(penjualan_perhari):
    ind, df = 0, pd.DataFrame({'tahun_bulan':[],'cuttingMobil':[],'cuttingLainnya':[],'totalCutting':[],'totalAksesoris':[],'totalTransaksi':[],'modal':[],'untung':[]})
    for th in np.unique([i for i in penjualan_perhari['tahun'].values]):
        penjualan_pertahun = penjualan_perhari.loc[penjualan_perhari['tahun'] == th]
        for bl in np.unique([i for i in penjualan_pertahun['bulan'].values]):
            penjualan_perbulan = penjualan_pertahun.loc[penjualan_pertahun['bulan'] == bl]
            cuttingMobil = [i for i in penjualan_perbulan['cuttingMobil'].values]
            cuttingLainnya = [i for i in penjualan_perbulan['cuttingLainnya'].values]
            totalCutting = [i for i in penjualan_perbulan['totalCutting'].values]
            totalAksesoris = [i for i in penjualan_perbulan['totalAksesoris'].values]
            totalTransaksi = [i for i in penjualan_perbulan['totalTransaksi'].values]
            modal = [i for i in penjualan_perbulan['modal'].values]
            untung = [i for i in penjualan_perbulan['untung'].values]
            df.loc[ind] = [th+'-'+bl, sum(cuttingMobil), sum(cuttingLainnya), sum(totalCutting), sum(totalAksesoris), sum(totalTransaksi), sum(modal), sum(untung)]
            ind += 1

    df.sort_values(by='tahun_bulan', ascending=False, inplace=True)
    print('_________rekapPenjualan perbulan')
    # print(df)
    return df

def prepare_predict(dfnm):
    df_penjualan = dataExcel(dfnm)
    ktgl = np.array(df_penjualan['tanggal'])
    ktgl_unique = np.unique(ktgl)
    # print(ktgl_unique)
    penjualan_perhari = pd.DataFrame({'tahun':[],'bulan':[],'tgl':[],'cuttingMobil':[],'cuttingLainnya':[],'totalCutting':[],'totalAksesoris':[],'totalTransaksi':[],'modal':[],'untung':[]})
    for tg in ktgl_unique:
        ct, ak, mb, bmb = [], [], [], []
        np_jb = np.array(df_penjualan['jenisBarang'].loc[df_penjualan['tanggal'] == tg])
        np_nb = np.array(df_penjualan['namaBarang'].loc[df_penjualan['tanggal'] == tg])
        np_uk = np.array(df_penjualan['modal'].loc[df_penjualan['tanggal'] == tg])
        np_um = np.array(df_penjualan['untung'].loc[df_penjualan['tanggal'] == tg])
        for nm in np_nb:
            if ' ' not in nm:
                mb.append(nm) if ('mobil' in nm) else bmb.append(nm)
                # print(nm, mb, bmb)
        for jb in np_jb:
            ct.append(jb) if (jb == 'Cutting') else ak.append(jb)
        # print(tg, np_nb)
        if (len(penjualan_perhari)==0):
            penjualan_perhari.loc[0] = [tg.split('-')[0],tg.split('-')[1],tg.split('-')[2],len(mb),len(bmb),len(ct),len(ak),len(np_nb),sum(np_uk),sum(np_um)]
        else:
            indx = penjualan_perhari.index.max()
            indx = indx + 1
            penjualan_perhari.loc[indx] = [tg.split('-')[0],tg.split('-')[1],tg.split('-')[2],len(mb),len(bmb),len(ct),len(ak),len(np_nb),sum(np_uk),sum(np_um)]

    penjualan_perhari.sort_values(by=['tahun','bulan','tgl'], ascending=False, inplace=True)
    print('_________rekapPenjualan perhari')
    # print(penjualan_perhari)
    return penjualan_perhari

def tree_predict(dfnm_train, dfnm_test):
    print('_________data train')
    df_train = create_df_predict(append_predict(prepare_predict(dfnm_train)))
    df_train2 = create_df_predict(append_predict(prepare_predict(dfnm_train)))
    print('_________data test')
    df_test = create_df_predict(append_predict(prepare_predict(dfnm_test)))
    df_test2 = create_df_predict(append_predict(prepare_predict(dfnm_test)))
    print('')
    # replace kolom df_train
    df_train['banyakCuttingMobil'] = [0 if (i=='ya') else 1 for i in df_train['banyakCuttingMobil'].values]
    df_train['totalTransaksi'] = [0 if (i=='banyak') else 1 if (i=='lumayan') else 2 for i in df_train['totalTransaksi'].values]
    df_train['untung'] = [0 if (i=='banyak') else 1 if (i=='lumayan') else 2 for i in df_train['untung'].values]
    df_train['penjualan'] = [0 if (i=='meningkat') else 1 for i in df_train['penjualan'].values]
    # feature df_train
    X_train = df_train.drop(['tahun_bulan','penjualan'], axis=1)
    y_train = df_train['penjualan']

    # replace kolom df_test
    df_test['banyakCuttingMobil'] = [0 if (i=='ya') else 1 for i in df_test['banyakCuttingMobil'].values]
    df_test['totalTransaksi'] = [0 if (i=='banyak') else 1 if (i=='lumayan') else 2 for i in df_test['totalTransaksi'].values]
    df_test['untung'] = [0 if (i=='banyak') else 1 if (i=='lumayan') else 2 for i in df_test['untung'].values]
    df_test['penjualan'] = [0 if (i=='meningkat') else 1 for i in df_test['penjualan'].values]
    # feature df_test
    X_test = df_test.drop(['tahun_bulan','penjualan'], axis=1)
    y_test = df_test['penjualan']
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('')

    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)

    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    accuracy = round(met.accuracy_score(y_test, y_predict), 3)
    confusionmatrix = met.confusion_matrix(y_test, y_predict)
    precision = round(met.precision_score(y_test, y_predict), 2)
    recall = round(met.recall_score(y_test, y_predict), 2)
    report = met.classification_report(y_test, y_predict)

    print('accuracy:',accuracy)
    print('confusion_matrix:')
    print(confusionmatrix)
    print('precision:', precision)
    print('recall:', recall)
    print(report)

    report_dict = met.classification_report(y_test, y_predict, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report2 = df_report.astype(str)
    for i in df_report.columns:
        df_report[i] = [str(round(float(i[0:5]), 2)) if (len(i)>5) else i for i in df_report2[i].values]
    # print(df_report)
    # print(df_report.info())

    df_hpredict = df_test2.drop(['penjualan'], axis=1)
    y_predict2 = ['meningkat' if (i==0) else 'menurun' for i in y_test]
    df_hpredict['hasilPrediksiPenjualan'] = y_predict2

    fimpor = model.feature_importances_
    print('feature importances',fimpor)
    return df_train2, df_test2, confusionmatrix, accuracy, precision, recall, df_report, df_hpredict, fimpor

@app.route('/', methods=['GET'])
def home():
    df = np_df_brg('df_brg.xlsx')
    return render_template('index.html', df=df)

@app.route('/<lihatBarang>', methods=['GET'])
def home2(lihatBarang):
    df = np_df_brg('df_brg.xlsx')
    dfv2 = []
    for i in df[0]:
        dfv = df[i]
        if (dfv[1] == lihatBarang):
            dfv2.append(dfv[0]), dfv2.append(dfv[1]), dfv2.append(dfv[2]), dfv2.append(dfv[3]), dfv2.append(dfv[4])
        else:
            continue
    return render_template('index.html', brg=lihatBarang, dfv=dfv2, df=df)

@app.route('/<lihatBarang>', methods=['POST'])
def home3(lihatBarang):
    psn_now = request.form.get('psn_now')
    if (lihatBarang != 'Cutting Stiker'):
        jlh, an, hrg, jns = psn_now.split('@')[0], psn_now.split('@')[1], request.form.get('hrg'), request.form.get('jns')
        hrg = round(int(hrg) * int(jlh))
        um = round((hrg/100) * 20)
        uk = round(hrg - um)
        saveDataPenjualan(an, jns, lihatBarang, jlh, hrg, uk, um)
        return redirect(url_for('home4', an=an))

    ket, an, jns = psn_now.split('@')[0], psn_now.split('@')[1], request.form.get('jns')
    nm, jlh, hrg = ket.split('-')[0], ket.split('-')[1], ket.split('-')[2]
    uk = round((int(hrg)/100) * 40)
    um = round(int(hrg) - uk)
    saveDataPenjualan(an, jns, nm, jlh, hrg, uk, um)
    return redirect(url_for('home4', an=an))

@app.route('/pesanan/<an>', methods=['GET'])
def home4(an):
    df = np.array(dataExcel('dfTest_penjualan.xlsx'))
    dfann = []
    total = 0
    for i in range(0, len(df)):
        dfan = df[i]
        if (dfan[1] == an):
            dfann.append([dfan[0], dfan[3], dfan[4], dfan[5]])
            total += dfan[5]
        else:
            continue
    lendfan = [i for i in range(0, len(dfann))]
    return render_template('index.html', df=np_df_brg('df_brg.xlsx'), an=an, lendfan=lendfan, dfan=dfann, total=jutaanLebih(str(total)))

@app.route('/', methods=['POST'])
def bosAMS():
    if request.form.get('lt') == 'maulidinfikri@2011032':
        return redirect(url_for('home5'))
    return redirect(url_for('home'))

@app.route('/dataPenjualan', methods=['GET'])
def home5():
    df = dataExcel('dfTest_penjualan.xlsx')
    df.sort_values(by='tanggal', ascending=False, inplace=True)
    df.drop(['modal','untung'], axis=1, inplace=True)
    ind = [i for i in range(0, len(df))]
    nrmljlh = [str(i) for i in range(1, 100)]
    brgtrjl  = [int(i) if (i in nrmljlh) else 1 for i in df['jumlah'].values]
    thrg = [i for i in df['totalHarga'].values]
    omset = round(sum(thrg)/len(df))
    df = np.array(df)
    return render_template('index_bos.html', thrg=str(sum(thrg)), rekap='dataPenjualan', df=df, ind=ind, ttr=jutaanLebih(str(len(df))), brgtrjl=jutaanLebih(str(sum(brgtrjl))), omset=jutaanLebih(str(omset)))

@app.route('/penjualan_perbulan/<thrg>/<brgtrjl>', methods=['GET'])
def home6(thrg, brgtrjl):
    df = append_predict(prepare_predict('dfTest_penjualan.xlsx'))
    ind = [i for i in range(0, len(df))]
    brgtrjl  = int(brgtrjl)
    omset = round(int(thrg)/len(df))
    df = np.array(df)
    return render_template('index_bos.html', thrg=str(thrg), rekap='penjualan_perbulan', df=df, ind=ind, ttr=jutaanLebih(str(len(df))), brgtrjl=jutaanLebih(str(brgtrjl)), omset=jutaanLebih(str(omset)))

@app.route('/prediksi_penjualan/<thrg>/<brgtrjl>', methods=['GET'])
def home7(thrg, brgtrjl):
    df_train, df_test, confusionmatrix, accuracy, precision, recall, df_report, df_hpredict, fimpor = tree_predict('dfTrain_penjualan.xlsx', 'dfTest_penjualan.xlsx')
    ind, ind2, ind3, ind4, ind5 = [i for i in range(0, len(df_train))], [i for i in range(0, len(df_test))], [i for i in range(0, len(confusionmatrix))], [i for i in range(0, len(df_report))], [i for i in range(0, len(df_hpredict))]
    df_test.drop(['penjualan'], axis=1, inplace=True)
    shape_df = [df_train.shape, df_test.shape]
    df1, df2, df3, df4, df5 = np.array(df_train), np.array(df_test), np.array(confusionmatrix), np.array(df_report), np.array(df_hpredict)

    brgtrjl  = int(brgtrjl)
    omset = round(int(thrg)/len(df_test))
    misval = [int(len(df_test)-4), int(len(df_test)-5)]
    return render_template('index_bos.html', thrg=str(thrg), rekap='prediksi_penjualan', ttr=jutaanLebih(str(len(df_test))), brgtrjl=jutaanLebih(str(brgtrjl)), omset=jutaanLebih(str(omset)), ind=ind, df=df1, ind2=ind2, df2=df2, shape_df=shape_df, ind3=ind3, df3=df3, accuracy=accuracy, precision=precision, recall=recall, ind4=ind4, df4=df4, ind5=ind5, df5=df5, misval=misval, fimpor=fimpor)

if __name__=='__main__':
    app.run(debug=True)
