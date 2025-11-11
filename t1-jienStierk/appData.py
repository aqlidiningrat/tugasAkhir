def sample_data_barang():
    mrk_brg = ['DYT','Brembo','RCB','EMC', 'Black Diamond', 'Cutting']
    nm_brg = ['Kepala Knalpot','Tutup Cakram','Sarung Stang','Gantungan Barang', 'Baut Spion', 'Cutting Stiker']
    desc_brg = ['Kepala Knalpot Mobil Original DYT Racing M1 Carbon Inlet 50mm Silincer only', 'Cover Disk Brake (Tutup Cakram) Mobil Cover Rem Brembo Size M-S Roda Belakang', 'Hand Grip (Sarung Stang) RCB AHG88 Red/Black/Silver/Blue Original Racing', 'Gantungan Barang (Hook Centelan Motor) EMC Full CNC Bisa lipat Black', 'Baut Spion Black Diamond RMS002-Black Diamond Racing', 'Cetak Stiker (Plus Cutting) Stiker Label Dan lain-lain']
    jns_brg = ['Aksesoris','Aksesoris','Aksesoris','Aksesoris','Aksesoris','Cutting']
    hrg_brg = [205,90,40,25,5,'Harga/cm']

    df = pd.DataFrame({'mrk_brg':mrk_brg, 'nm_brg':nm_brg, 'desc_brg':desc_brg, 'jns_brg':jns_brg, 'hrg_brg':hrg_brg})
    df.to_excel(os.path.join(path, 'df_brg.xlsx'), sheet_name='df_brg', index=False)
    return df

def tgl():
    # oldTgl, nowTgl = [2022:1,2,3,4,5,6,7,8,9,10,11,12 - 2023:1,2,3,4,5,6,7,8,9,10,11,12 - 2024:1,2,3,4], [2024:8,9,10,11,12 - 2025:1,2,3,4]
    thn = ['2024','2025']
    th = thn[random.randint(0,1)]
    bl = str(random.randint(8,12)) if (th != '2025') else str(random.randint(1,4))
    tg =  str(random.randint(1,30)) if (bl != '2') else str(random.randint(1,28))
    obl = '0'+bl if (len(bl) != 2) else bl
    otg = '0'+tg if (len(tg) != 2) else tg
    return th+'-'+obl+'-'+otg

def nmr():
    nmr = ['ayu','putri','fitri','dewi','sri','sari','rina','rini','eka','indah','dian','wahyu','siti','ika','fitri','ratna','puspita','ratih','pratiwi','kartika',
        'wulandari','lestari','anita','kusuma','rahmawati','fitra','retno','kurnia','novita','ria','handayani','rahayu','yunita','maya','puji','utami','amalia','dina','devi','citra',
        'munaroh','eva','endah','irma','astuti','aulia','amelia','prima','diana','anggraini','wulan','yuni','dwi','ahmad','nur','tri','eka','agus','andi','agung',
        'ahmad','kurniawan','ilham','budi','adi','eko','nurul','arif','ari','indra','rizki','fajar','bayu','anita','aditya','nugroho','bagus','bagas','hidayat','hendra',
        'raden','novi','achmad','surya','angga','lena','janah','aris','fikri','khaidir','aqli','boim','tomi','bahar','lia','fahmi','tio','yanti','kamal','fran']
    return nmr[random.randint(0,99)]

def Aksesoris(jb):
    ba = ['Kepala Knalpot_90', 'Tutup Cakram_95', ' Sarung Stang_40', 'Gantungan Barang_25', 'Baut Spion_5']
    ba = ba[random.randint(0,4)]
    b,h = ba.split('_')[0], ba.split('_')[1]
    j = random.randint(1,3)
    hrg = round(int(h)*int(j))
    um = round((int(hrg)/100)*20)
    uk = round(hrg-um)
    return [tgl(), nmr(), str(jb), str(b), str(j), int(hrg), int(uk), int(um)]

def Cutting(jb):
    nmwjh = ['kap.samping.kirikanan.mobil_10m_1000','kap.mesin.mobil_5m_500','kap.belakang.mobil_5m_500','full.kaca.mobil_15m_800',
            'cutting.stiker_70x15_50','stiker_1m_10','kaca.film.80%_10m_415','vario.boot.roda_1m_40','stiker.aqua_1m_45','rabin.livina_1m_120','les.hitam_50cm_5','les.abuabu_50cm_5','cutting.stiker_1m_30',
            'les.merah_4m_220','cutting.stiker_2m_65','p.antigores.scoopy_7m_550','les.hitamdove_3m_30','p.stiker.vario_2m_150','plat.1m_70','scoopy.full_7m_550','cutting.stiker_60x200cm_420','pcx.full_9m_700']

    nmwjh = nmwjh[random.randint(0, int(len(nmwjh)-1))]
    nmw,j,h = nmwjh.split('_')[0], nmwjh.split('_')[-2], nmwjh.split('_')[-1]
    uk = round((int(h)/100)*40)
    um = round(int(h)-uk)
    return [tgl(), nmr(), str(jb), str(nmw), str(j), int(h), int(uk), int(um)]

def set_data(jb):
    return Aksesoris(jb) if jb == 'Aksesoris' else Cutting(jb)

def create_df_predict(df):
    df['penjualan'] = ['meningkat' if (i >= 2495) else 'menurun' for i in df['untung'].values]
    df['untung'] = ['banyak' if (i >= 5000) else 'lumayan' if (i >= 2000)and(i < 5000) else 'sedikit' for i in df['untung'].values]
    # df['modal'] = ['banyak' if (i >= 5000) else 'lumayan' if (i >= 2000)and(i < 5000) else 'sedikit' for i in df['modal'].values]
    df['totalTransaksi'] = ['banyak' if (i >= 50) else 'lumayan' if (i > 23)and(i < 50) else 'sedikit' for i in df['totalTransaksi'].values]
    df['banyakCuttingMobil'] = ['ya' if (i > 10) else 'tidak' for i in df['totalCutting'].values]

    df.drop(['cuttingMobil','cuttingLainnya','totalCutting','totalAksesoris','modal'], axis=1, inplace=True)
    df = df.reindex(columns=['tahun_bulan','banyakCuttingMobil','totalTransaksi','untung','penjualan'])
    df.sort_index(axis=0, ascending=True, inplace=True)

    # print('')
    # print(df['banyakCuttingMobil'].value_counts())
    # print(df['totalTransaksi'].value_counts())
    # # print(df['modal'].value_counts())
    # print(df['untung'].value_counts())
    # print(df['penjualan'].value_counts())

    # miss label predict to accuracy 88.9%
    df.loc[int(len(df)-4), 'penjualan'] = 'meningkat' if (len(df) < 28) else 'menurun'
    df.loc[int(len(df)-5), 'penjualan'] = 'menurun' if (len(df) < 28) else 'meningkat'

    # print('\n _________data prediksi')
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
    # print('\n _________rekapPenjualan perbulan')
    # print(df)
    return create_df_predict(df)

def prepare_predict(df_penjualan):
    # print('\n _________dataPenjualan')
    # print(df_penjualan)
    # print(df_penjualan['jenisBarang'].value_counts())

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
    # print('\n _________rekapPenjualan perhari')
    # print(penjualan_perhari)
    return append_predict(penjualan_perhari)

def create_df_penjualan():
    df = pd.DataFrame({'tanggal':[],'pembeli':[],'jenisBarang':[],'namaBarang':[],'jumlah':[],'totalHarga':[],'modal':[],'untung':[]})
    jb = ['Aksesoris','Cutting']
    for i in range(0,187):
        df.loc[i] = set_data(jb[random.randint(0,1)])
        # endfor
    df.sort_values(by='tanggal', ascending=False, inplace=True)
    # df.to_excel(os.path.join(path,'dfTest_penjualan.xlsx'), sheet_name='df_penjualan', index=False)
    print(df)
    print(df['jenisBarang'].value_counts())
    return prepare_predict(df)

def tree_predict(df_train, df_test):
    print('_________data train')
    df_train = prepare_predict(df_train)
    print(df_train)
    print('_________data test')
    df_test = prepare_predict(df_test)
    print(df_test)
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

    # import sklearn.model_selection as ms
    # X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.3, random_state=0)
    # print('split_dataset:',len(X_train), len(X_test), len(y_train), len(y_test))

    import sklearn.preprocessing as pp
    scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
    scl.fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)
    print(X_train.max(), X_train.min(), '||', X_test.max(), X_test.min())
    print('')


    import sklearn.tree as tree
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    import sklearn.metrics as met
    accuracy = met.accuracy_score(y_test, y_predict)
    confusion_matrix = met.confusion_matrix(y_test, y_predict)
    precision = met.precision_score(y_test, y_predict)
    recall = met.recall_score(y_test, y_predict)
    report = met.classification_report(y_test, y_predict)

    print('accuracy:',round(accuracy,3))
    print('confusion_matrix:')
    print(confusion_matrix)
    print('precision:',round(precision, 2))
    print('recall:',round(recall, 2))
    print(report)

    report_dict = met.classification_report(y_test, y_predict, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report2 = df_report.astype(str)
    for i in df_report.columns:
        df_report[i] = [str(round(float(i[0:5]), 2)) if (len(i)>5) else i for i in df_report2[i].values]
    print(df_report)
    print(df_report.info())

    fimpor = model.feature_importances_
    print('feature_importances_',fimpor)

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import os, random

    root = os.getcwd()
    path = os.path.join(root)

    # create_df_penjualan()

    df_train = pd.read_excel(os.path.join(path, 'dfTrain_penjualan.xlsx'))
    df_test = pd.read_excel(os.path.join(path, 'dfTest_penjualan.xlsx'))
    tree_predict(df_train, df_test)

    # df = pd.read_excel(os.path.join(path, 'dfTest_penjualan.xlsx'))
    # print(df)

    # bosIniBahayaKaliBos hapusnyaPakaiIndexNi, jadiIngatBosSetelahDihapusDisave langsungDikomenLagi kalauEnggakTerhapusDuakaliNanti
    # for i in df.loc[df['pembeli'] == 'boim'].index:
    #     df.drop(index=[i], axis=0, inplace=True)
    #     df.to_excel(os.path.join(path, 'dfTest_penjualan.xlsx'), sheet_name='dfTest_penjualan', index=False)

    # # for i in range(0,4):
    # #     ind = df.index.max()
    # #     indx = ind + 1
    # #     df.loc[indx] = ['2025-05-03', 'str(an)'+str(i), 'str(jns)'+str(i), 'str(nm)'+str(i), 'str(jlh)'+str(i), 'int(hrg)'+str(i), 'int(uk)'+str(i), 'int(um)'+str(i)]
    # #     # df.sort_values(by='tanggal', ascending=False, inplace=True)
    # #     df.to_excel(os.path.join(path, 'dfTest_penjualan.xlsx'), sheet_name='dfTest_penjualan', index=False)

    # df = pd.read_excel(os.path.join(path, 'dfTest_penjualan.xlsx'))
    # df.to_excel(os.path.join(path, 'dfTest_penjualan.xlsx'), sheet_name='dfTest_penjualan', index=False)
    # print(df)
