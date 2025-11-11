from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename

from fpdf import FPDF
from fpdf.fonts import FontFace

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import sklearn.model_selection as ms
import sklearn.metrics as met

import pandas as pd
import random
import os


root = os.getcwd()
path = os.path.join(root)

app = Flask(__name__, template_folder='templates')
app.secret_key = 'muhammadTomi'
app.config['UPLOAD_FOLDER'] = path

ALLOWED_EXTENSIONS = set(['xlsx'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def buatPDF(dfHpredict, Akurasi, kdPDF):
    class PDFTable:
        def __init__(self, title: str, data: pd.DataFrame):
            """
            Initialize the PDF with a title and the DataFrame to convert into a table.
            :param title: Title for the PDF document.
            :param data: Pandas DataFrame that holds the data to be displayed in the table.
            """
            self.title = title
            self.data = data
            self.pdf = FPDF(orientation='L', format='A4')
            self.pdf.add_page()
            self.pdf.set_font("Times", size=7)

        def add_title(self):
            """Adds the title to the PDF."""
            self.pdf.set_font("Times", style="B", size=17)
            # filtering_pilihSatu
            if (kdPDF == '1:3'):
                # title2
                self.pdf.cell(0, 10, 'SISWA '+Akurasi[1], align="C")
                self.pdf.ln()
            elif (kdPDF == '1:1'):
                # title2
                self.pdf.cell(0, 10, 'SISWA '+Akurasi[1], align="C")
                self.pdf.ln()
            elif (kdPDF == '1:2'):
                # title2
                self.pdf.cell(0, 10, 'SISWA '+Akurasi[1], align="C")
                self.pdf.ln()

            # filtering_pilihDua
            elif (kdPDF == '2:1,3'):
                # title2
                self.pdf.cell(0, 10, 'SISWA '+Akurasi[1]+' '+Akurasi[2], align="C")
                self.pdf.ln()
            elif (kdPDF == '2:1,2'):
                # title2
                self.pdf.cell(0, 10, 'SISWA '+Akurasi[1]+' '+Akurasi[2], align="C")
                self.pdf.ln()
            elif (kdPDF == '2:2,3'):
                # title2
                self.pdf.cell(0, 10, 'SISWA '+Akurasi[1]+' '+Akurasi[2], align="C")
                self.pdf.ln()

            # filtering_pilihTiga
            elif (kdPDF == '3:1,2,3'):
                # title2
                self.pdf.cell(0, 10, 'SISWA '+Akurasi[1]+' '+Akurasi[2]+' '+Akurasi[3], align="C")
                self.pdf.ln()

            else:
                print(' dataFull... \n dataFull... \n dataFull...')

            # title1
            self.pdf.cell(0, 10, self.title, align="C")
            self.pdf.ln()  # Add a line break

        def add_table(self):
            self.pdf.set_font("Times", style="B", size=7)
            """Adds the table from the DataFrame to the PDF."""
            # setColorText&bgText
            if ('XII AMP' in Akurasi):
                bgText = (19,75,112)
                text = (255,255,255)
            elif ('XII APHP' in Akurasi):
                bgText = (39,78,19)
                text = (255,255,255)
            elif ('XII ATP' in Akurasi):
                bgText = (255,103,25)
                text = (255,255,255)
            elif ('XII ATU' in Akurasi):
                bgText = (251,172,25)
                text = (255,255,255)
            elif ('XII TSM' in Akurasi):
                bgText = (68,68,68)
                text = (255,255,255)
            elif ('LAKI-LAKI' in Akurasi):
                bgText = (7,7,7) # black
                text = (255,255,255)
            elif ('PEREMPUAN' in Akurasi):
                bgText = (231, 136, 149) # pink
                text = (255,255,255)
            elif ('LULUS' in Akurasi):
                bgText = (89,113,157)
                text = (255,255,255)
            elif ('LULUS PRESTASI' in Akurasi):
                bgText = (43,169,114)
                text = (255,255,255)
            else:
                bgText = (214, 0, 45) # colorBacgroundHeading
                text = (255,255,255) # colorTextHeading

            # Create the table
            headings_style = FontFace(emphasis="ITALICS", color=text, fill_color=bgText, size_pt=8)
            with self.pdf.table(headings_style=headings_style, text_align='CENTER') as table:
                # Add headers (column names)
                header = self.data.columns.tolist()
                row = table.row()
                for column in header:
                    row.cell(column)

                # Add rows (data)
                for index, data_row in self.data.iterrows():
                    row = table.row()
                    for item in data_row:
                        row.cell(str(item))

        def save_pdf(self, filename: str):
            """Generates the PDF and saves it to a file."""
            self.pdf.output(filename)

    # Create a Pandas DataFrame
    df = pd.DataFrame(dfHpredict)
    # Create an instance of the PDFTable class
    pdf_table = PDFTable(title="DataHasilPrediksi_AplikasiTugasAkhir_MuhammadTomi_2011016", data=df)
    # Add title and table to the PDF
    pdf_table.add_title()
    pdf_table.add_table()
    # Save the PDF to a file
    pdf_table.save_pdf(os.path.join(path, 'dataHasilPrediksi_tomi.pdf'))


@app.route('/')
def home():
    return render_template('index.html', ttl='pengguna')

@app.route('/', methods=['POST'])
def siapa():
    usnm = request.form.get('usnm')
    pswd = request.form.get('pswd')

    # cekAkses
    users = ('muhammadTomi','muhammadTomi','muhammadTomi')
    pswdUser = (2011016,2011016,2011016)
    status = ('Admin','Admin','Admin')

    try:
        pswd = int(pswd)
    except Exception as e:
        flash('Username atau password salah..')
        return redirect(request.url)
    else:
        if int(pswd) not in pswdUser:
            flash('Password yang anda masukan salah..')
            # print('type(pswd):',type(pswd))
            # print('pswdUser[0]:',type(pswdUser[0]))
            return redirect(request.url)
        elif usnm not in users:
            flash('Username yang anda masukan salah..')
            # print('type(usnm):',type(usnm))
            # print('type(users[0]):',type(users[0]))
            return redirect(request.url)

        elif (usnm == users[0])and(pswd != pswdUser[0]):
            flash('Password yang anda masukan tidak sesuai..')
            return redirect(request.url)
        elif (usnm == users[1])and(pswd != pswdUser[1]):
            flash('Password yang anda masukan tidak sesuai..')
            return redirect(request.url)
        elif (usnm == users[2])and(pswd != pswdUser[2]):
            flash('Password yang anda masukan tidak sesuai..')
            return redirect(request.url)
        # else:
            # iniBisa404

    # finally:
        # pass

    # cekStatus
    if (usnm == users[0]):
        sts = status[0]
    elif (usnm == users[1]):
        sts = status[1]
    elif (usnm == users[2]):
        sts = status[2]

    return redirect(url_for('data', opsi='masuk', usnm=usnm, sts=sts))

@app.route('/filtering/', methods=['POST'])
def filterData():
    usnm = request.form.get('usnm') # hidden
    sts = request.form.get('sts') # hidden
    pilihJk = request.form.get('jk')
    pilihKelas = request.form.get('kelas')
    pilihStatus = request.form.get('pilihStatus')

    df = pd.read_excel(os.path.join(path,'dfHpredict_tomi.xlsx'))
    df['TANGGAL LAHIR'] = df['TANGGAL LAHIR'].astype('str')
    df.NISN = df.NISN.astype('str')
    # setNISN.str(10)
    nisnStr = []
    for values in df['NISN'].values:
        strVals = str(values).rjust(10, '0')
        nisnStr.append(strVals)
        # print(strVals)
    df['NISN'] = nisnStr
    df.NISN = df.NISN.astype('str')

    # filtering_pilihSatu
    if (pilihJk == 'semuaJenis')and(pilihKelas == 'semuaKelas')and(pilihStatus != 'semuaHasil'):
        print('filtering:',pilihKelas,pilihJk,pilihStatus)
        df2 = df.loc[df['Hasil Prediksi'] == pilihStatus]
        lenDT = df2.axes[0]
        Akurasi = 'SISWA  '+pilihStatus
        kdPDF = '1:3'

    elif (pilihJk == 'semuaJenis')and(pilihStatus == 'semuaHasil')and(pilihKelas != 'semuaKelas'):
        print('filtering:',pilihKelas,pilihJk,pilihStatus)
        df2 = df.loc[df['KELAS'] == pilihKelas]
        lenDT = df2.axes[0]
        Akurasi = 'SISWA  '+pilihKelas
        kdPDF = '1:1'

    elif (pilihKelas == 'semuaKelas')and(pilihStatus == 'semuaHasil')and(pilihJk != 'semuaJenis'):
        print('filtering:',pilihKelas,pilihJk,pilihStatus)
        df2 = df.loc[df['JENIS KELAMIN'] == pilihJk]
        lenDT = df2.axes[0]
        Akurasi = 'SISWA  '+pilihJk
        kdPDF = '1:2'

    # filtering_pilihDua
    elif (pilihJk == 'semuaJenis')and(pilihKelas != 'semuaKelas')and(pilihStatus != 'semuaHasil'):
        print('filtering:',pilihKelas,pilihJk,pilihStatus)
        df2 = df.loc[df['Hasil Prediksi'] == pilihStatus]
        df2 = df2.loc[df2['KELAS'] == pilihKelas]
        lenDT = df2.axes[0]
        Akurasi = 'SISWA  '+pilihKelas+'  '+pilihStatus
        kdPDF = '2:1,3'

    elif (pilihJk != 'semuaJenis')and(pilihStatus == 'semuaHasil')and(pilihKelas != 'semuaKelas'):
        print('filtering:',pilihKelas,pilihJk,pilihStatus)
        df2 = df.loc[df['KELAS'] == pilihKelas]
        df2 = df2.loc[df2['JENIS KELAMIN'] == pilihJk]
        lenDT = df2.axes[0]
        Akurasi = 'SISWA  '+pilihKelas+'  '+pilihJk
        kdPDF = '2:1,2'

    elif (pilihKelas == 'semuaKelas')and(pilihStatus != 'semuaHasil')and(pilihJk != 'semuaJenis'):
        print('filtering:',pilihKelas,pilihJk,pilihStatus)
        df2 = df.loc[df['Hasil Prediksi'] == pilihStatus]
        df2 = df2.loc[df2['JENIS KELAMIN'] == pilihJk]
        lenDT = df2.axes[0]
        Akurasi = 'SISWA  '+pilihJk+'  '+pilihStatus
        kdPDF = '2:2,3'

    # filtering_pilihTiga
    elif (pilihJk != 'semuaJenis')and(pilihKelas != 'semuaKelas')and(pilihStatus != 'semuaHasil'):
        print('filtering:',pilihKelas,pilihJk,pilihStatus)
        df2 = df.loc[df['KELAS'] == pilihKelas]
        df2 = df2.loc[df2['Hasil Prediksi'] == pilihStatus]
        df2 = df2.loc[df2['JENIS KELAMIN'] == pilihJk]
        lenDT = df2.axes[0]
        Akurasi = 'SISWA  '+pilihKelas+'  '+pilihJk+'  '+pilihStatus
        kdPDF = '3:1,2,3'

    elif (pilihJk == 'semuaJenis')and(pilihStatus == 'semuaHasil')and(pilihKelas == 'semuaKelas'):
        print('filterRedirect:',pilihKelas,pilihJk,pilihStatus)
        return redirect(url_for('data', opsi='filtering', usnm=usnm, sts=sts))

    else:
        print('filter204:',pilihKelas,pilihStatus,pilihJk)
        print('204 No Content Server berhasil memproses permintaan tapi tidak menampilkan konten apapun.')
        stsSvr = ['204','No Content','Server berhasil memproses permintaan tapi tidak menampilkan konten apapun.']
        return render_template('error.html', opsi='keluarDariElse', stsSvr=stsSvr, usnm=usnm, sts=sts)

    ygDiFilter = len(df2)
    df2.sort_values(by='KELAS', ascending=True, inplace=True)
    return render_template('data.html', ttl='filtering', data=df2, lenDT=lenDT, usnm=usnm, sts=sts, Akurasi=Akurasi, ygDiFilter=ygDiFilter, kdPDF=kdPDF)

@app.route('/cariData/', methods=['POST'])
def cari():
    usnm = request.form.get('usnm') # hidden
    sts = request.form.get('sts') # hidden
    cariNISN = request.form.get('cari')
    # cariData
    # perhatikanBoss kalauNilaiDidalamKolomString-tapi-isinya-anggakaDawaliNol'0076'->waktuDibacaPd.read_excel(tipeDatanyaJadiIntBoss)
    df = pd.read_excel(os.path.join(path,'dfHpredict_tomi.xlsx'))
    df['TANGGAL LAHIR'] = df['TANGGAL LAHIR'].astype('str')
    df.NISN = df.NISN.astype('str')
    # setNISN.str(10)
    nisnStr = []
    for values in df['NISN'].values:
        strVals = str(values).rjust(10, '0')
        nisnStr.append(strVals)
        # print(strVals)
    df['NISN'] = nisnStr
    df.NISN = df.NISN.astype('str')
    # print(df.info())

    print(cariNISN,type(cariNISN))
    dfNISN = df.NISN.values
    print(dfNISN[1])
    print(cariNISN in dfNISN)

    if (len(cariNISN) < 10):
        print('400 bad request kesalahan request disisi client')
        stsSvr = ['400','Bad Request','Kesalahan request disisi client']
        # return 'trialboss'+stsSvr[0]+stsSvr[1]+stsSvr[2]
        return render_template('error.html', opsi=cariNISN, stsSvr=stsSvr, usnm=usnm, sts=sts)

    datanya = df.loc[df['NISN'] == cariNISN]
    print(datanya)
    print(datanya.index)
    lenDT = datanya.axes[0]

    # iniBisa404
    if (cariNISN not in df['NISN'].values):
        print('404 not found request client tidak ditemukan')
        stsSvr = ['404', 'Not Found','Request client tidak ditemukan']
        return render_template('error.html', opsi=cariNISN, stsSvr=stsSvr, usnm=usnm, sts=sts)

    # udahPastiAdaDatanya
    # datanya.sort_values(by='nama', ascending=True, inplace=True) # belumPerlu
    return render_template('card.html', ttl='cariData', lenDT=lenDT, data=datanya, usnm=usnm, sts=sts)

@app.route('/dataMurid/<opsi>/<usnm>/<sts>', methods=['GET'])
def data(opsi,usnm,sts):
    print('-----------------------------------dataTrain, head(), info(), value_counts(), describe([LULUS,LULUS PRESTASI])')
    dataTrain = pd.read_excel(os.path.join(path,'dataTrain_tomi.xlsx'))
    dataTrain['TANGGAL LAHIR'] = dataTrain['TANGGAL LAHIR'].astype('str')
    encodingTrain = {'ABSEN':{'A':100, 'B':75, 'C':50, 'D':25}, 'LABEL':{'LULUS':0, 'LULUS PRESTASI':1}}
    dataTrain.replace(encodingTrain, inplace=True)
    print(dataTrain.head())
    print(dataTrain.info())
    print(dataTrain['LABEL'].value_counts())
    print(dataTrain[['UTS','UAS','UKK','TOTAL NILAI','ABSEN','LABEL']].loc[dataTrain['LABEL'] == 0].describe())
    print(dataTrain[['UTS','UAS','UKK','TOTAL NILAI','ABSEN','LABEL']].loc[dataTrain['LABEL'] == 1].describe())

    print('-----------------------------------setXTrain, concat([]), info() value_counts(LULUS, LULUS PRESTASI), describe()')
    dataTrain = dataTrain[['UTS','UAS','UKK','ABSEN','LABEL']]
    def setXls(xn_uts, xn_uas, xn_ukk, rtg):
        print(rtg,type(rtg))
        xls = []
        absn = [75, 50, 25]
        for nxt in range(0,500):
            nUTS, nUAS, nUKK, nAbsen = random.randint(xn_uts,90), random.randint(xn_uas,90), random.randint(xn_ukk,90), absn[random.randint(0,2)]
            y_n = nUTS + nUAS + nUKK
            if (y_n < 250):
                xls.append([nUTS, nUAS, nUKK, nAbsen, 0])

        return xls

    def setXprstsi(xn_uts, xn_uas, xn_ukk, rtg):
        print(rtg,type(rtg))
        xprsti = []
        for nxt in range(0,500):
            nUTS, nUAS, nUKK, nAbsen = random.randint(xn_uts,99), random.randint(xn_uas,99), random.randint(xn_ukk,99), 100
            y_n = nUTS + nUAS + nUKK
            if (y_n >= 250):
                xprsti.append([nUTS, nUAS, nUKK, nAbsen, 1])

        return xprsti

    while True:
        xtL = setXls(65,65,65,'L')
        print('ygLulus:',len(xtL))
        if (len(xtL) > 430):
            break

    while True:
        xtP = setXprstsi(80,80,80,'P')
        print('ygPrestasi:',len(xtP))
        if (len(xtP) >= 490):
            break

    dfxtL = pd.DataFrame(xtL, columns=['UTS','UAS','UKK','ABSEN', 'LABEL'])
    dfxtP = pd.DataFrame(xtP, columns=['UTS','UAS','UKK','ABSEN','LABEL'])
    print(dfxtL['LABEL'].value_counts())
    print(dfxtP['LABEL'].value_counts())
    # exit()

    dataTrain = pd.concat([dataTrain,dfxtL,dfxtP], axis=0)
    print(dataTrain.info())
    print(dataTrain['LABEL'].value_counts())
    print(dataTrain.describe())

    print('-----------------------------------train_test_split')
    X = dataTrain[['UTS','UAS','UKK','ABSEN']]
    y = dataTrain['LABEL']
    X_train,X_test,y_train,y_test = ms.train_test_split(X,y, test_size=1, random_state=0)
    print('X_train&y_train:', X_train.shape, y_train.shape, '|| X_test&y_test:', X_test.shape, y_test.shape)
    print(y_train.value_counts())

    # # StandardScalerDataTrain
    # scl = StandardScaler(copy=True, with_mean=True, with_std=True)
    # scl.fit(X_train)
    # X_train = scl.fit_transform(X_train)
    # print(X_train.min(), X_train.max())

    print('-----------------------------------dataTest, head(), info(), str(nisnDataBaru)')
    dataTest = pd.read_excel(os.path.join(path, 'dataTest_tomi.xlsx'))
    dataTest['TANGGAL LAHIR'] = dataTest['TANGGAL LAHIR'].astype('str')
    print(dataTest.head())
    print(dataTest.info())

    print('__cekApakahDatTestKosong....')
    if (dataTest.shape[0] == 0):
        print('409 Conflict, Server tidak bisa menyelesaikan permintaan karena terjadi konflik pada resource target')
        stsSvr = ['409','Conflict','Server tidak bisa menyelesaikan permintaan karena terjadi konflik pada resource target.']
        return render_template('error.html', opsi='hubungiTomi', stsSvr=stsSvr, usnm=usnm, sts=sts)

    print('__cekApakahBarisPertama==Kosong....')
    allDfNull = dataTest.isnull().all()
    if (allDfNull.iloc[0] == True):
        print('410 Gone, resource hilang dari server asal secara permanen dan tidak ada redirect')
        stsSvr = ['410','Gone','resource hilang dari server asal secara permanen dan tidak ada redirect.']
        return render_template('error.html', opsi='hubungiTomi', stsSvr=stsSvr, usnm=usnm, sts=sts)

    # nisnDataBaru
    inDataBaru = dataTest.NISN.axes[0].max()
    # print(inDataBaru, type(inDataBaru))
    nisnDataBaru = str(dataTest.NISN[inDataBaru]).rjust(10,'0')
    # print(nisnDataBaru, type(nisnDataBaru))
    print('nisnDataBaru: '+str(nisnDataBaru), type(str(nisnDataBaru)))

    print('-----------------------------------pengecekanKolomDataTest')
    nklm = 0
    klm = list(['NAMA PESERTA DIDIK','NISN','JENIS KELAMIN','TEMPAT LAHIR','TANGGAL LAHIR','AGAMA','KELAS','UTS','UAS','UKK','TOTAL NILAI','ABSEN'])
    tpdklm = list([['nama1 nama2'],10,['lk pr'],['tm  pyd'],['dd/mm/yy'],['islam'],['xiiatu XII ATU'],6,7,8,218,['A B C D']])
    while True:
        for nmklm in dataTest.columns:
            # (kesalahanNamaKolom(nmklm not in dataTest.columns)) or (jikaKolomLegkapTapiDataKosong(data.values != null)) or (kesalahanTipeData('NAMA PESERTA DIDIK' != str(), 'NISN' != int(), dst))
            if (nmklm not in klm) or (dataTest[klm[nklm]].dtypes != type(tpdklm[nklm])):
                print('406 Not Acceptable, Respon server bertentangan dengan nilai yang ditetapkan di header Accept.')
                stsSvr = ['406','Not Acceptable','Respon server bertentangan dengan nilai yang ditetapkan di header Accept.']
                return render_template('error.html', opsi='hubungiTomi', stsSvr=stsSvr, usnm=usnm, sts=sts)
                # break
            else:
                print(dataTest.axes[1][nklm])

            print('kolom_'+str(nklm)+':', nmklm in klm, dataTest.shape[0] != 0, dataTest[klm[nklm]].dtypes == type(tpdklm[nklm]))
            nklm = nklm + 1

        break

    print('-----------------------------------set_dfhPredict, value_counts(), describe()')
    dfHpredict = dataTest
    encodingTest = {'ABSEN':{'A':100, 'B':75, 'C':50, 'D':25}}
    dataTest.replace(encodingTest, inplace=True)

    X_test = dataTest[['UTS','UAS','UKK','ABSEN']]
    ytest = []
    for invals in range(0, len(X_test)):
        if (X_test['UTS'][invals] + X_test['UAS'][invals] + X_test['UKK'][invals] >= 118) and (X_test['ABSEN'][invals] == 100):
            ytest.append(1)
        else:
            ytest.append(0)

    y_test = pd.Series(ytest)
    print('X_test&y_test:',X_test.shape, y_test.shape)
    print(y_test.value_counts())
    print(dfHpredict.describe())

    print('-----------------------------------training model_machineLearning \n... \n... \n...')
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train,y_train)

    print('-----------------------------------hasilPrediksi, head(), info(), buatPDF(), to_excel()')
    y_prediksi = naive_bayes.predict(X_test)
    dfHpredict['Hasil Prediksi'] = y_prediksi
    encodingHpredict = {'ABSEN':{100:'A', 75:'B', 50:'C', 25:'D'}, 'Hasil Prediksi':{0:'LULUS', 1:'LULUS PRESTASI'}}
    dfHpredict.replace(encodingHpredict, inplace=True)
    # setNomorUrut
    noM = []
    for nmr in range(1, int(len(dfHpredict) + 1)):
        noM.append(nmr)
        # print(nmr)
    dfHpredict.insert(0, column='NO', value=noM)
    # setNISN.str(10)
    nisnStr = []
    for values in dfHpredict['NISN'].values:
        strVals = str(values).rjust(10, '0')
        nisnStr.append(strVals)
        # print(strVals)
    dfHpredict['NISN'] = nisnStr
    dfHpredict.NISN = dfHpredict.NISN.astype('str')
    dfHpredict['TANGGAL LAHIR'] = dfHpredict['TANGGAL LAHIR'].astype('str')
    print(dfHpredict.head())
    print(dfHpredict.info())
    # save
    dfHpredict.sort_values(by=['KELAS'], ascending=True, inplace=True)
    dfHpredict.to_excel(os.path.join(path, 'dfHpredict_tomi.xlsx'), sheet_name='hPredict', index=False)

    print('-----------------------------------classification_report')
    confusionmatrix = met.confusion_matrix(y_test, y_prediksi)
    print('=>Confusionmatrix:[TP[',confusionmatrix[0][0],']|FP[',confusionmatrix[0][1],']|FN[',confusionmatrix[1][0],']|TN[',confusionmatrix[1][1],']]')
    print(confusionmatrix)
    accuray = met.accuracy_score(y_test, y_prediksi)
    print('==> Accuracy:',accuray)
    precision = met.precision_score(y_test, y_prediksi) # hanyaUntukBinaryClassification
    print('==> Precision:',precision)
    sensitifity = met.recall_score(y_test, y_prediksi) # hanyaUntukBinaryClassification
    print('==> sensitifity:',sensitifity)
    report = met.classification_report(y_test, y_prediksi)
    print('==> Report:')
    print(report)

    if (opsi == 'Tambah'):
        flash('Data berhasil di '+opsi+'.. '+str(nisnDataBaru))
    elif (opsi == 'Ubah')or(opsi == 'Hapus'):
        flash('Data berhasil di '+opsi+'..')
    elif (opsi == 'inputFile'):
        flash('File sukses diupload Boss.. ')

    # settingSekaliLagi
    dfHpredict = pd.read_excel(os.path.join(path, 'dfHpredict_tomi.xlsx'))
    # setNISN.str(10)
    nisnStr = []
    for values in dfHpredict['NISN'].values:
        strVals = str(values).rjust(10, '0')
        nisnStr.append(strVals)
        # print(strVals)
    dfHpredict['NISN'] = nisnStr
    dfHpredict.NISN = dfHpredict.NISN.astype('str')
    dfHpredict['TANGGAL LAHIR'] = dfHpredict['TANGGAL LAHIR'].astype('str')
    lenDT = dfHpredict.axes[0]
    Akurasi = 'hasilPrediksi  Akurasi '+str(round(accuray, 3))+'%'
    return  render_template('data.html', ttl='dataMurid', data=dfHpredict, lenDT=lenDT, usnm=usnm, sts=sts, Akurasi=Akurasi, kdPDF='dataFull')

@app.route('/tambahData/<opsi>/<usnm>/<sts>', methods=['GET','POST'])
def tambahData(opsi,usnm,sts):
    if request.method == 'GET':
        return render_template('opsi.html', opsi=opsi, usnm=usnm, sts=sts)

    elif request.method == 'POST':
        nama = request.form.get('nma')
        nisn = str(random.randint(0000000000,9999999999))
        jk = request.form.get('jk')
        tLahir = request.form.get('tLahir')
        tglLahir = request.form.get('tglLahir')
        strTgll = str(tglLahir)
        print(strTgll)
        if ('/' in strTgll):
            splitTgll = strTgll.split('/') # dd/mm/yy
            tglLahir = splitTgll[2]+'-'+splitTgll[1]+'-'+splitTgll[0]
        elif ('-' in strTgll):
            splitTgll = strTgll.split('-') # yy-mm-dd
            tglLahir = splitTgll[0]+'-'+splitTgll[1]+'-'+splitTgll[2]
        else:
            print('411 Length Required, Request tidak menentukan header Content-Length yang diperlukan oleh resource.')
            stsSvr = ['411','Length Required','Request tidak menentukan header Content-Length yang diperlukan oleh resource.']
            return render_template('error.html', opsi='hubungiTomi', stsSvr=stsSvr, usnm=usnm, sts=sts)

        agama = request.form.get('agama')
        kelas = request.form.get('kls')
        uts = request.form.get('uts')
        uas = request.form.get('uas')
        ukk = request.form.get('ukk')
        absen = request.form.get('absen')

        # tambahData
        df = pd.read_excel(os.path.join(path, 'dataTest_tomi.xlsx'))
        df['TANGGAL LAHIR'] = df['TANGGAL LAHIR'].astype('str')
        df.NISN = df.NISN.astype('str')
        # print(df.info())
        nisnStr = []
        for values in df['NISN'].values:
            strVals = str(values).rjust(10, '0')
            nisnStr.append(strVals)
            # print(strVals)
        df['NISN'] = nisnStr
        df.NISN = df.NISN.astype('str')
        # print(df.info())

        maxIn = df.index.values.max()
        maxIn = maxIn + 1
        df.loc[maxIn] = [str(nama.upper()), str(nisn), str(jk), str(tLahir.upper()), str(tglLahir), str(agama), str(kelas), int(uts), int(uas), int(ukk), int(int(uts) + int(uas) + int(ukk)), str(absen)]
        print(df.loc[maxIn])
        # return 'trialBoss..'+str(nama.upper())+nisn+jk+str(tLahir.upper())+strTgll+agama+kelas+uts+uas+ukk+str(int(uts) + int(uas) + int(ukk))+absen
        # save
        df.to_excel(os.path.join(path, 'dataTest_tomi.xlsx'), sheet_name='dataTest', index=False)
        return  redirect(url_for('data', opsi=opsi, usnm=usnm, sts=sts))

@app.route('/ubahData/<opsi>/<nisn>/<usnm>/<sts>', methods=['GET','POST'])
def ubahData(opsi,nisn,usnm,sts):
    df = pd.read_excel(os.path.join(path, 'dataTest_tomi.xlsx'))
    df['TANGGAL LAHIR'] = df['TANGGAL LAHIR'].astype('str')
    df.NISN = df.NISN.astype('str')
    print(df.info())
    nisnStr = []
    for values in df['NISN'].values:
        strVals = str(values).rjust(10, '0')
        nisnStr.append(strVals)
        # print(strVals)
    df['NISN'] = nisnStr
    df.NISN = df.NISN.astype('str')
    # print(df.info())

    if request.method == 'GET':
        print(nisn, type(nisn))

        dfInd = df.loc[df['NISN'] == nisn]
        print(dfInd.NISN.values, type(dfInd.NISN.values))
        # valueForm
        nama = dfInd['NAMA PESERTA DIDIK'].values
        nisn = dfInd.NISN.values
        jk = dfInd['JENIS KELAMIN'].values
        tLahir = dfInd['TEMPAT LAHIR'].values
        tglLahir = dfInd['TANGGAL LAHIR'].values
        strTgll = str(tglLahir[0])
        print(strTgll)
        if ('/' in strTgll):
            splitTgll = strTgll.split('/') # dd/mm/yy
            tglLahir = splitTgll[2]+'-'+splitTgll[1]+'-'+splitTgll[0]
        elif ('-' in strTgll):
            splitTgll = strTgll.split('-') # yy-mm-dd
            tglLahir = splitTgll[0]+'-'+splitTgll[1]+'-'+splitTgll[2]
        else:
            print('411 Length Required, Request tidak menentukan header Content-Length yang diperlukan oleh resource.')
            stsSvr = ['411','Length Required','Request tidak menentukan header Content-Length yang diperlukan oleh resource.']
            return render_template('error.html', opsi='hubungiTomi', stsSvr=stsSvr, usnm=usnm, sts=sts)

        agama = dfInd['AGAMA'].values
        kelas = dfInd['KELAS'].values
        uts = dfInd['UTS'].values
        uas = dfInd['UAS'].values
        ukk = dfInd['UKK'].values
        absen = dfInd['ABSEN'].values

        print(nama,type(nama), nisn,type(nisn), jk,type(jk), tLahir,type(tLahir), tglLahir,type(tglLahir), agama,type(agama), kelas,type(kelas), uts,type(uts), uas,type(uas), ukk,type(ukk), absen,type(absen))
        optG = []
        for values in range(65,100):
            optG.append(values)

        print('optG:',optG)
        # perhatikanBos_ygArray
        return render_template('opsi.html', opsi=opsi, usnm=usnm, sts=sts, nama=nama[0], nisn=nisn[0], jk=jk[0], tLahir=tLahir[0], tglLahir=tglLahir, agama=agama[0], kelas=kelas[0], uts=uts[0], uas=uas[0], ukk=ukk[0], absen=absen[0], optG=optG)

    elif request.method == 'POST':
        nama = request.form.get('nma')
        jk = request.form.get('jk')
        tLahir = request.form.get('tLahir')
        tglLahir = request.form.get('tglLahir')
        strTgll = str(tglLahir)
        splitTgll = strTgll.split('-')
        tglLahir = splitTgll[0]+'-'+splitTgll[1]+'-'+splitTgll[2]

        agama = request.form.get('agama')
        kelas = request.form.get('kls')
        uts = request.form.get('uts')
        uas = request.form.get('uas')
        ukk = request.form.get('ukk')
        absen = request.form.get('absen')

        # perhatikanBos_ygArray
        print(nisn, type(nisn))

        dfInd = df.loc[df['NISN'] == nisn]
        print('cek len(dfInd):',len(dfInd))
        if (len(dfInd) > 1):
            print('302 found request dialihkan sementara dan akan direspon kembali dimasa mendatang')
            stsSvr = ['302','Found Request','double row NISN pada database']
            # return 'trialboss'+stsSvr[0]+stsSvr[1]+stsSvr[2]
            return render_template('error.html', opsi='hubungiTomi', stsSvr=stsSvr, usnm=usnm, sts=sts)

        print(dfInd.index)
        print(df.loc[dfInd.index])
        # print(df.loc[dfInd.index] == None)

        # ubahData
        print('_______niYangdihapusBoss')
        print(dfInd.loc[dfInd.index.values])
        df.drop(dfInd.index.values, inplace=True)
        print('_______niYangdiTambahBoss')
        maxIn = df.index.values.max()
        maxIn = maxIn + 1
        df.loc[maxIn] = [str(nama.upper()), str(nisn), str(jk), str(tLahir.upper()), str(tglLahir), str(agama), str(kelas), int(uts), int(uas), int(ukk), int(int(uts) + int(uas) + int(ukk)), str(absen)]
        print(df.loc[maxIn])
        # save
        df.to_excel(os.path.join(path, 'dataTest_tomi.xlsx'), sheet_name='dataTest', index=False)
        return redirect(url_for('data', opsi=opsi, usnm=usnm, sts=sts))

@app.route('/hapusData/<opsi>/<nisn>/<usnm>/<sts>', methods=['GET'])
def hapusData(opsi,nisn,usnm,sts):
    df = pd.read_excel(os.path.join(path,'dataTest_tomi.xlsx'))
    df['TANGGAL LAHIR'] = df['TANGGAL LAHIR'].astype('str')
    df.NISN = df.NISN.astype('str')
    # print(df.info())
    nisnStr = []
    for values in df['NISN'].values:
        strVals = str(values).rjust(10, '0')
        nisnStr.append(strVals)
        # print(strVals)
    df['NISN'] = nisnStr
    df.NISN = df.NISN.astype('str')
    # print(df.info())

    # perhatikanBos_ygArray
    print(nisn, type(nisn))
    dfInd = df.loc[df['NISN'] == nisn]
    print(dfInd)
    print(dfInd.index.values)

    # hapusData
    df.drop(dfInd.index.values, inplace=True)
    print('datanyaSudahDihapus_Boss..')
    # save
    df.to_excel(os.path.join(path, 'dataTest_tomi.xlsx'), sheet_name='dataTest', index=False)
    return redirect(url_for('data', opsi=opsi, usnm=usnm, sts=sts))

@app.route('/adminInputFile/', methods=['GET','POST'])
def adminInputFile():
    if request.method == 'GET':
        # bg = " 'https://images.unsplash.com/photo-1731410612759-d93cede4edbc?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D' "
        bg = " 'https://i.pinimg.com/originals/e8/c6/ae/e8c6aeb27686f608f20c770e906de13b.gif' "
        print(bg)
        print(bg[0:7])
        print(bg[-1:-7])
        return render_template('index.html', bg=bg, ttl='Admin')

    elif request.method == 'POST':
        usnm = request.form.get('usnm')
        pswd = request.form.get('pswd')

        # cekAkses
        users = ('muhammadTomi', 'muhammadTomi', 'muhammadTomi')
        pswdUser = (2011016, 2011016, 2011016)

        status = ('Admin', 'Admin', 'Admin')
        try:
            pswd = int(pswd)
        except Exception as e:
            flash('Username atau password salah..')
            return redirect(request.url)
        else:
            if int(pswd) not in pswdUser:
                flash('Password yang anda masukan salah..')
                # print('type(pswd):',type(pswd))
                # print('pswdUser[0]:',type(pswdUser[0]))
                return redirect(request.url)
            elif usnm not in users:
                flash('Username yang anda masukan salah..')
                # print('type(usnm):',type(usnm))
                # print('type(users[0]):',type(users[0]))
                return redirect(request.url)

            elif (usnm == users[0])and(pswd != pswdUser[0]):
                flash('Password yang anda masukan tidak sesuai..')
                return redirect(request.url)
            elif (usnm == users[1])and(pswd != pswdUser[1]):
                flash('Password yang anda masukan tidak sesuai..')
                return redirect(request.url)
            elif (usnm == users[2])and(pswd != pswdUser[2]):
                flash('Password yang anda masukan tidak sesuai..')
                return redirect(request.url)
            # else:
                # iniBisa404

        # finally:
            # pass

        # cekStatus
        if (usnm == users[0]):
            sts = status[0]
        elif (usnm == users[1]):
            sts = status[1]
        elif (usnm == users[2]):
            sts = status[2]

        return redirect(url_for('inputFile', opsi='inputFile', usnm=usnm, sts=sts))


@app.route('/inputFile/<opsi>/<usnm>/<sts>', methods=['GET','POST'])
def inputFile(opsi,usnm,sts):
    if request.method == 'GET':
        return render_template('opsi.html', opsi=opsi, usnm=usnm, sts=sts)

    elif request.method == 'POST':
        file = request.files['fileExcel'] #tangkap request.file user
        print(file)

        if file.filename == '': # jika Tidak Ada File yang dipilih siTomi
            flash ('Tidakada file yang dipilih')
            return redirect(request.url)

        elif file and allowed_file(file.filename): #jika AdaFile dan formatnya Ada Didalam Allowed_file()
            # amankan filename
            filename = secure_filename(file.filename)
            # cekNamaFile
            if (filename != 'dataTest_tomi.xlsx'):
                flash('Nama fileExcel tidak sesuai')
                return redirect(request.url)

            # simpan filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('upload_file filename: '+filename)
            # flash('File sukses diupload Boss.. '+filename)
            return redirect(url_for('data',opsi=opsi, usnm=usnm, sts=sts))

        else: # jika yg diupload bukan fileExcel
            flash('Format File tidak diizinkan')
            return redirect(request.url)

@app.route('/display/<jk>/<kls>')
def displayImage(jk,kls):
    print(jk,kls)
    if (kls == 'XII AMP'):
        if (jk == 'LAKI-LAKI'):
            dariGithub = 'amp.PNG'
        else:
            dariGithub = 'amp2.PNG'

    elif (kls == 'XII APHP'):
        if (jk == 'LAKI-LAKI'):
            dariGithub = 'aphp.PNG'
        else:
            dariGithub = 'aphp2.PNG'

    elif (kls == 'XII ATP'):
        if (jk == 'LAKI-LAKI'):
            dariGithub = 'atp.PNG'
        else:
            dariGithub = 'atp2.PNG'

    elif (kls == 'XII ATU'):
        if (jk == 'LAKI-LAKI'):
            dariGithub = 'atu.png'
        else:
            dariGithub = 'atu2.png'

    else:
        if (jk == 'LAKI-LAKI'):
            dariGithub = 'tsm.PNG'
        else:
            dariGithub = 'tsm2.PNG'

    return redirect(url_for('static', filename='students/'+dariGithub))

@app.route('/download_pdf/<Akurasi>/<kdPDF>')
def download_pdf(Akurasi, kdPDF):
    df = pd.read_excel(os.path.join(path, 'dfHpredict_tomi.xlsx'))
    df['TANGGAL LAHIR'] = df['TANGGAL LAHIR'].astype('str')
    df.NISN = df.NISN.astype('str')
    # hapusKolomNomorUrut
    df.drop(['NO'], axis=1, inplace=True)
    print(df.info())

    # seWarna_Kelas&Hasil_untukPDF
    Akurasi = Akurasi.split('  ')
    print(Akurasi, len(Akurasi))
    print('kdPDF:',kdPDF)

    # filtering_pilihSatu
    if (kdPDF == '1:3'):
        df = df.loc[df['Hasil Prediksi'] == Akurasi[1]]
    elif (kdPDF == '1:1'):
        df = df.loc[df['KELAS'] == Akurasi[1]]
    elif (kdPDF == '1:2'):
        df = df.loc[df['JENIS KELAMIN'] == Akurasi[1]]

    # filtering_pilihDua
    elif (kdPDF == '2:1,3'):
        df = df.loc[df['KELAS'] == Akurasi[1]]
        df = df.loc[df['Hasil Prediksi'] == Akurasi[2]]
    elif (kdPDF == '2:1,2'):
        df = df.loc[df['KELAS'] == Akurasi[1]]
        df = df.loc[df['JENIS KELAMIN'] == Akurasi[2]]
    elif (kdPDF == '2:2,3'):
        df = df.loc[df['JENIS KELAMIN'] == Akurasi[1]]
        df = df.loc[df['Hasil Prediksi'] == Akurasi[2]]

    # filtering_pilihTiga
    elif (kdPDF == '3:1,2,3'):
        df = df.loc[df['KELAS'] == Akurasi[1]]
        df = df.loc[df['JENIS KELAMIN'] == Akurasi[2]]
        df = df.loc[df['Hasil Prediksi'] == Akurasi[3]]

    else:
        print(' dataFull... \n dataFull... \n dataFull...')

    # setNISN.str(10)
    nisnStr = []
    for values in df['NISN'].values:
        strVals = str(values).rjust(10, '0')
        nisnStr.append(strVals)
        # print(strVals)
    df['NISN'] = nisnStr
    # setNomorUrut
    noM = []
    for nmr in range(1, int(len(df) + 1)):
        noM.append(nmr)
        # print(nmr)
    df.insert(0, column='NO', value=noM)
    print(df.info())

    # buildPDF
    buatPDF(df, Akurasi, kdPDF)

    print('__buatPDF(dfHpredict_tomi.xlsx)')
    df.sort_values(by=['KELAS'], ascending=True, inplace=True)
    return send_file(os.path.join(path, 'dataHasilPrediksi_tomi.pdf'), as_attachment=True)

if __name__=='__main__':
    app.run(debug=True)
