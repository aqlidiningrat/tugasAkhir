# from flask import Flask, render_templates, request, send_file
# import os
# app = Flask(__name__, folder_template='templates')
# @app.route('/')
# def home():
#     return render_templates('index.html')
# @app.route('/download/')
# def download():
#     return send_file(os.path.join(root,'dataTest_tomi.xlsx'), as_attachment=True)

# # _________setImage
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# root = os.getcwd()
# path = os.path.join(root,'static','tmi_192x192.png')
# img = np.asarray((Image.open(path)))
# # print(img)
# # print(repr(img))
# # img_plot = plt.imshow(img)
# # plt.show()
# lum_img = img[:,:,0] # ExtrakcingForChannelGrayscale
# # plt.figure()
# # plt.figsize(5,5)
# plt.imshow(lum_img, cmap='Reds_r')
# plt.axis('off')
# # plt.show()
# # plt.savefig(os.path.join(root,'static','tmiRed.png'), bbox_inches='tight', transparent="True", pad_inches=0)
# # plt.savefig(IMG_DIR + 'match.png',bbox_inches='tight', transparent="True", pad_inches=0)

import pandas as pd
import random
import os

from sklearn.preprocessing import StandardScaler
import sklearn.metrics as met
import sklearn.model_selection as ms
from sklearn.naive_bayes import GaussianNB

from fpdf import FPDF
from fpdf.fonts import FontFace

root = os.getcwd()

print('-----------------------------------dataTrain, head(), info(), describe([LULUS,LULUS PRESTASI])')
dataTrain = pd.read_excel(os.path.join(root,'dataTrain_tomi.xlsx'))
dataTrain['TANGGAL LAHIR'] = dataTrain['TANGGAL LAHIR'].astype('str')
encodingTrain = {'ABSEN':{'A':100, 'B':75, 'C':50, 'D':25}, 'LABEL':{'LULUS':0, 'LULUS PRESTASI':1}}
dataTrain.replace(encodingTrain, inplace=True)
print(dataTrain.head())
print(dataTrain.info())
print(dataTrain['LABEL'].value_counts())
print(dataTrain[['UTS','UAS','UKK','TOTAL NILAI','ABSEN','LABEL']].loc[dataTrain['LABEL'] == 0].describe())
print(dataTrain[['UTS','UAS','UKK','TOTAL NILAI','ABSEN','LABEL']].loc[dataTrain['LABEL'] == 1].describe())

print('-----------------------------------setXTrain concat([]), info(), value_counts(LULUS, LULUS PRESTASI), describe()')
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

# print('-----------------------------------dataTest head(), info()')
# dataTest = pd.read_excel(os.path.join(root, 'dataTest_tomi.xlsx'))
# print(dataTest.head())
# print(dataTest.info())
#
# print('__cekApakahDatTestKosong.....')
# if (dataTest.shape[0] == 0):
#     print('409 Conflict, Server tidak bisa menyelesaikan permintaan karena terjadi konflik pada resource target')
#     exit()
#
# print('__cekApakahBarisPertama==Kosong....')
# allDfNull = dataTest.isnull().all()
# if (allDfNull.iloc[0] == True):
#     print('410 Gone, resource hilang dari server asal secara permanen dan tidak ada redirect')
#     exit()
#
# print('-----------------------------------pengecekanKolomDataTest')
# nklm = 0
# klm = list(['NAMA PESERTA DIDIK','NISN','JENIS KELAMIN','TEMPAT LAHIR','TANGGAL LAHIR','AGAMA','KELAS','UTS','UAS','UKK','TOTAL NILAI','ABSEN'])
# tpdklm = list([['nama1 nama2'],10,['lk pr'],['tm  pyd'],['dd/mm/yy'],['islam'],['xiiatu XII ATU'],6,7,8,218,['A B C D']])
# while True:
#     for nmklm in dataTest.columns:
#         # (kesalahanNamaKolom(nmklm not in dataTest.columns)) or (jikaKolomLegkapTapiDataKosong(data.values != null)) or (kesalahanTipeData('NAMA PESERTA DIDIK' != str(), 'NISN' != int(), dst))
#         if (nmklm not in klm) or (dataTest.shape[0] == 0) or (dataTest[klm[nklm]].dtypes != type(tpdklm[nklm])):
#             break
#         else:
#             print(dataTest.axes[1][nklm])
#
#         print('kolom_'+str(nklm)+':', nmklm in klm, dataTest.shape[0] != 0, dataTest[klm[nklm]].dtypes == type(tpdklm[nklm]))
#         nklm = nklm + 1
#
#     break

print('__set_Xtest...random')
set_Xtest = []
for nilaiTest in range(0,50):
    absn = [100, 75, 50, 25]
    nUTS, nUAS, nUKK, nAbsen = random.randint(60,69), random.randint(70,79), random.randint(80,99), absn[random.randint(0,1)]
    y_n = nUTS + nUAS + nUKK
    if (y_n >= 218)and(nAbsen == 100):
        set_Xtest.append([nUTS, nUAS, nUKK, y_n, nAbsen, 1])
    elif (y_n >= 218):
        set_Xtest.append([nUTS, nUAS, nUKK, y_n, nAbsen, 0])
    else:
        set_Xtest.append([nUTS, nUAS, nUKK, y_n, nAbsen, 0])

Xtest = pd.DataFrame(set_Xtest, columns=['UTS','UAS','UKK','totalNilai','ABSEN','Seharusnya'])

# print('__test_manual...')
# set_Xtest = [['eviDariati',90,90,90,270,'A','LULUS PRESTASI'],['eviDariati2',80,90,90,260,'B','LULUS PRESTASI'],['eviDariati3',80,90,80,250,'B','LULUS PRESTASI'],['eviDariati4',90,80,80,250,'C','LULUS PRESTASI'],['eviDariati5',71,86,82,249,'A','LULUS'],['eviDariati6',88,89,72,249,'B','LULUS'],['eviDariati7',88,98,63,249,'C','LULUS'],['eviDariati8',76,68,66,210,'A','LULUS']]
# set_Xtest = [['eviDariati',24,62,95,92,'A','LULUS PRESTASI']]
# Xtest = pd.DataFrame(set_Xtest, columns=['NAMA','UTS','UAS','UKK','totalNilai','ABSEN','labelTest'])

print('-----------------------------------set_dfhPredict, value_counts(), describe()')
# dfHpredict = dataTest
# encodingTest = {'ABSEN':{'A':100, 'B':75, 'C':50, 'D':25}}
# dataTest.replace(encodingTest, inplace=True)
# print(dataTest[['UTS','UAS','UKK','TOTAL NILAI','ABSEN']].describe())

dfHpredict = Xtest

X_test = Xtest[['UTS','UAS','UKK','ABSEN']]
ytest = []
for invals in range(0, len(X_test)):
    if (X_test['UTS'][invals] + X_test['UAS'][invals] + X_test['UKK'][invals] >= 218) and (X_test['ABSEN'][invals] == 100):
        ytest.append(1)
    else:
        ytest.append(0)

y_test = pd.Series(ytest)
print('X_test&y_test:',X_test.shape, y_test.shape)
print(y_test.value_counts())
print(dfHpredict.describe())

# # StandardScalerDataTest
# # X_test = scl.fit_transform(Xtest[['UTS','UAS','UKK','ABSEN']])
# # print(X_test.min(), X_test.max())

print('-----------------------------------training model_machineLearning \n... \n... \n...')
naive_bayes = GaussianNB()
naive_bayes.fit(X_train,y_train)

print('-----------------------------------hasilPrediksi, head(), info(), buatPDF(), to_excel()')
y_prediksi = naive_bayes.predict(X_test)
dfHpredict['Hasil Prediksi'] = y_prediksi
encodingHpredict = {'ABSEN':{100:'A', 75:'B', 50:'C', 25:'D'}, 'Seharusnya':{0:'LULUS', 1:'LULUS PRESTASI'}, 'Hasil Prediksi':{0:'LULUS', 1:'LULUS PRESTASI'}}
dfHpredict.replace(encodingHpredict, inplace=True)
# setNomorUrut
noM,smplNM = [],[]
for nmr in range(1, int(len(dfHpredict) + 1)):
    noM.append(nmr)
    smplNM.append('namaTest'+str(nmr))

dfHpredict.insert(0, column='NO', value=noM)
dfHpredict.insert(1, column='NAMA', value=smplNM)

# # setNISN.str(10)
# nisnStr = []
# for values in dfHpredict['NISN'].values:
#     strVals = str(values).rjust(10, '0')
#     nisnStr.append(strVals)
#     # print(strVals)
# dfHpredict['NISN'] = nisnStr
# dfHpredict.NISN = dfHpredict.NISN.astype('str')
# dfHpredict['TANGGAL LAHIR'] = dfHpredict['TANGGAL LAHIR'].astype('str')
# print(dfHpredict.head())
# print(dfHpredict.info())
# # save
# dfHpredict.to_excel(os.path.join(root, 'dfHpredict_tomi.xlsx'), sheet_name='hPredict', index=False)

print(dfHpredict.head())

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
        self.pdf.cell(0, 10, self.title, align="C")
        self.pdf.ln()  # Add a line break

    def add_table(self):
        self.pdf.set_font("Times", style="B", size=7)
        """Adds the table from the DataFrame to the PDF."""
        # colorBacgroundHeading
        white = (255, 255, 255)
        # colorTextHeading
        grey = (127, 130, 131)

        # Create the table
        headings_style = FontFace(emphasis="ITALICS", color=white, fill_color=grey, size_pt=8)
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
pdf_table = PDFTable(title="TEST_FPDF2", data=df)
# Add title and table to the PDF
pdf_table.add_title()
pdf_table.add_table()
# Save the PDF to a file
pdf_table.save_pdf("fpdf2_test.pdf")
