from flask import Flask, render_template, url_for, redirect, request, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import random
import os

app = Flask(__name__)
root = os.getcwd()
path = os.path.join(root,) #'mysite'
app.secret_key = 'TA_PutriamaMelatiSukma_2111021'
app.config['UPLOAD_FOLDER'] = os.path.join(path, 'static')
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

def nominalrupiah(total):
    if (int(total) < 10000): # (max 9.999)
        satuan = total[0:1]
        ribuan = total[1:]
        return satuan+'.'+ribuan
    elif (int(total) < 100000): # (max 99.999)
        puluhan = total[0:2]
        ribuan = total[2:]
        return puluhan+'.'+ribuan
    elif (int(total) < 1000000): # (max 999.999)
        ratusan = total[0:3]
        ribuan = total[3:]
        return ratusan+'.'+ribuan
    elif (int(total) < 10000000): # (max 9.999.999)
        jutaan = total[0:1]
        ratusan = total[1:4]
        ribuan = total[4:]
        return jutaan+'.'+ratusan+'.'+ribuan
    elif (int(total) < 100000000): # (max 99.999.999)
        jutaan = total[0:2]
        ratusan = total[2:5]
        ribuan = total[5:]
        return jutaan+'.'+ratusan+'.'+ribuan
    elif (int(total) < 1000000000): # (max 999.999.999)
        jutaan = total[0:3]
        ratusan = total[3:6]
        ribuan = total[6:]
        return jutaan+'.'+ratusan+'.'+ribuan
    else:
        return '-'

def dataobat():
    df = pd.read_excel(os.path.join(path, 'fix-obat1.xlsx'))
    return df

def ringkasandataobat(df):
    r0 = len(df)
    r1 = len(df['jenis'].value_counts())
    r2 = len(df['ketersediaan'].loc[df['ketersediaan'] == 'Stabil'])
    r3 = len(df['ketersediaan'].loc[df['ketersediaan'] == 'Menurun'])
    r4 = df['stok'].sum()
    r5 = nominalrupiah(str(df['harga_beli'].sum()))
    return [r0, r1, r2, r3, r4, r5]

def valueint(nidt):
    try:
        nidtint = int(nidt)
    except Exception as e:
        return False
    return nidtint

def readdata(crgex):
    df = dataobat()
    # cek operator pencarian
    if ('=' in crgex):
        nmclm = crgex.split('=')[0]
        nidt = crgex.split('=')[-1]
        if (nmclm not in df.columns):
            df2 = df.loc[df['kode_obat'] == nidt]
            return df2, crgex, False
        if (not valueint(nidt)):
            df2 = df.loc[df['kode_obat'] == nidt]
            return df2, crgex, False
        else:
            nidtint = valueint(nidt)
            df2 = df.loc[df[nmclm] == nidtint]
            return df2, crgex, True
    elif ('>' in crgex):
        nmclm = crgex.split('>')[0]
        nidt = crgex.split('>')[-1]
        if (nmclm not in df.columns):
            df2 = df.loc[df['kode_obat'] == nidt]
            return df2, crgex, False
        if (not valueint(nidt)):
            df2 = df.loc[df['kode_obat'] == nidt]
            return df2, crgex, False
        else:
            nidtint = valueint(nidt)
            df2 = df.loc[df[nmclm] > nidtint]
            return df2, crgex, True
    elif ('<' in crgex):
        nmclm = crgex.split('<')[0]
        nidt = crgex.split('<')[-1]
        if (nmclm not in df.columns):
            df2 = df.loc[df['kode_obat'] == nidt]
            return df2, crgex, False
        if (not valueint(nidt)):
            df2 = df.loc[df['kode_obat'] == nidt]
            return df2, crgex, False
        else:
            nidtint = valueint(nidt)
            df2 = df.loc[df[nmclm] < nidtint]
            return df2, crgex, True
    # cek apakah input adadidalam baris data
    else:
        if (crgex in df.values):
            for nmclm in df.columns:
                df2 = df.loc[df[nmclm] == crgex]
                if (len(df2) == 0):
                    continue
                else:
                    return df2, crgex, True
        else:
            df2 = df.loc[df['kode_obat'] == crgex]
            return df2, crgex, False

def fbg():
    fnfonts = ['Caveat','Roboto Slab', 'Cedarville Cursive','Dancing Script',
                'Great Vibes','Lobster','Playwrite US Modern','Roboto Slab']
    fonts = fnfonts[random.randint(0,len(fnfonts)-1)]
    fnbgimg = ['pd'+str(i)+'.jpg' for i in range(1,16)]
    bgimg = fnbgimg[random.randint(0,len(fnbgimg)-1)]
    return fonts, bgimg, 'username / password salah'

@app.route('/', methods=['GET','POST'])
def home():
    if (request.method == 'POST'):
        usnm = request.form.get('usnm')
        pswd = request.form.get('pswd')
        if (usnm == 'sukma')and(pswd=='2111021'):
            return render_template('loading.html', fonts=fbg()[0], bgimg=fbg()[1])
        else:
            return render_template('index.html', fonts=fbg()[0], bgimg=fbg()[1], salah=fbg()[2])
    else:
        return render_template('index.html', fonts=fbg()[0], bgimg=fbg()[1])

@app.route('/sukma/', methods=['GET','POST'])
def admin():
    if request.method == 'GET':
        ringkasan = ringkasandataobat(dataobat())
        return render_template('admin.html', df=dataobat(), ringkasan=ringkasan, hpcdt=True,
                                scrdt='Semua Data')
    else:
        crgex = request.form.get('crdt')
        return redirect(url_for('caridata', crgex=crgex))

@app.route('/sukma/caridata/<crgex>')
def caridata(crgex):
    df, scrdt, hpcdt = readdata(crgex)
    print(df.info(), scrdt, hpcdt)
    ringkasan = ringkasandataobat(df)
    return render_template('admin.html', df=df, ringkasan=ringkasan, hpcdt=hpcdt, scrdt=scrdt)

if __name__ == '__main__':
    app.run(debug=True)
