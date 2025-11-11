def clmrange(df, clm):
    crange = df[clm].loc[df[clm] != 0].unique()
    for i in df[clm].loc[df[clm] == 0].index:
        df.loc[i, clm] = random.choice(crange)
        # endofor
    return df

def persiapan_data(df):
    df.fillna(0, inplace=True)
    df['kode_obat'] = [str(100000+i).replace('1','b', 1) for i in range(1, len(df)+1)]
    df[['harga_beli','harga_jual','stok']] = df[['harga_beli','harga_jual','stok']].round().astype('int')
    df['usia'] = df['usia'].replace(['Anak, bayi','Bayi & Anak','Bayi, Anak'],'Bayi-Anak')
    df['usia'] = df['usia'].replace(['Dewasa dan Anak','Dewasa & Anak','Dewasa & Anak remaja','Dewasa & Anak,remaja','Anak & Dewasa','Anak, Remaja &  Dewasa'],'Anak-Dewasa')
    df['usia'] = df['usia'].replace(['Dewasa,remaja','Dewasa ,remaja','Dewasa, remaja','Remaja & Dewasa','Remaja &  Dewasa','Dewasa & Remaja'],'Remaja-Dewasa')
    df['usia'] = df['usia'].replace(['Dewasa ', 'Dewasa Pria'],'Dewasa')
    df['usia'] = df['usia'].replace({'Dewasa & Lansia':'Dewasa-Lansia', 'Bayi, Anak, Remaja &  Dewasa':'Semua Usia'})
    # df.rename(columns={'jenis':'manfaat'}, inplace=True)

    # ======= preprocessing data kosong banyak kolom
    listclm = []
    for i in range(0, len(df.columns)):
        if (i==3):
            continue
        # print(i)
        listclm.append(df.columns[i])
        # endfor
    # print(listclm)
    for clm in listclm:
        if(0 in df[clm].values):
            df = clmrange(df, clm)
            # endif
        print(clm, (0 in df[clm].values))
        # endfor

    # ======= preprocessing data kosong kolom harga jual + 100000
    for i in df.index:
        hbeli = int(df.loc[i]['harga_beli'])
        hjual = int(hbeli+5000)
        df.loc[i, 'harga_jual'] = hjual
        # endofor

    # ======= preprocessing data selesai...
    print(df.info())
    print(df.describe())
    print(df)
    df.to_excel(os.path.join(os.getcwd(),'fix-obat1.xlsx'), sheet_name='fix-obat1', index=False)

def persiapan_data2(df):
    print(df.info())
    # df['cbeli'] = [1 if(i <= 50000) else 2 if (i<=100000) else 3 for i in df['harga_beli']]
    # print(df['cbeli'].value_counts())
    # df['cjual'] = [1 if(i <= 50000) else 2 if (i<=100000) else 3 for i in df['harga_jual']]
    # print(df['cjual'].value_counts())
    # df['cstok'] = [1 if(i <= 80) else 2 if (i<=150) else 3 for i in df['stok']]
    # print(df['cstok'].value_counts())
    sakral = []
    for i in range(0, len(df)-200):
        sakral.append(random.randint(0, len(df)))
        # endofor
    # print(sakral)
    df['ketersediaan'] = ['Stabil' if(i in sakral) else 'Menurun' for i in df.index]
    print(df['ketersediaan'].value_counts())
    df.to_excel(os.path.join(os.getcwd(),'fix-obat1.xlsx'), sheet_name='fix-obat1', index=False)
    # ======= preprocessing data2 selesai...

def prediksi_nb(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    model_nb = GaussianNB()
    model_nb.fit(X_train,y_train)
    score = model_nb.score(X_test, y_test)
    y_predict = model_nb.predict(X_test)
    import sklearn.metrics as met
    report = met.classification_report(y_test, y_predict)
    return score, report

def prediksi_ds(X_train, X_test, y_train, y_test):
    import sklearn.tree as tree
    model_ds = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    model_ds.fit(X_train, y_train)
    score = model_ds.score(X_test, y_test)
    y_predict = model_ds.predict(X_test)
    import sklearn.metrics as met
    report = met.classification_report(y_test, y_predict)
    return score, report

def ketersediaanBarang():
    df = pd.read_excel(os.path.join(os.getcwd(),'fix-obat1.xlsx'))
    df.replace({'ketersediaan':{'inStock':0, 'lowStock':1}}, inplace=True)
    X = df[['harga_beli','harga_jual','stok']]
    y = df['ketersediaan']
    import sklearn.model_selection as ms
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0)
    scorenb, reportnb = prediksi_nb(X_train, X_test, y_train, y_test)
    scoreds, reportds = prediksi_ds(X_train, X_test, y_train, y_test)
    print(reportnb)
    print(reportds)
    # ======= prediksi ketersediaan barang selesai...

if __name__=='__main__':
    import pandas as pd
    import numpy as np
    import os, random

    # ======= tahap1
    df = pd.read_excel(os.path.join(os.getcwd(), 'obat1.xlsx'))
    persiapan_data(df)

    # ======= tahap2
    df2 = pd.read_excel(os.path.join(os.getcwd(), 'fix-obat1.xlsx'))
    persiapan_data2(df2)

    # ======= tahap3
    # ketersediaanBarang()
