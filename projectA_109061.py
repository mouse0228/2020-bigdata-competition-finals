# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:36:23 2020

@author: 王宏彬
"""

import os 
import pandas as pd
import numpy as np
from keras.utils import Progbar
import warnings
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings("ignore")

os.chdir("/projectA/Train")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_train_data():
    dir1 = "/projectA/Train/"
    all_path = next(os.walk("/projectA/Train/"))[2]
    df = pd.read_csv(dir1+all_path[0]).astype(np.float32)
    a = Progbar(len(all_path))
    a.update(1)
    for i in range(len(all_path)-1):
        df = df.merge(pd.read_csv(dir1+all_path[i+1]).astype(np.float32),how = 'outer')
        a.update(i+2)
    dir2 = "/projectA/Train2/"
    all_path_2 = next(os.walk("/projectA/Train2/"))[2]
    b = Progbar(len(all_path_2))
    for i in range(len(all_path_2)):
        df = df.merge(pd.read_csv(dir2+all_path_2[i]).drop(['SeqNo'],axis=1).astype(np.float32),how = 'outer')
        b.update(i+1) 
    df = df[df['F_1']>2495][df[df['F_1']>2495]['F_1']<2505].merge(df[df['F_1']<5],how = 'outer').merge(
        df[df['F_1']>4995][df[df['F_1']>4995]['F_1']<5005],how = 'outer').merge(
        df[df['F_1']>7495][df[df['F_1']>7495]['F_1']<7505],how = 'outer').merge(
        df[df['F_1']>9995][df[df['F_1']>9995]['F_1']<10005],how = 'outer')
    y = df[['O1','O2','O3']]
    df = df.drop(['O1','O2','O3'],axis=1)
    print(df['F_6'].value_counts())
    print(df['F_8'].value_counts())
    print(df['F_9'].value_counts())
    print(df['F_10'].value_counts())
    print(df['F_15'].value_counts())
    print(df['F_16'].value_counts())
    print(df['F_17'].value_counts())
    print(df['F_20'].value_counts())
    print(df['F_21'].value_counts()) 
    df = df.drop(['SeqNo','F_6','F_8','F_9','F_10','F_15','F_16','F_17','F_20','F_21'],axis=1)  
    return df,y


def read_test_data():
    dir_test = "/projectA/Test/"
    all_path = next(os.walk("/projectA/Test/"))[2]
    df = pd.read_csv(dir_test+all_path[0]).astype(np.float32)
    a = Progbar(len(all_path))
    a.update(1)
    for i in range(len(all_path)-1):
        df = df.merge(pd.read_csv(all_path[i+1]).astype(np.float32),how = 'outer')
        a.update(i+2)    
    df = df.drop(['SeqNo','F_6','F_8','F_9','F_10','F_15','F_16','F_17','F_20','F_21'],axis=1)    
    return df



(df,y) = read_train_data()
df2 = read_test_data()



#######################################################################################################

rnd_clf = RandomForestRegressor(max_features=5,n_estimators=36,max_depth=22,min_samples_leaf=1)
rnd_clf.fit(df, y['O1'])

rnd_clf2 =RandomForestRegressor(max_features=4,n_estimators=400,max_depth=25,min_samples_leaf=1)
rnd_clf2.fit(df, y['O2'])

rnd_clf3 = RandomForestRegressor(max_features=3,n_estimators=200,max_depth=20,min_samples_leaf=1)
rnd_clf3.fit(df, y['03'])

#######################################################################################################



Y_test1 = rnd_clf.predict(df2)
Y_test2 = rnd_clf2.predict(df2)
Y_test3 = rnd_clf3.predict(df2)

Y_test = pd.read_csv('/home/109061/projectA_template.csv')

Y_test['O1_P'] = Y_test1
Y_test['O2_P'] = Y_test2
Y_test['O3_P'] = Y_test3

Y_test.to_csv("/home/109061/submit/109061_projectA_test.csv",sep=',',index=0)

