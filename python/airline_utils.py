import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import metrics, linear_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import itertools
from PIL import Image
import os

from matplotlib import cm
def getcmaprgb(N, cmap):
    """Get the RGB values of N colors across a colormap"""
    return cmap(np.linspace(0,255,N).astype(int))

def plot_avg_err(data, y='dep_delay', by='carrier', of='dow', labels=[], car = ['DL','EV','WN', 'AA']):
    """Function to plot the data[y] on data[of], grouped by data[by], foor values of by in car
    Plot average and std dev.
    """
    dft = data[[by,of,y]].copy()
    df_agg = dft.groupby([by,of]).agg(['mean','sem'])
    
    plt.figure(figsize=(10,5))  

    size = len(labels)
    colors = getcmaprgb(len(car),cm.nipy_spectral)
    
    df_agg0 = dft.groupby([of]).agg(['mean','sem'])
    mm = df_agg0[y]['mean'].values
    ss = df_agg0[y]['sem'].values
    x = df_agg0.index
    plt.errorbar(x,mm,yerr=ss,
             label='all',linewidth=3)
    for i, c in enumerate(car):
        mm = df_agg.loc[c][y]['mean'].values
        ss = df_agg.loc[c][y]['sem'].values
        x = df_agg.loc[c].index
        plt.errorbar(x,mm,yerr=ss,
                 color=colors[i],label=c,linewidth=3)
        
    plt.ylabel(y, size=20)
    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.xlabel(of)
    plt.xticks(np.arange(min(df_agg.loc[c].index),min(df_agg.loc[c].index)+size),labels,size=15)
    plt.legend(loc='best',fontsize=15,frameon=True)

    ax = plt.gca()
    ax.grid(True)
    for line in ax.get_xgridlines():
        line.set_linewidth(0)
    for line in ax.get_ygridlines():
        line.get_ydata
        line.set_linewidth(1)

def make_onehot_feat_dict_from_vals(df, feat_key, feat_name, levels):
    '''
    Function to transform features by one hot encoding, by discretizing using levels
    '''
    # Create keys
    N_feat = len(levels) - 1
    keys = [0]*N_feat
    for i in range(N_feat):
        keys[i] = 'f_'+feat_name+'_'+ str(levels[i])
    # Find the indices for each class
    feat_dict = {}
    for i in range(N_feat):
        feat_dict[keys[i]] = np.transpose(
                    np.logical_and(df[feat_key].values>= levels[i],
                                df[feat_key].values<levels[i+1]))
    return feat_dict


def make_onehot_feat_dict(df, feat_key, feat_name):    
    '''Function to transform categorical features by one hot encoding
    '''
    feat_vals = df[feat_key].values
    all_vals = np.unique(feat_vals)
    N_vals = len(all_vals)
    N_feat = N_vals - 1

    # Create keys
    keys = [0]*N_feat
    for i in range(N_feat):
        keys[i] = 'f_'+feat_name+'_'+ str(all_vals[i])

    # Create value for each training example in dict
    feat_dict = {}
    for i, k in enumerate(keys):
        this_day = all_vals[i]
        feat_dict[k] = feat_vals == this_day
    return feat_dict

def make_pics_transparent():
    '''Function which makes picture from 
    ../pics folder transparents and copies 
    it into ../pics_transparent

    '''
    for pic in os.listdir('../output/pics/'):
        img = Image.open('../output/pics/'+pic)
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save("../output/pics_transp/"+pic, "PNG")        
    
def load_data(path, origin='any',carrier='any'):
    """Function which reads all csvs from folder into 1 dataframe. optionally only origins or carriers as inputs specify.
    """
    files = os.listdir(path)
    files
    list_pds=[]
    for f in files:
        p = pd.read_csv(path + f)
        if origin != 'any':
            p = p.loc[(p['ORIGIN'] == origin)]
        if carrier != 'any':
            p = p.loc[(p['UNIQUE_CARRIER'] == carrier)]
        list_pds.append(p)
    data = pd.concat(list_pds)
    data = preprocess_data(data, org=origin)
    return data

def preprocess_data(data, org='any', dest ='any'):
    """Function to 
    - remoce cancelled and diverted flights
    - convert the fieldnames into lower case names used in the notebook
    - filter if origin or  dest inputs specify
    - convert departure and arrival time to short forms (only hours)
    """
    data.rename(columns={'CRS_ARR_TIME':'scheduled_arr_time','ARR_TIME':'arr_time',
                         'DAY_OF_WEEK':'dow','DAY_OF_MONTH':'dom','DISTANCE':'distance',
                         'MONTH':'month','YEAR':'year','CRS_DEP_TIME':'scheduled_dep_time','DEP_DELAY':'dep_delay',
                         'UNIQUE_CARRIER':'carrier','ORIGIN':'origin','DEST':'dest','CANCELLED':'cancelled',
                         'DIVERTED':'diverted'},inplace=True)

    # remove flights which are diverted or canceled => ArrTime is 0
    data_out = data.loc[~data['arr_time'].isnull()].copy()
    data_out['dep_time_short'] = data_out['scheduled_dep_time'].astype(int).astype(str).apply(lambda x:x.rjust(4)[:2])
    data_out['arr_time_short'] = data_out['scheduled_arr_time'].astype(int).astype(str).apply(lambda x:x.rjust(4)[:2])

    if org!='any':
        data_out = data_out.loc[(data_out['origin']==org)]
    if dest!='any':
        data_out = data_out.loc[(data_out['dest']==dest)]
    
    #remove the line with empty DepTime
    data_out=data_out.loc[data_out['dep_time_short'].str.strip()!='']
    data_out=data_out.loc[data_out['arr_time_short'].str.strip()!='']
    data_out['dep_time_short'] = data_out['dep_time_short'].astype(int)
    data_out['arr_time_short'] = data_out['arr_time_short'].astype(int)
    return data_out

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='YlGnBu'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=12)

    plt.tight_layout()
    plt.ylabel('True label',size=12)
    plt.xlabel('Predicted label',size=12)
    
def two_scales(ax1, time, data1, data2, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('C')
    ax1.set_ylabel('recall')

    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('precision')
    return ax1, ax2