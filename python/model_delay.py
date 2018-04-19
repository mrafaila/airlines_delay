import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
from airline_utils import *
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
    
def model_rtg(data_origins):
    '''Function to model the Return to gate for each origin
    Categorical features are transformed by one hot encoding 
    ('dow','month','dep_time_short', 'carrier', 'dest')
    Area under curve and confusion matrix are plotted
    '''
    cols = ['CRS_ELAPSED_TIME',
            'LATE_AIRCRAFT_DELAY','CARRIER_DELAY',
            'count_flights_carrier','count_flights_origin','count_flights_route',
            'dow','month', 'dep_time_short', 'carrier', 'origin','dest', 'distance',
            'flight_hours',
            'rtg']
    features = ['CRS_ELAPSED_TIME',
                'LATE_AIRCRAFT_DELAY','CARRIER_DELAY',
                'flight_hours',
                'count_flights_carrier','count_flights_origin','count_flights_route', 
                'dow','month','dep_time_short', 'carrier', 'dest', 'distance']

    X_train={}
    y_train={}
    X_test={}
    y_test={}

    train_aucs = {}
    test_aucs = {}
    y_train_p = {}
    y_test_p = {}
    y_pred={}
    plt.figure()
    plt.xticks(size=15)
    plt.yticks(size=15)

    cnf_matrix=np.zeros((2,2))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    rtg = data_origins.groupby('origin')['rtg'].agg('sum')
    origins = rtg.sort_values(ascending=False).index[:10]

    for origin in origins:
        print(origin)
        color = next(colors)

        df = data_origins.loc[(data_origins['origin']==origin)].copy()
        df = df[cols].dropna(how='any').copy()

        categ_cols =  ['dep_time_short', 'carrier', 'dest','month', 'dow']

        labelencoder=LabelEncoder()
        for c in categ_cols:
            df[c] = labelencoder.fit(df[c]).transform(df[c])
        X = df[features].as_matrix()
        y = df['rtg'] 

        X_train[origin], X_test[origin], y_train[origin], y_test[origin] = train_test_split(
             X, y, test_size=0.25, random_state=42)
        models = LogisticRegression(class_weight='balanced')
        models.fit(X_train[origin], y_train[origin])

        # Get probabilities
        y_train_p[origin] = models.predict_proba(X_train[origin])[:,1]
        y_test_p[origin] = models.predict_proba(X_test[origin])[:,1]

        y_pred[origin] = models.predict(X_test[origin])

        # Evaluate model
        train_aucs[origin] = roc_auc_score(y_train[origin], y_train_p[origin])
        test_aucs[origin] = roc_auc_score(y_test[origin], y_test_p[origin])

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test[origin], y_test_p[origin])
        plt.plot(fpr, tpr, color=color)
        #label='ROC curve of classifier {0} (area = {1:0.2f})'
        #     ''.format(origin, test_aucs[origin]))
        cnf_matrix = cnf_matrix+confusion_matrix(y_test[origin], y_test_p[origin]>0.5)

    plt.plot([0, 1], [0, 1], 'k--')#, lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix.astype(int), classes=['not rtg','rtg'],
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix.astype(int), classes=['not rtg','rtg'], normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    # Plot auc
    plt.figure()
    accs = plt.hist(list(test_aucs.values()))
    plt.xlabel('AUC',size=20)
    plt.ylabel('#origins',size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title('AUC on all origins',size=20)
    
def model_delayed(data_origins):

    flights_df = data_origins.copy()

    # Filter data to keys of interest
    keys = ['month', 'dow', 'dom', 'carrier', 
            'origin',
           'dest', 
            'dep_time_short',
           'dep_delay', 'TAXI_OUT', 
            'CRS_ELAPSED_TIME',
            'AIR_TIME', 'distance', 
            'TAXI_IN',
            'arr_time_short'
           ]
    flights_df = flights_df[keys]
    # Shuffle data
    delay_cutoff = 15

    flights_df = flights_df.reset_index(drop=True)
    labels_preshuffle = flights_df['dep_delay'].values > delay_cutoff
    np.random.seed(0)
    flight_shuff_idx = np.random.permutation(flights_df.index)
    labels_shuffle = labels_preshuffle[flight_shuff_idx]
    flights_df = flights_df.loc[flight_shuff_idx]
    flights_df = flights_df.reset_index(drop=True)

    all_airports, airport_inverse, airport_count = np.unique(flights_df['origin'],return_counts=True,return_inverse=True)
    # Determine number of flights for the origin airport
    Nflights_orig = np.zeros(len(airport_inverse))
    for i in range(len(all_airports)):
        Nflights_orig[np.where(airport_inverse==i)] = airport_count[i]
    flights_df = flights_df.loc[flights_df.index[Nflights_orig>=7300]]
    flights_df = flights_df.dropna()


    flights_df = flights_df.reset_index(drop=True)
    labels_preshuffle = flights_df['dep_delay'].values > delay_cutoff

    np.random.seed(0)
    flight_shuff_idx = np.random.permutation(flights_df.index)
    labels_shuffle = labels_preshuffle[flight_shuff_idx]

    flights_df = flights_df.loc[flight_shuff_idx]

    flights_df = flights_df.reset_index(drop=True)
    sns.set_style('white')

    cutoffs = np.arange(60,780,60)
    original_feat = make_onehot_feat_dict_from_vals(flights_df,
                        'CRS_ELAPSED_TIME', 'dur', cutoffs)

    daysfeat_dict = make_onehot_feat_dict(flights_df, 'dow', 'day')
    monthsfeat_dict = make_onehot_feat_dict(flights_df, 'month', 'mo')
    dapfeat_dict = make_onehot_feat_dict(flights_df, 'origin', 'dap')
    alfeat_dict = make_onehot_feat_dict(flights_df, 'carrier', 'al')

    # Add departure hour as a feature
    hrfeat_dict = make_onehot_feat_dict(flights_df, 'dep_time_short', 'hr')

    all_dicts = [original_feat, daysfeat_dict, monthsfeat_dict, alfeat_dict, hrfeat_dict]
    feat_dict = all_dicts[0].copy()
    for d in all_dicts[1:]:
        feat_dict.update(d)

    df_feat = pd.DataFrame.from_dict(feat_dict)
    df_feat.head()
    airport_keys = np.array(list(dapfeat_dict.keys()))
    X_train = {}
    X_test = {}
    y_train = {}
    y_test = {}
    for apk in airport_keys:
        # Isolate X and y for each airport
        ap_idx = np.transpose(np.argwhere(dapfeat_dict[apk]))[0]
        X_all = df_feat.loc[ap_idx].values
        y_all = labels_shuffle[ap_idx]

        # Calculate train and test set size
        N_flights = len(y_all)
        N_train = int(N_flights*.7)
        N_test = N_flights - N_train

        # Make train and test sets
        X_train[apk[-3:]] = X_all[:N_train]
        X_test[apk[-3:]] = X_all[N_train:]
        y_train[apk[-3:]] = y_all[:N_train]
        y_test[apk[-3:]] = y_all[N_train:]

    train_aucs = {}
    test_aucs = {}
    train_ps = {}
    test_ps = {}
    y_pred={}
    plt.figure()
    cnf_matrix=np.zeros((2,2))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    lw = 2
    for k,color in zip(X_train.keys(),colors):
        print(k)
        #models = ensemble.GradientBoostingClassifier(**params)
        models = LogisticRegression(class_weight='balanced')
        models.fit(X_train[k], y_train[k])

        # Get probabilities
        train_ps[k] = models.predict_proba(X_train[k])[:,1]
        test_ps[k] = models.predict_proba(X_test[k])[:,1]

        y_pred[k] = models.predict(X_test[k])

        # Evaluate model
        train_aucs[k] = roc_auc_score(y_train[k], train_ps[k])
        test_aucs[k] = roc_auc_score(y_test[k], test_ps[k])    

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test[k], test_ps[k])
        plt.plot(fpr, tpr, color=color, lw=lw)
        cnf_matrix = cnf_matrix+confusion_matrix(y_test[k], test_ps[k]>0.5)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix.astype(int), classes=['not delayed','delayed'],
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix.astype(int), classes=['not delayed','delayed'], normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    # Plot accuracy
    plt.figure(figsize=(8,5))
    accs = plt.hist(list(test_aucs.values()),bins=np.arange(0.6,0.8,0.02))
    plt.xlabel('AUC',size=20)
    plt.ylabel('count',size=20)
    plt.xticks(np.arange(0.6,0.8,0.02), size=15)
    plt.yticks(size=15)
    plt.title('Area under curve',size=20)
    
    