"""
We use SVR as a baseline model on drug and gene expression data.
"""

from sklearn.svm import SVR
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num = 15

while num != 0:
    num -= 1


    df = pd.read_csv('LatentVec_Drug+GeneExp(cgc+eliminated+unsampledGene+unsampledDrug).txt', sep='\t')


    #df = pd.read_csv('LatentVec_Drug+RawVec_GeneExp(cgc+eliminated+unsampledDrug).txt', sep='\t')

    # Let x be encoded vectors of drug data
    x = df.loc[:, 'd0':]
    x = x.values.tolist()

    xx = []
    for i in range(len(x[0])):
        xx.append([x[j][i] for j in range(len(x))])

    transformer = Normalizer().fit(xx)
    xx = transformer.transform(xx)

    x = []
    for i in range(len(xx[0])):
        x.append([xx[j][i] for j in range(len(xx))])

    # Let y be lnIC50 values, which is the performance of a drug used against a cancer cell line
    y = df.loc[:, 'LN_IC50']
    y = y.values.tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1/9)

    # train

    # svr = GridSearchCV(SVR(), param_grid={"kernel": ("rbf"), "C": np.logspace(-3, 3, 7), "gamma": np.logspace(-3, 3, 7)})
    svr = SVR(kernel="poly", C=10, )
    svr.fit(x_train, y_train)
    joblib.dump(svr, '.\\svr.pkl')


    best_valid = 0
    y_pre = svr.predict(x_valid)
    r2_valid = r2_score(y_valid, y_pre)

    if r2_valid > best_valid:
        best_valid = r2_valid
        # test
        svr = joblib.load('.\\svr.pkl')
        y_pre = svr.predict(x_test)
        r2 = r2_score(y_test, y_pre)
        rmse = mean_squared_error(y_test, y_pre) ** 0.5
        print(r2, rmse)

        plt.scatter(y_test, y_pre, c='blue')
        plt.xlabel('True drug response (IC50)', color='k')
        plt.ylabel('Predicted drug response (IC50)', color='k')
        plt.title("R2_score: {0: .6f}".format(r2))
        plt.savefig("CGC_SVR.eps", dpi=800)
        plt.show()
        '''
        #df = pd.read_csv('R2_Score_rawcgc(cgc+eliminated+unsampledDrug).txt', sep='\t')
        
        df = df.append({'id': int(len(df)),
                        'R2_test': r2,
                        'RMSE_test': rmse,
                        },
                    ignore_index=True)

        
        df.to_csv('R2_Score_rawcgc(cgc+eliminated+unsampledDrug).txt', sep='\t', index=False)
        '''

