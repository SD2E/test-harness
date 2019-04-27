from sklearn.svm import SVC
from harness.th_model_classes.class_sklearn_classification import SklearnClassification
import numpy as np

features = ['_rxn_M_acid',
            '_rxn_M_inorganic',
            '_rxn_M_organic',
#            '_feat_org_inorg_2ratio',
            '_feat_org_inorg_ratio',            
]

def proc_data(df):
    df['_feat_org_inorg_ratio'] = (np.arctan2(df._rxn_M_organic,df._rxn_M_inorganic)-np.pi/4)**2
    df['_feat_org_inorg_2ratio'] = (np.arctan2(df._rxn_M_organic,df._rxn_M_inorganic)-3*np.pi/8)**2
    return df[features]

class RxnRatioOnlySVM(SklearnClassification):
    def __init__(self):
        self.features = features
        self.svm = SVC(kernel='rbf',
                       decision_function_shape='ovr',
                       probability=True,
                       C=10,
        )

    def fit(self,X,y):
        X = proc_data(X)
        self.svm.fit(X,y)

    def predict(self,X):
        X = proc_data(X)
        return self.svm.predict(X)
    
    def predict_proba(self,X):
        X = proc_data(X)   
        return self.svm.predict_proba(X)
    
def rxn_intuition_svm():
    model = RxnRatioOnlySVM()
    return SklearnClassification(model=model,
                                     model_author="Scott Novotney, Josh Schrier",
                                     model_description="RBF xgb w/org/inorg ratio, inorganic and acid amounts ",
    )


