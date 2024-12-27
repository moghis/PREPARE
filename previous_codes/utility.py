from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, mean_squared_error
import numpy as np
from sklearn.base import clone
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from lightgbm  import early_stopping 

# create new features by combining 2003 and 2012 scores and numbering ordinal variables
def feature_engineering(data):
    data['rjob_hrswk_change']=data['rjob_hrswk_12'].fillna(0)-data['rjob_hrswk_03'].fillna(0).astype(float)
    data['rjob_hrswk']=data[['rjob_hrswk_03','rjob_hrswk_12']].mean(axis=1).astype(float)
    data['max_work_year']=data[['rjob_end_12','rjob_end_03']].max(axis=1).astype(float)
    data['years_since_work']=(data['year']-data['max_work_year']).astype(float)
    data['hincome_change']=(data['hincome_12'].fillna(0)-data['hincome_03'].fillna(0)).astype(float)
    data['niadl_change']=(data['n_iadl_12'].fillna(0)-data['n_iadl_03'].fillna(0)).astype(float)
    data['adl_max']=(data[['n_adl_12','n_adl_03']].fillna(0)).max(axis=1).astype(float)
    data['iadl_max']=(data[['n_iadl_12','n_iadl_03']].fillna(0)).max(axis=1).astype(float)
    data['neg_adl']=6-data['adl_max']
    data['neg_iadl']=4-data['iadl_max']
    data['adl_change']=(data['n_adl_12'].fillna(0)-data['n_adl_03'].fillna(0)).astype(float)
    data['depr_change']=(data['n_depr_12'].fillna(0)-data['n_depr_03'].fillna(0)).astype(float)
    data['glob_hlth_03']=data['glob_hlth_03'].replace({'5. Poor':0, '4. Fair':1, '3. Good':2, '2. Very good': 3, '1. Excellent':4}).astype(float)
    data['glob_hlth_12']=data['glob_hlth_12'].replace({'5. Poor':0, '4. Fair':1, '3. Good':2, '2. Very good': 3, '1. Excellent':4}).astype(float)
    data['glob_hlth']=data[['glob_hlth_03', 'glob_hlth_12']].sum(axis=1).astype(float)
    data['bmi_03']=data['bmi_03'].replace({'1. Underweight': 1, '2. Normal weight': 2, '3. Overweight':3, '4. Obese':4, '5. Morbidly obese':5}).astype(float)
    data['bmi_12']=data['bmi_12'].replace({'1. Underweight': 1, '2. Normal weight': 2, '3. Overweight':3, '4. Obese':4, '5. Morbidly obese':5}).astype(float)
    data['bmi']=data[['bmi_03', 'bmi_12']].sum(axis=1).astype(float)
    data['employment_03']=data['employment_03'].replace({'1. Currently Working': 'Working', '2. Currently looking for work':'Looking for work', '3. Dedicated to household chores': 'House', '4. Retired, incapacitated, or does not work': 'No work'})
    data['employment_12']=data['employment_12'].replace({'1. Currently Working': 'Working', '2. Currently looking for work':'Looking for work', '3. Dedicated to household chores': 'House', '4. Retired, incapacitated, or does not work': 'No work'})
    data['memory_12']=data['memory_12'].replace({'5. Poor':0, '4. Fair':1, '3. Good':2, '2. Very good': 3, '1. Excellent':4}).astype(float)
    data['edu_gru_03']=data['edu_gru_03'].replace({'0. No education':0,'1. 1–5 years':1, '2. 6 years':2, '3. 7–9 years':3,'4. 10+ years':4}).astype(float)
    data['edu_gru_12']=data['edu_gru_12'].replace({'0. No education':0,'1. 1–5 years':1, '2. 6 years':2, '3. 7–9 years':3,'4. 10+ years':4}).astype(float)
    data['urban_03']=data['urban_03'].fillna(0).replace({ '1. 100,000+':2, '0. <100,000':1}).astype(float)
    data['urban_12']=data['urban_03'].fillna(0).replace({ '1. 100,000+':2, '0. <100,000':1}).astype(float)
    data['issste']=data[['issste_03', 'issste_12']].fillna(0).max(axis=1).astype(float)
    data['urban']=data[['urban_03', 'urban_12']].max(axis=1).astype(float)
    data['edu_gru']=data[['edu_gru_03', 'edu_gru_12']].max(axis=1).astype(float)
    data['hincome']=data[['hincome_03', 'hincome_12']].max(axis=1).astype(float)
    data['illnesses']=data[['n_illnesses_03', 'n_illnesses_12']].max(axis=1).astype(float)
    data['alc_tob_03']=data[['alcohol_03','tobacco_03']].sum(axis=1).astype(float)
    data['alc_tob_12']=data[['alcohol_12','tobacco_12']].sum(axis=1).astype(float)
    data['alc_tob']=data[['alc_tob_03', 'alc_tob_12']].max(axis=1).astype(float)
    data['rearnings']=data[['rearnings_03', 'rearnings_12']].max(axis=1).astype(float)
    data.drop(columns=['issste_03', 'issste_12','urban_03', 'urban_12','edu_gru_03', 'edu_gru_12','bmi_03', 'bmi_12','alc_tob_03', 'alc_tob_12','alcohol_03','tobacco_03','alcohol_12','tobacco_12','hincome_03', 'hincome_12' ], inplace=True)
    return data

#get final features for model based on correlation with target variable and removing variables that are highly correlated with each other
def get_final_features(train_data,threshold):

    features=[]

    # remove ID, other identifying/non-input columns
    for col in train_data.columns:
        if 'uid' not in col:
            features.append(col)

    #select final features based on correlation with target variable
    correlations=train_data[features].corr()[['composite_score']]
    final_features=list(correlations[((correlations['composite_score']>.07) | (correlations['composite_score']<-.07))].T.columns.values) 

    #remove other redundant features
    threshold = threshold
    correlation_matrix = train_data[final_features].drop(columns='composite_score').corr()
    highly_correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated_features.add(colname)

    final_features=list(set(final_features)-highly_correlated_features)

    return final_features

#train model with Kfolds based on the unique uids and return the RMSE on t he validation set
def TrainML(model_class, df ,final_features, n_splits,SEED):
    SKF = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_S = []
    test_S = []
    oof_non_rounded = np.zeros(len(df), dtype=float) 
    unique_uids = df['uid'].unique()
    for fold, (train_id, val_id) in enumerate(tqdm(SKF.split(unique_uids), desc="Training Folds", total=n_splits)):
        df_train, df_val = df.iloc[train_id], df.iloc[val_id]
        X_train, X_val = df_train[final_features], df_val[final_features]
        y_train, y_val=df_train['composite_score'], df_val['composite_score']
        model = clone(model_class)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        oof_non_rounded[val_id] = y_val_pred
        train_rmse = mean_squared_error(y_train, y_train_pred.round(0).astype(int), squared=False)
        val_rmse = mean_squared_error(y_val, y_val_pred.round(0).astype(int), squared=False)
        train_S.append(train_rmse)
        test_S.append(val_rmse)
        print(f"Fold {fold+1} - Train AUC: {train_rmse:.4f}, Validation AUC: {val_rmse:.4f}")
        clear_output(wait=True)

    print(f"Mean Train RMSE --> {np.mean(train_S):.4f}")
    print(f"Mean Validation RMSE ---> {np.mean(test_S):.4f}")




def get_cat_cols(data):
    # Get the columns with object datatype
    cat_columns=[]
    dummies=[]
    for col in data.columns:
        if data[col].dtype=='object' and 'uid' not in col:
            cat_columns.append(col)
            dummies.append(col)
        elif data[col].dtype!='object' and 'uid' not in col and (data[col].max()==1.0):
            cat_columns.append(col)
            data[col].fillna(0, inplace=True)
        else:
            continue
    return cat_columns, dummies


#encode categorical columns with OneHotEncoder
def encode_cat_cols(train_data, test_data, cat_cols, dummy_cols):
    enc = OneHotEncoder()
    enc.fit(train_data[dummy_cols])
    encoded_train_data=enc.transform(train_data[dummy_cols]).toarray()
    encoded_test_data=enc.transform(test_data[dummy_cols]).toarray()
    feature_names = enc.get_feature_names_out(dummy_cols)
    train_data.drop(columns=dummy_cols, inplace=True)
    test_data.drop(columns=dummy_cols, inplace=True)
    encoded_train_df = pd.DataFrame(encoded_train_data, columns=feature_names)
    encoded_test_df = pd.DataFrame(encoded_test_data, columns=feature_names)
    train_data[feature_names]=encoded_train_df[feature_names]
    test_data[feature_names]=encoded_test_df[feature_names]
    return train_data, test_data, feature_names