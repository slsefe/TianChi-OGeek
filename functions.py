import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pylab as plt

from sklearn.metrics import f1_score
from sklearn.externals import joblib

# 1. data preprocessing and save
def train_preprocess(train_path, train_propessed_path):
    # Prefix has four null values that are mistaken for None and is populated with 'realnull'
    if os.path.exists(train_propessed_path):
        train_df = pickle.load(open(train_propessed_path,'rb'))
    else:
        train_df = pd.read_csv(train_path, sep='\t', header=None,
                       names=['prefix', 'query_prediction', 'title', 'tag', 'label'])
        # print('data is null?\n',train_df.isnull().sum())
        # fill the np.nan of column 'prefix' with 'null'
        train_df['prefix'] = train_df['prefix'].fillna('null')
        # print('data is null?\n',train_df.isnull().sum())
        
        # modify the wrong line 1815101
        # print(train_df[train_df.prefix=='花开花又'])
        train_df.loc[train_df.prefix=='花开花又','query_prediction'] = '{"花开花又落": "0.635", "花开花又落是什么歌": "0.365"}'
        train_df.loc[train_df.prefix=='花开花又','title'] = '大雁听过我的歌%2C山丹丹花开花又落....一年又一年?谁听过这首歌%2C知道歌名吗?'
        train_df.loc[train_df.prefix=='花开花又','tag'] = '知道'
        train_df.loc[train_df.prefix=='花开花又','label'] = 0
        # print(train_df[train_df.prefix=='花开花又'])
        # add the missed line to end
        train_df.loc[1999999,'prefix'] = '花开花又'
        train_df.loc[1999999,'query_prediction'] = '{"花开花又落": "0.635", "花开花又落是什么歌": "0.365"}'
        train_df.loc[1999999,'title'] = '等你花开花又落'
        train_df.loc[1999999,'tag'] = '音乐'
        train_df.loc[1999999,'label'] = 0
        # print(train_df[train_df.prefix=='花开花又'])
        
        # the label has 0,1,'0','1',transform '0','1' to 0,1 respectively
        # print(train_df.label.value_counts())
        train_df.loc[train_df.label=='0','label']=0
        train_df.loc[train_df.label=='1','label']=1
        # print(train_df.label.value_counts())
        train_df['label'] = train_df.label.astype('int')
        # print(train_df.info())
        pickle.dump(train_df, open(train_propessed_path,'wb'))
    return train_df


def load_data(train_path, train_preprocessed_path, valid_path, valid_preprocessed_path, test_path, test_preprocessed_path):
    
    if os.path.exists(train_preprocessed_path):
        train_df = pickle.load(open(train_preprocessed_path,'rb'))
    else:
        train_df = train_preprocess(train_path, train_preprocessed_path)
    
    if os.path.exists(valid_preprocessed_path):
        valid_df = pickle.load(open(valid_preprocessed_path,'rb'))
    else:
        valid_df = pd.read_csv(valid_path, sep='\t', header=None,names=['prefix', 'query_prediction', 'title', 'tag', 'label'])
        valid_df['label'] = valid_df.label.astype('int')
        pickle.dump(valid_df, open(valid_preprocessed_path,'wb'))
    
    if os.path.exists(test_preprocessed_path):
        test_df = pickle.load(open(test_preprocessed_path,'rb'))
    else:
        test_df = pd.read_csv(test_path, sep='\t', header=None, names=['prefix', 'query_prediction', 'title', 'tag'])
        test_df['label'] = -1
        pickle.dump(test_df, open(test_preprocessed_path,'wb'))

    return train_df, valid_df, test_df


# 2. feature engineering
def split_prediction(text):
    if pd.isna(text): 
        return []
    else:
        return [s.strip() for s in text.replace("{", "").replace("}", "").split(", ")]


def length_feature(data, feature, upper = 8):
    if feature == 'query_prediction':
        data['pred'] = data[feature].apply(split_prediction)
        feature = 'pred'
    data[feature+'_len'] = data[feature].apply(len)
    data[feature+'_len'] = data[feature+'_len'].map(lambda x: upper if x > upper else x)
    return data


def length_features(train_df, valid_df, test_df, features = ['prefix', 'query_prediction', 'title']):
    for feature in features:
        train_df = length_feature(train_df, feature)
        valid_df = length_feature(valid_df, feature)
        test_df = length_feature(test_df, feature)
    return train_df, valid_df, test_df
    


def create_single_feature(feature, train_df, valid_df, test_data):
    feature_path = './features/'+feature+'.pkl'
    if os.path.exists(feature_path):
        temp = pickle.load(open(feature_path,'rb'))
    else:
        # average click number for items in feature
        a = train_df['label'].sum()/train_df[feature].nunique()
        # loga = np.log1p(train_df['label'].sum())/np.log(train_df[feature].nunique())
        
        # average search number for items in feature
        b =train_df['label'].count()/train_df[feature].nunique()
        # logb = np.log1p(train_df['label'].count())/np.log1p(train_df[feature].nunique())

        # click number & search number for specific value of feature
        temp = train_df[[feature,'label']].groupby(feature, as_index = False)['label'].agg({feature+'_click':'sum', feature+'_count':'count'})
        # temp[feature+'_click_smooth'] = (temp[feature+'_click']+a)/2
        # temp[feature+'_count_smooth'] = (temp[feature+'_count']+b)/2
        
        # log transform of click & search number
        # temp[feature+'_click_log'] = np.log1p(temp[feature+'_click'])
        # temp[feature+'_count_log'] = np.log1p(temp[feature+'_count'])
        
        # twice log transform of click & search number
        # temp[feature+'_click_log2'] = np.log1p(temp[feature+'_click_log'])
        # temp[feature+'_count_log2'] = np.log1p(temp[feature+'_count_log'])

        # diff for click & search number
        # temp[feature+'_click_diff'] = temp[feature+'_click'] - temp[feature+'_click'].mean()
        # temp[feature+'_count_diff'] = temp[feature+'_count'] - temp[feature+'_count'].mean()
        
        # ratio for click & search number
        # temp[feature+'_click_ratio'] = temp[feature+'_click']/temp[feature+'_click'].sum()
        # temp[feature+'_count_ratio'] = temp[feature+'_count']/temp[feature+'_count'].sum()
        
        # ratio for log of click & search number
        # temp[feature+'_click_ratio_log'] = np.log1p(temp[feature+'_click'])-np.log1p(temp[feature+'_click'].sum())
        # temp[feature+'_count_ratio_log'] = np.log1p(temp[feature+'_count'])-np.log1p(temp[feature+'_count'].sum())

        # absolutely transform ratio
        temp[feature+'_ctr_abs'] = temp[feature+'_click']/(temp[feature+'_count'])
        
        # smoothly transform ratio
        temp[feature+'_ctr_smooth'] = (temp[feature+'_click']+a)/(temp[feature+'_count']+b)
        
        # diff of absolutely transform ratio
        # temp[feature+'_ctr_diff'] = temp[feature+'_ctr_abs'] - temp[feature+'_ctr_abs'].mean()
        
        # relatively transform ratio
        # temp[feature+'_ctr_rela'] = temp[feature+'_ctr_diff']/(temp[feature+'_ctr_abs'].mean())

        # log of absolutely transform ratio
        # temp[feature+'_ctr_log_abs'] = temp[feature+'_click_log']-(temp[feature+'_count_log'])
        
        # log of smoothly transform ratio
        # temp[feature+'_ctr_log_smooth'] = (temp[feature+'_click_log']+np.log1p(a))/(temp[feature+'_count_log']+np.log1p(b))
        
        # log of diff of absolutely transform ratio
        # temp[feature+'_ctr_log_diff'] = temp[feature+'_ctr_log_abs'] - temp[feature+'_ctr_log_abs'].mean()
        
        # log of relatively transform ratio
        # temp[feature+'_ctr_log_rela'] = temp[feature+'_ctr_log_diff']/(temp[feature+'_ctr_log_abs'].mean())

        pickle.dump(temp, open(feature_path,'wb'))
    
    return temp


def create_feature_pair(item_g, train_df, valid_df, test_data):
    feature_path = './features/'+'_'.join(item_g)+'.pkl'
    if os.path.exists(feature_path):
        temp = pickle.load(open(feature_path,'rb'))
    else:
        a = train_df['label'].sum()/train_df[item_g].nunique()
        b = train_df['label'].count()/train_df[item_g].nunique()
        temp = train_df.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'_count':'count'})
        # # 平滑点击数与搜索数
        # temp['_'.join(item_g)+'_click_smooth'] = temp['_'.join(item_g)+'_click']+a
        # temp['_'.join(item_g)+'_count_smooth'] = temp['_'.join(item_g)+'_count']+b
        # # 点击数和搜索数的log变换
        # temp['_'.join(item_g)+'_click_log'] = np.log1p(temp['_'.join(item_g)+'_click'])
        # temp['_'.join(item_g)+'_count_log'] = np.log1p(temp['_'.join(item_g)+'_count'])
        
        # # 点击数和搜索数与均值的差值
        # temp['_'.join(item_g)+'_click_diff'] = temp['_'.join(item_g)+'_click'] - temp['_'.join(item_g)+'_click'].mean()
        # temp['_'.join(item_g)+'_count_diff'] = temp['_'.join(item_g)+'_count'] - temp['_'.join(item_g)+'_count'].mean()
        
        # # 点击数占比与搜索数占比
        # temp['_'.join(item_g)+'_click_ratio'] = temp['_'.join(item_g)+'_click']/temp['_'.join(item_g)+'_click'].sum()
        # temp['_'.join(item_g)+'_count_ratio'] = temp['_'.join(item_g)+'_count']/temp['_'.join(item_g)+'_count'].sum()
        # # 平滑点击数占比与搜索数占比
        # temp['_'.join(item_g)+'_click_ratio_smooth'] = (temp['_'.join(item_g)+'_click']+a)/(2*temp['_'.join(item_g)+'_click'].sum())
        # temp['_'.join(item_g)+'_count_ratio_smooth'] = (temp['_'.join(item_g)+'_count']+b)/(2*temp['_'.join(item_g)+'_count'].sum())
        # # 点击数log占比与搜索数log占比
        # temp['_'.join(item_g)+'_click_ratio_log'] = np.log1p(temp['_'.join(item_g)+'_click_ratio'])
        # temp['_'.join(item_g)+'_count_ratio_log'] = np.log1p(temp['_'.join(item_g)+'_count_ratio'])
        
        # 绝对转化率
        temp['_'.join(item_g)+'_ctr_abs'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'_count'])
        # 贝叶斯平滑转换率
        temp['_'.join(item_g)+'_ctr_smooth'] = (temp['_'.join(item_g)+'_click']+a)/(temp['_'.join(item_g)+'_count']+b)
        # # 绝对转化率差值
        # temp['_'.join(item_g)+'_ctr_diff'] = temp['_'.join(item_g)+'_ctr_abs'] - temp['_'.join(item_g)+'_ctr_abs'].mean()
        # # 相对转化率
        # temp['_'.join(item_g)+'_ctr_rela'] = temp['_'.join(item_g)+'_ctr_diff']/(temp['_'.join(item_g)+'_ctr_abs'].mean())
        
        # # prefix的log绝对转换率
        # temp['_'.join(item_g)+'_ctr_log_abs'] = temp['_'.join(item_g)+'_click_log']-(temp['_'.join(item_g)+'_count_log'])
        # # prefix的log贝叶斯平滑转换率
        # temp['_'.join(item_g)+'_ctr_log_smooth'] = (temp['_'.join(item_g)+'_click_log']+np.log1p(a))/(temp['_'.join(item_g)+'_count_log']+np.log1p(b))
        # # prefix的log绝对转化率差值
        # temp['_'.join(item_g)+'_ctr_log_diff'] = temp['_'.join(item_g)+'_ctr_log_abs'] - temp['_'.join(item_g)+'_ctr_log_abs'].mean()
        # # prefix的log相对转化率
        # temp['_'.join(item_g)+'_ctr_log_rela'] = temp['_'.join(item_g)+'_ctr_log_diff']/(temp['_'.join(item_g)+'_ctr_log_abs'].mean())
        
        pickle.dump(temp, open(feature_path,'wb'))
    return temp


def feature_engineering(train_df, valid_df, test_df, items = ['prefix', 'title', 'tag']):
    
    # get length features
    # train_df, valid_df, test_df = length_features(train_df, valid_df, test_df)
    
    # extract features
    for item in items:
        temp = create_single_feature(item, train_df, valid_df, test_df)
        train_df = pd.merge(train_df, temp, on=item, how='left')
        valid_df = pd.merge(valid_df, temp, on=item, how='left')
        test_df = pd.merge(test_df, temp, on=item, how='left')
    
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            item_g = [items[i], items[j]]
            temp = create_feature_pair(item_g, train_df, valid_df, test_df)
            train_df = pd.merge(train_df, temp, on=item_g, how='left')
            valid_df = pd.merge(valid_df, temp, on=item_g, how='left')
            test_df = pd.merge(test_df, temp, on=item_g, how='left')

    # fill miss values of new features in valid & test data with mean
    for col in valid_df.columns[train_df.dtypes == 'float64']:
        valid_df.fillna(valid_df[col].mean())
    for col in test_df.columns[test_df.dtypes == 'float64']:
        test_df.fillna(test_df[col].mean())
    
    return train_df, valid_df, test_df

# 3.1 Construct training set, validation set and test set
def split_data(train_df, valid_df, test_df):
    # train_df = train_df.drop(['prefix','query_prediction', 'pred', 'title', 'tag'], axis = 1)
    # valid_df = valid_df.drop(['prefix','query_prediction', 'pred', 'title', 'tag'], axis = 1)
    # test_df = test_df.drop(['prefix','query_prediction', 'pred', 'title', 'tag'], axis = 1)
    train_df = train_df.drop(['prefix','query_prediction', 'title', 'tag'], axis = 1)
    valid_df = valid_df.drop(['prefix','query_prediction', 'title', 'tag'], axis = 1)
    test_df = test_df.drop(['prefix','query_prediction', 'title', 'tag'], axis = 1)

    X = train_df.drop(['label'], axis = 1)
    y = train_df['label']
    X_valid = valid_df.drop(['label'], axis = 1)
    y_valid = valid_df['label']
    X_test = test_df.drop(['label'], axis = 1)

    del train_df, valid_df, test_df

    return X, y, X_valid, y_valid, X_test


# 3.2 train model
def train_lgb_model(X, y, X_valid, y_valid, params, model_path):
    lgb_train = lgb.Dataset(X, y)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=[lgb_eval, lgb_train],
                    valid_names=['eval', 'train'],
                    early_stopping_rounds=50,
                    verbose_eval=True,
                    )

    # f1-score on train data
    print('f1-score on train data:',f1_score(y, np.where(gbm.predict(X, num_iteration=gbm.best_iteration)>0.5, 1,0)))
    # f1-score on validation data
    print('f1-score on validation data:',f1_score(y_valid, np.where(gbm.predict(X_valid, num_iteration=gbm.best_iteration)>0.5, 1,0)))
    print(gbm.best_score)

    joblib.dump(gbm, model_path)
    # gbm.save_model(model_file, num_iteration=gbm.best_iteration)
    return gbm

def load_model(X, y, X_valid, y_valid, params, model_path):
    if os.path.exists(model_path):
        gbm = joblib.load(model_path)
    else:
        gbm = train_lgb_model(X, y, X_valid, y_valid, params, model_path)

    return gbm


def plot_feature_importance(model, fig_path, max_num_features=30):
    plt.figure(figsize=(12,6), dpi=400)
    lgb.plot_importance(model, max_num_features=max_num_features)
    plt.title("Featurer importances")
    plt.savefig(fig_path, dpi=400)


def model_predict(model, test_df, X_test, submit_path):
    test_df['label'] = model.predict(X_test, num_iteration = model.best_iteration)
    test_df['label'] = test_df['label'].apply(lambda x: round(x))

    test_df['label'].to_csv(submit_path, header=None, index = False)
