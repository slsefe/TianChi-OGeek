import os
import pickle
import pandas as pd
import numpy as np


def train_preprocess(train_path, train_propessed_path):
    # Prefix has four null values that are mistaken for None and is populated with 'realnull'
    if os.path.exists(train_propessed_path):
        train_preprocessed = pickle.load(open(train_propessed_path,'rb'))
    else:
        train_df = pd.read_csv(train_path, sep='\t', header=None,
                       names=['prefix', 'query_prediction', 'title', 'tag', 'label'])
        print('data is null?\n',train_df.isnull().sum())
        # fill the np.nan of column 'prefix' with 'null'
        train_df['prefix'] = train_df['prefix'].fillna('null')
        print('data is null?\n',train_df.isnull().sum())
        
        # modify the wrong line 1815101
        print(train_df[train_df.prefix=='花开花又'])
        train_df.loc[train_df.prefix=='花开花又','query_prediction'] = '{"花开花又落": "0.635", "花开花又落是什么歌": "0.365"}'
        train_df.loc[train_df.prefix=='花开花又','title'] = '大雁听过我的歌%2C山丹丹花开花又落....一年又一年?谁听过这首歌%2C知道歌名吗?'
        train_df.loc[train_df.prefix=='花开花又','tag'] = '知道'
        train_df.loc[train_df.prefix=='花开花又','label'] = 0
        print(train_df[train_df.prefix=='花开花又'])
        # add the missed line to end
        train_df.loc[1999999,'prefix'] = '花开花又'
        train_df.loc[1999999,'query_prediction'] = '{"花开花又落": "0.635", "花开花又落是什么歌": "0.365"}'
        train_df.loc[1999999,'title'] = '等你花开花又落'
        train_df.loc[1999999,'tag'] = '音乐'
        train_df.loc[1999999,'label'] = 0
        print(train_df[train_df.prefix=='花开花又'])
        
        # the label has 0,1,'0','1',transform '0','1' to 0,1 respectively
        print(train_df.label.value_counts())
        train_df.loc[train_df.label=='0','label']=0
        train_df.loc[train_df.label=='1','label']=1
        print(train_df.label.value_counts())
        train_df['label'] = train_df.label.astype('int')
        print(train_df.info())
        train_preprocessed = train_df
        del train_df
        pickle.dump(train_preprocessed, open(train_propessed_path,'wb'))
    return train_preprocessed


def create_single_feature(feature, train_df, valid_df, test_data):
    feature_path = './features/'+feature+'.pkl'
    if os.path.exists(feature_path):
        temp = pickle.load(open(feature_path,'rb'))
    else:
        # feature的平均点击次数
        a = train_df['label'].sum()/train_df[feature].nunique()
        loga = np.log1p(train_df['label'].sum())/np.log(train_df[feature].nunique())
        # feature的平均搜索次数
        b =train_df['label'].count()/train_df[feature].nunique()
        logb = np.log1p(train_df['label'].count())/np.log1p(train_df[feature].nunique())

        # feature的点击次数和搜索次数
        temp = train_df[[feature,'label']].groupby(feature, as_index = False)['label'].agg({feature+'_click':'sum', feature+'_count':'count'})
        temp[feature+'_click_smooth'] = (temp[feature+'_click']+a)/2
        temp[feature+'_count_smooth'] = (temp[feature+'_count']+b)/2
        # prefix点击数和搜索数的log变换
        temp[feature+'_click_log'] = np.log1p(temp[feature+'_click'])
        temp[feature+'_count_log'] = np.log1p(temp[feature+'_count'])
        # 对prefix点击数和搜索数的log变换再做一次log变换
        temp[feature+'_click_log2'] = np.log1p(temp[feature+'_click_log'])
        temp[feature+'_count_log2'] = np.log1p(temp[feature+'_count_log'])

        # prefix的点击数差值与搜索数差值
        temp[feature+'_click_diff'] = temp[feature+'_click'] - temp[feature+'_click'].mean()
        temp[feature+'_count_diff'] = temp[feature+'_count'] - temp[feature+'_count'].mean()

        # prefix的点击数占比与搜索数占比
        temp[feature+'_click_ratio'] = temp[feature+'_click']/temp[feature+'_click'].sum()
        temp[feature+'_count_ratio'] = temp[feature+'_count']/temp[feature+'_count'].sum()
        # prefix的点击数log占比与搜索数log占比
        temp[feature+'_click_ratio_log'] = np.log1p(temp[feature+'_click'])-np.log1p(temp[feature+'_click'].sum())
        temp[feature+'_count_ratio_log'] = np.log1p(temp[feature+'_count'])-np.log1p(temp[feature+'_count'].sum())

        # prefix的绝对转化率
        temp[feature+'_ctr_abs'] = temp[feature+'_click']/(temp[feature+'_count'])
        # prefix的贝叶斯平滑转换率
        temp[feature+'_ctr_smooth'] = (temp[feature+'_click']+a)/(temp[feature+'_count']+b)
        # prefix的绝对转化率差值
        temp[feature+'_ctr_diff'] = temp[feature+'_ctr_abs'] - temp[feature+'_ctr_abs'].mean()
        # prefix的相对转化率
        temp[feature+'_ctr_rela'] = temp[feature+'_ctr_diff']/(temp[feature+'_ctr_abs'].mean())

        # prefix的log绝对转换率
        temp[feature+'_ctr_log_abs'] = temp[feature+'_click_log']-(temp[feature+'_count_log'])
        # prefix的log贝叶斯平滑转换率
        temp[feature+'_ctr_log_smooth'] = (temp[feature+'_click_log']+np.log1p(a))/(temp[feature+'_count_log']+np.log1p(b))
        # prefix的log绝对转化率差值
        temp[feature+'_ctr_log_diff'] = temp[feature+'_ctr_log_abs'] - temp[feature+'_ctr_log_abs'].mean()
        # prefix的log相对转化率
        temp[feature+'_ctr_log_rela'] = temp[feature+'_ctr_log_diff']/(temp[feature+'_ctr_log_abs'].mean())

        pickle.dump(temp, open(feature_path,'wb'))
    
    return temp


def split_prediction(text):
    if pd.isna(text): 
        return []
    else:
        return [s.strip() for s in text.replace("{", "").replace("}", "").split(", ")]


def create_feature_pair(item_g, train_df, valid_df, test_data):
    feature_path = './features/'+'_'.join(item_g)+'.pkl'
    if os.path.exists(feature_path):
        temp = pickle.load(open(feature_path,'rb'))
    else:
        a = train_df['label'].sum()/train_df[item_g].nunique()
        b = train_df['label'].count()/train_df[item_g].nunique()
        temp = train_df.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'_count':'count'})
        # 平滑点击数与搜索数
        temp['_'.join(item_g)+'_click_smooth'] = temp['_'.join(item_g)+'_click']+a
        temp['_'.join(item_g)+'_count_smooth'] = temp['_'.join(item_g)+'_count']+b
        # 点击数和搜索数的log变换
        temp['_'.join(item_g)+'_click_log'] = np.log1p(temp['_'.join(item_g)+'_click'])
        temp['_'.join(item_g)+'_count_log'] = np.log1p(temp['_'.join(item_g)+'_count'])
        
        # 点击数和搜索数与均值的差值
        temp['_'.join(item_g)+'_click_diff'] = temp['_'.join(item_g)+'_click'] - temp['_'.join(item_g)+'_click'].mean()
        temp['_'.join(item_g)+'_count_diff'] = temp['_'.join(item_g)+'_count'] - temp['_'.join(item_g)+'_count'].mean()
        
        # 点击数占比与搜索数占比
        temp['_'.join(item_g)+'_click_ratio'] = temp['_'.join(item_g)+'_click']/temp['_'.join(item_g)+'_click'].sum()
        temp['_'.join(item_g)+'_count_ratio'] = temp['_'.join(item_g)+'_count']/temp['_'.join(item_g)+'_count'].sum()
        # 平滑点击数占比与搜索数占比
        temp['_'.join(item_g)+'_click_ratio_smooth'] = (temp['_'.join(item_g)+'_click']+a)/(2*temp['_'.join(item_g)+'_click'].sum())
        temp['_'.join(item_g)+'_count_ratio_smooth'] = (temp['_'.join(item_g)+'_count']+b)/(2*temp['_'.join(item_g)+'_count'].sum())
        # 点击数log占比与搜索数log占比
        temp['_'.join(item_g)+'_click_ratio_log'] = np.log1p(temp['_'.join(item_g)+'_click_ratio'])
        temp['_'.join(item_g)+'_count_ratio_log'] = np.log1p(temp['_'.join(item_g)+'_count_ratio'])
        
        # 绝对转化率
        temp['_'.join(item_g)+'_ctr_abs'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'_count'])
        # 贝叶斯平滑转换率
        temp['_'.join(item_g)+'_ctr_smooth'] = (temp['_'.join(item_g)+'_click']+a)/(temp['_'.join(item_g)+'_count']+b)
        # 绝对转化率差值
        temp['_'.join(item_g)+'_ctr_diff'] = temp['_'.join(item_g)+'_ctr_abs'] - temp['_'.join(item_g)+'_ctr_abs'].mean()
        # 相对转化率
        temp['_'.join(item_g)+'_ctr_rela'] = temp['_'.join(item_g)+'_ctr_diff']/(temp['_'.join(item_g)+'_ctr_abs'].mean())
        
        # prefix的log绝对转换率
        temp['_'.join(item_g)+'_ctr_log_abs'] = temp['_'.join(item_g)+'_click_log']-(temp['_'.join(item_g)+'_count_log'])
        # prefix的log贝叶斯平滑转换率
        temp['_'.join(item_g)+'_ctr_log_smooth'] = (temp['_'.join(item_g)+'_click_log']+np.log1p(a))/(temp['_'.join(item_g)+'_count_log']+np.log1p(b))
        # prefix的log绝对转化率差值
        temp['_'.join(item_g)+'_ctr_log_diff'] = temp['_'.join(item_g)+'_ctr_log_abs'] - temp['_'.join(item_g)+'_ctr_log_abs'].mean()
        # prefix的log相对转化率
        temp['_'.join(item_g)+'_ctr_log_rela'] = temp['_'.join(item_g)+'_ctr_log_diff']/(temp['_'.join(item_g)+'_ctr_log_abs'].mean())
        
        pickle.dump(temp, open(feature_path,'wb'))
    return temp


         