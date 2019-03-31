# 1.1 import necessary libraries
import numpy as np
import pandas as pd
import os
import pickle
import lightgbm as lgb
import xgboost as xgb

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, GridSearchCV
# from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import matplotlib.pylab as plt
%matplotlib inline
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 12, 4

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_seq_items', None)


# 1.2 load data
# train_file = 'd:/oppo/input/oppo_round1_train_20180929.txt'
valid_file = 'd:/oppo/input/oppo_round1_vali_20180929.txt'
test_A_file = 'd:/oppo/input/oppo_round1_test_A_20180929.txt'
# train_df = pd.read_csv(train_file, sep='\t', header=None,names=['prefix', 'query_prediction', 'title', 'tag', 'label'])
valid_df = pd.read_csv(valid_file, sep='\t', header=None,names=['prefix', 'query_prediction', 'title', 'tag', 'label'])
test_data = pd.read_csv(test_A_file, sep='\t', header=None, names=['prefix', 'query_prediction', 'title', 'tag'])
test_data['label'] = -1


# 1.3 preprocessing of the train data
def train_preprocess():
    print('preprocessing train data start')
    # Prefix has four null values that are mistaken for None and is populated with 'realnull'
    file_path = 'd:/oppo/input/train_preprocessed.pkl'
    if os.path.exists(file_path):
        train_preprocessed = pickle.load(open(file_path,'rb'))
    else:
        train_file = 'd:/oppo/input/oppo_round1_train_20180929.txt'
        train_df = pd.read_csv(train_file, sep='\t', header=None,
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
        pickle.dump(train_preprocessed,open(file_path,'wb'))
        train_df.to_csv('d:/oppo/input/train_preprocessed.csv',sep='\t',index=None,header=None)
    print('preprocessing train data end')
    return train_preprocessed
train_df = train_preprocess()


# 1.4 merge training data and test data
# train_data = pd.concat([train_df,valid_df])
# del train_df,valid_df
train_df['label'] = train_df['label'].apply(lambda x: int(x))
valid_df['label'] = valid_df['label'].apply(lambda x: int(x))
test_data['label'] = test_data['label'].apply(lambda x: int(x))



# 2.1 prefix features
# prefix的平均点击次数
a = train_df['label'].sum()/train_df['prefix'].nunique()
loga = np.log1p(train_df['label'].sum())/np.log(train_df['prefix'].nunique())
# prefix的平均搜索次数
b =train_df['label'].count()/train_df['prefix'].nunique()
logb = np.log1p(train_df['label'].count())/np.log1p(train_df['prefix'].nunique())

# prefix的点击次数和搜索次数
temp = train_df[['prefix','label']].groupby('prefix', as_index = False)['label'].agg({'prefix_click':'sum', 'prefix_count':'count'})
# prefix点击数和搜索数的log变换
temp['prefix_click_log'] = np.log1p(temp['prefix_click'])
temp['prefix_count_log'] = np.log1p(temp['prefix_count'])
# 对prefix点击数和搜索数的log变换再做一次log变换
temp['prefix_click_log2'] = np.log1p(temp['prefix_click_log'])
temp['prefix_count_log2'] = np.log1p(temp['prefix_count_log'])

# prefix的点击数差值与搜索数差值
temp['prefix_click_diff'] = temp['prefix_click'] - temp['prefix_click'].mean()
temp['prefix_count_diff'] = temp['prefix_count'] - temp['prefix_count'].mean()

# prefix的点击数占比与搜索数占比
temp['prefix_click_ratio'] = temp['prefix_click']/temp['prefix_click'].sum()
temp['prefix_count_ratio'] = temp['prefix_count']/temp['prefix_count'].sum()
# prefix的点击数log占比与搜索数log占比
temp['prefix_click_ratio_log'] = np.log1p(temp['prefix_click'])-np.log1p(temp['prefix_click'].sum())
temp['prefix_count_ratio_log'] = np.log1p(temp['prefix_count'])-np.log1p(temp['prefix_count'].sum())

# prefix的绝对转化率
temp['prefix_ctr_abs'] = temp['prefix_click']/(temp['prefix_count'])
# prefix的贝叶斯平滑转换率
temp['prefix_ctr_smooth'] = (temp['prefix_click']+a)/(temp['prefix_count']+b)
# prefix的绝对转化率差值
temp['prefix_ctr_diff'] = temp['prefix_ctr_abs'] - temp['prefix_ctr_abs'].mean()
# prefix的相对转化率
temp['prefix_ctr_rela'] = temp['prefix_ctr_diff']/(temp['prefix_ctr_abs'].mean())

# prefix的log绝对转换率
temp['prefix_ctr_log_abs'] = temp['prefix_click_log']-(temp['prefix_count_log'])
# prefix的log贝叶斯平滑转换率
temp['prefix_ctr_log_smooth'] = (temp['prefix_click_log']+np.log1p(a))/(temp['prefix_count_log']+np.log1p(b))
# prefix的log绝对转化率差值
temp['prefix_ctr_log_diff'] = temp['prefix_ctr_log_abs'] - temp['prefix_ctr_log_abs'].mean()
# prefix的log相对转化率
temp['prefix_ctr_log_rela'] = temp['prefix_ctr_log_diff']/(temp['prefix_ctr_log_abs'].mean())

train_df = pd.merge(train_df, temp, on='prefix', how='left')
valid_df = pd.merge(valid_df, temp, on='prefix', how='left')
test_data = pd.merge(test_data, temp, on='prefix', how='left')
del temp
# 对于valid和test数据集中新出现的数值型特征，所有的缺失值用均值代替
for col in valid_df.columns[train_df.dtypes == 'float64']:
    valid_df.fillna(valid_df[col].mean())
for col in test_data.columns[test_data.dtypes == 'float64']:
    test_data.fillna(test_data[col].mean())


# 2.2 length of prefix features
train_df['prefix_len'] = train_df['prefix'].apply(len)
valid_df['prefix_len'] = valid_df['prefix'].apply(len)
test_data['prefix_len'] = test_data['prefix'].apply(len)

train_df['prefix_len'] = train_df['prefix_len'].map(lambda x: 8 if x>8 else x)
valid_df['prefix_len'] = valid_df['prefix_len'].map(lambda x: 8 if x>8 else x)
test_data['prefix_len'] = test_data['prefix_len'].map(lambda x: 8 if x>8 else x)
train_df[['prefix_len','label']].groupby(['prefix_len'])['label'].describe()

# 平均每个prefix_len的点击次数
a = train_df['label'].sum()/train_df['prefix_len'].nunique()
# 平均每个prefix_len的搜索次数
b = train_df['label'].count()/train_df['prefix_len'].nunique()

# prefix的点击次数和搜索次数
temp = train_df[['prefix_len','label']].groupby('prefix_len', as_index = False)['label'].agg({'prefixlen_click':'sum', 'prefixlen_count':'count'})

temp['prefixlen_click_log'] = np.log1p(temp['prefixlen_click'])
temp['prefixlen_count_log'] = np.log1p(temp['prefixlen_count'])

# prefixlen的点击数差值与搜索数差值
temp['prefixlen_click_diff'] = temp['prefixlen_click'] - temp['prefixlen_click'].mean()
temp['prefixlen_count_diff'] = temp['prefixlen_count'] - temp['prefixlen_count'].mean()

# prefix的点击数占比与搜索数占比
temp['prefixlen_click_ratio'] = temp['prefixlen_click']/temp['prefixlen_click'].sum()
temp['prefixlen_count_ratio'] = temp['prefixlen_count']/temp['prefixlen_count'].sum()

# prefix的绝对转化率
temp['prefixlen_ctr_abs'] = temp['prefixlen_click']/(temp['prefixlen_count'])
# prefix的贝叶斯平滑转换率
temp['prefixlen_ctr_smooth'] = (temp['prefixlen_click']+a)/(temp['prefixlen_count']+b)
# prefix的绝对转化率差值
temp['prefixlen_ctr_diff'] = temp['prefixlen_ctr_abs'] - temp['prefixlen_ctr_abs'].mean()
# prefix的相对转化率
temp['prefixlen_ctr_rela'] = temp['prefixlen_ctr_diff']/(temp['prefixlen_ctr_abs'].mean())

train_df = pd.merge(train_df, temp, on='prefix_len', how='left')
valid_df = pd.merge(valid_df, temp, on='prefix_len', how='left')
test_data = pd.merge(test_data, temp, on='prefix_len', how='left')
temp.to_csv('d:/oppo/input/prefixlen_feature.txt',sep='\t',index=False)
del temp


# 2.3 query_prediction features
def split_prediction(text):
    if pd.isna(text): return []
    return [s.strip() for s in text.replace("{", "").replace("}", "").split(", ")]
train_df['pred_list'] = train_df['query_prediction'].apply(split_prediction)
train_df['pred_len'] = train_df['pred_list'].apply(len)

valid_df['pred_list'] = valid_df['query_prediction'].apply(split_prediction)
valid_df['pred_len'] = valid_df['pred_list'].apply(len)

test_data['pred_list'] = test_data['query_prediction'].apply(split_prediction)
test_data['pred_len'] = test_data['pred_list'].apply(len)
# 平均每个pred_len的点击次数
a = train_df['label'].sum()/train_df['pred_len'].nunique()
# 平均每个pred_len的搜索次数
b = train_df['label'].count()/train_df['pred_len'].nunique()
# prefix的点击次数和搜索次数
temp = train_df[['pred_len','label']].groupby('pred_len', as_index = False)['label'].agg({'predlen_click':'sum', 'predlen_count':'count'})

temp['predlen_click_smooth'] = temp['predlen_click']+a
temp['predlen_count_smooth'] = temp['predlen_count']+b
temp['predlen_click_log'] = np.log1p(temp['predlen_click'])
temp['predlen_count_log'] = np.log1p(temp['predlen_count'])
# prefixlen的点击数差值与搜索数差值
temp['predlen_click_diff'] = temp['predlen_click'] - temp['predlen_click'].mean()
temp['predlen_count_diff'] = temp['predlen_count'] - temp['predlen_count'].mean()
# pred的点击数占比与搜索数占比
temp['predlen_click_ratio'] = temp['predlen_click']/temp['predlen_click'].sum()
temp['predlen_count_ratio'] = temp['predlen_count']/temp['predlen_count'].sum()
# pred的绝对转化率
temp['predlen_ctr_abs'] = temp['predlen_click']/(temp['predlen_count'])
# pred的贝叶斯平滑转换率
temp['predlen_ctr_smooth'] = (temp['predlen_click']+a)/(temp['predlen_count']+b)
# pred的绝对转化率差值
temp['predlen_ctr_diff'] = temp['predlen_ctr_abs'] - temp['predlen_ctr_abs'].mean()
# pred的相对转化率
temp['predlen_ctr_rela'] = temp['predlen_ctr_diff']/(temp['predlen_ctr_abs'].mean())

train_df = pd.merge(train_df, temp, on='pred_len', how='left')
valid_df = pd.merge(valid_df, temp, on='pred_len', how='left')
test_data = pd.merge(test_data, temp, on='pred_len', how='left')
# temp.to_csv('d:/oppo/input/predlen_feature.txt',sep='\t',index=False)
del temp


# 2.4 title features
a = train_df['label'].sum()/(train_df['title'].nunique())
b = train_df['label'].count()/(train_df['title'].nunique())
# title的点击次数和搜索次数
temp = train_df[['title','label']].groupby('title', as_index = False)['label'].agg({'title_click':'sum', 'title_count':'count'})
temp['title_click_smooth'] = (temp['title_click']+a)/2
temp['title_count_smooth'] = (temp['title_count']+b)/2
# title点击数和搜索数的log变换
temp['title_click_log'] = np.log1p(temp['title_click'])
temp['title_count_log'] = np.log1p(temp['title_count'])
# 对prefix点击数和搜索数的log变换再做一次log变换
temp['title_click_log2'] = np.log1p(temp['title_click_log'])
temp['title_count_log2'] = np.log1p(temp['title_count_log'])
# prefix的点击数差值与搜索数差值
temp['title_click_diff'] = temp['title_click'] - temp['title_click'].mean()
temp['title_count_diff'] = temp['title_count'] - temp['title_count'].mean()
# prefix的点击数占比与搜索数占比
temp['title_click_ratio'] = temp['title_click']/temp['title_click'].sum()
temp['title_count_ratio'] = temp['title_count']/temp['title_count'].sum()
# prefix的点击数log占比与搜索数log占比
temp['title_click_ratio_log'] = np.log1p(temp['title_click'])-np.log1p(temp['title_click'].sum())
temp['title_count_ratio_log'] = np.log1p(temp['title_count'])-np.log1p(temp['title_count'].sum())
# prefix的绝对转化率
temp['title_ctr_abs'] = temp['title_click']/(temp['title_count'])
temp['title_ctr_abs'].describe()
# prefix的贝叶斯平滑转换率
temp['title_ctr_smooth'] = (temp['title_click']+a)/(temp['title_count']+b)
temp['title_ctr_smooth'].describe()
# prefix的绝对转化率差值
temp['title_ctr_diff'] = temp['title_ctr_abs'] - temp['title_ctr_abs'].mean()
temp['title_ctr_diff'].describe()
# prefix的相对转化率
temp['title_ctr_rela'] = temp['title_ctr_diff']/(temp['title_ctr_abs'].mean())
temp['title_ctr_rela'].describe()
# prefix的log绝对转换率
temp['title_ctr_log_abs'] = temp['title_click_log']-(temp['title_count_log'])
# prefix的log贝叶斯平滑转换率
temp['title_ctr_log_smooth'] = (temp['title_click_log']+np.log1p(a))/(temp['title_count_log']+np.log1p(b))
# prefix的log绝对转化率差值
temp['title_ctr_log_diff'] = temp['title_ctr_log_abs'] - temp['title_ctr_log_abs'].mean()
# prefix的log相对转化率
temp['title_ctr_log_rela'] = temp['title_ctr_log_diff']/(temp['title_ctr_log_abs'].mean())

train_df = pd.merge(train_df, temp, on='title', how='left')
valid_df = pd.merge(valid_df, temp, on='title', how='left')
test_data = pd.merge(test_data, temp, on='title', how='left')
temp.to_csv('d:/oppo/input/title_feature.txt',sep='\t',index=False)
del temp


# 2.5 title length features

# prefix长度特征
train_df['prefix_len'] = train_df['prefix'].apply(len)
valid_df['prefix_len'] = valid_df['prefix'].apply(len)
test_data['prefix_len'] = test_data['prefix'].apply(len)

train_df[['prefix_len','label']].groupby(['prefix_len'])['label'].describe()

train_df['prefix_len'] = train_df['prefix_len'].map(lambda x: 8 if x>8 else x)
valid_df['prefix_len'] = valid_df['prefix_len'].map(lambda x: 8 if x>8 else x)
test_data['prefix_len'] = test_data['prefix_len'].map(lambda x: 8 if x>8 else x)
train_df[['prefix_len','label']].groupby(['prefix_len'])['label'].describe()

# 平均每个prefix_len的点击次数
a = train_df['label'].sum()/train_df['prefix_len'].nunique()
a

# 平均每个prefix_len的搜索次数
b = train_df['label'].count()/train_df['prefix_len'].nunique()
b

# prefix的点击次数和搜索次数
temp = train_df[['prefix_len','label']].groupby('prefix_len', as_index = False)['label'].agg({'prefixlen_click':'sum', 'prefixlen_count':'count'})

temp['prefixlen_click_log'] = np.log1p(temp['prefixlen_click'])
temp['prefixlen_count_log'] = np.log1p(temp['prefixlen_count'])

# prefixlen的点击数差值与搜索数差值
temp['prefixlen_click_diff'] = temp['prefixlen_click'] - temp['prefixlen_click'].mean()
temp['prefixlen_count_diff'] = temp['prefixlen_count'] - temp['prefixlen_count'].mean()

# prefix的点击数占比与搜索数占比
temp['prefixlen_click_ratio'] = temp['prefixlen_click']/temp['prefixlen_click'].sum()
temp['prefixlen_count_ratio'] = temp['prefixlen_count']/temp['prefixlen_count'].sum()

# prefix的绝对转化率
temp['prefixlen_ctr_abs'] = temp['prefixlen_click']/(temp['prefixlen_count'])
# prefix的贝叶斯平滑转换率
temp['prefixlen_ctr_smooth'] = (temp['prefixlen_click']+a)/(temp['prefixlen_count']+b)
# prefix的绝对转化率差值
temp['prefixlen_ctr_diff'] = temp['prefixlen_ctr_abs'] - temp['prefixlen_ctr_abs'].mean()
# prefix的相对转化率
temp['prefixlen_ctr_rela'] = temp['prefixlen_ctr_diff']/(temp['prefixlen_ctr_abs'].mean())

train_df = pd.merge(train_df, temp, on='prefix_len', how='left')
valid_df = pd.merge(valid_df, temp, on='prefix_len', how='left')
test_data = pd.merge(test_data, temp, on='prefix_len', how='left')
del temp


# 2.6 tag features

# prefix的点击次数和搜索次数
temp = train_df[['tag','label']].groupby('tag', as_index = False)['label'].agg({'tag_click':'sum', 'tag_count':'count'})
# tag点击数和搜索数的log变换
temp['tag_click_log'] = np.log1p(temp['tag_click'])
temp['tag_count_log'] = np.log1p(temp['tag_count'])
# tag的点击数差值与搜索数差值
temp['tag_click_diff'] = temp['tag_click'] - temp['tag_click'].mean()
temp['tag_count_diff'] = temp['tag_count'] - temp['tag_count'].mean()
# tag的点击数占比与搜索数占比
temp['tag_click_ratio'] = temp['tag_click']/temp['tag_click'].sum()
temp['tag_count_ratio'] = temp['tag_count']/temp['tag_count'].sum()
# prefix的点击数log占比与搜索数log占比
temp['title_click_ratio_log'] = np.log1p(temp['tag_click_ratio'])
temp['title_count_ratio_log'] = np.log1p(temp['tag_count_ratio'])
# prefix的贝叶斯平滑转换率
temp['tag_ctr_smooth'] = (temp['tag_click']+a)/(temp['tag_count']+b)
temp['tag_ctr_smooth'].describe()
# prefix的绝对转化率差值
temp['tag_ctr_diff'] = temp['tag_ctr_abs'] - temp['tag_ctr_abs'].mean()
# prefix的相对转化率
temp['tag_ctr_rela'] = temp['tag_ctr_diff']/(temp['tag_ctr_abs'].mean())
# prefix的log绝对转换率
temp['tag_ctr_log_abs'] = temp['tag_click_log']-(temp['tag_count_log'])
# prefix的log贝叶斯平滑转换率
temp['tag_ctr_log_smooth'] = (temp['tag_click_log']+np.log1p(a))/(temp['tag_count_log']+np.log1p(b))
# prefix的log绝对转化率差值
temp['tag_ctr_log_diff'] = temp['tag_ctr_log_abs'] - temp['tag_ctr_log_abs'].mean()
# prefix的log相对转化率
temp['tag_ctr_log_rela'] = temp['tag_ctr_log_diff']/(temp['tag_ctr_log_abs'].mean())

train_df = pd.merge(train_df, temp, on='tag', how='left')
valid_df = pd.merge(valid_df, temp, on='tag', how='left')
test_data = pd.merge(test_data, temp, on='tag', how='left')
temp.to_csv('d:/oppo/input/tag_feature.txt',sep='\t',index=False)
del temp


# 2.7 combination features

# prefix&prefix_len, prefix_prediction_len, prefix&title, prefix&tag, prefix_len&prediction_len, prefix_len&title, prefix_len&tag, 
# prediction_len&title, prediction_len&tag, title&tag
items = ['prefix', 'prefix_len', 'pred_len', 'title','title_len', 'tag']
for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
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
        
        train_df = pd.merge(train_df, temp, on=item_g, how='left')
        valid_df = pd.merge(valid_df, temp, on=item_g, how='left')
        test_data = pd.merge(test_data, temp, on=item_g, how='left')
        temp.to_csv('d:/oppo/input/'+items[i]+items[j]+'.txt',sep='\t',index=False)
        del temp



# 3.1 Construct training set, validation set and test set
train_df_ = train_df.drop(['prefix','query_prediction', 'title', 'tag'], axis = 1)
valid_df_ = valid_df.drop(['prefix','query_prediction', 'title', 'tag'], axis = 1)
test_data_ = test_data.drop(['prefix','query_prediction', 'title', 'tag'], axis = 1)

# 3.2 training model
print('train beginning')

X = train_df_.drop(['label'], axis = 1)
y = train_df_['label']
X_valid = valid_df_.drop(['label'], axis = 1)
y_valid = valid_df_['label']
X_test = test_data_.drop(['label'], axis = 1)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 32,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

lgb_train = lgb.Dataset(X, y)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                verbose_eval=50,
                )

# f1-score on train data
print('f1-score on validation data:',f1_score(y, np.where(gbm.predict(X, num_iteration=gbm.best_iteration)>0.5, 1,0)))
# f1-score on validation data
print('f1-score on validation data:',f1_score(y_valid, np.where(gbm.predict(X_valid, num_iteration=gbm.best_iteration)>0.5, 1,0)))
print(gbm.best_score)

# 3.3 plot feature importance 
plt.figure(figsize=(12,6))
lgb.plot_importance(gbm, max_num_features=30)
plt.title("Featurertances")
plt.show()

# 3.4 predict test data
test_data['label'] = gbm.predict(X_test, num_iteration=gbm.best_iteration)
test_data['label'] = test_data['label'].apply(lambda x: round(x))
test_data['label'].describe()

# 3.5 save commit file
test_data['label'].to_csv('d:/oppo/submit/result201810132220.csv',header=None,index = False)
