# TianChi-OGeek

### 赛题背景描述
[比赛官网](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.6.5ead2784MTFoDV&raceId=231688)

在搜索业务下有一个场景叫实时搜索（Instance Search）,就是在用户不断输入过程中，实时返回查询结果。此次赛题来自OPPO手机搜索排序优化的一个子场景，并做了相应的简化，意在解决query-title语义匹配的问题。简化后，本次题目内容主要为,基于百万最新真实用户搜索数据的实时搜索场景下query-title的ctr预估问题。给定用户输入prefix（用户输入，查询词前缀）以及文章标题、文章类型等数据，预测用户是否点击。


### 数据集说明

此次初赛数据约235万,其中训练集200万，验证集5万，A榜测试集5万，B榜测试集25万，数据全部来源于OPPO日常搜索真实用户点击数据，数据集内可能存在重复、矛盾、同一搜索词对应多个点击等情况。

数据格式： 数据分4列，\t分隔

|字段|说明|数据示例|
|---|---|---|
|prefix	|用户输入（query前缀）|	刘德|
|query_prediction	|根据当前前缀，预测的用户完整需求查询词，最多10条；预测的查询词可能是前缀本身，数字为统计概率|	{“刘德华”:  “0.5”, “刘德华的歌”: “0.3”, …}|
|title	|文章标题	|刘德华|
|tag	|文章内容标签	|百科|
|label|	是否点击	|0或1|

文本编码格式：UTF-8

### 评分标准

本次竞赛的评价标准采用F1 score 指标，正样本为1。计算方法参考https://en.wikipedia.org/wiki/F1_score

初赛成绩: 400/2888, F1=0.7270

### 代码使用说明

`main.py`为主函数,包括数据预处理, 特征工程, 模型训练和预测三个部分,所有的函数在`functions.py`中. Python版本为3.6,安装相应的包之后,运行`python main.py`即可.

### 项目结构说明
- input
    - oppo_round1_train_20180929.txt
    - oppo_round1_vali_20180929.txt
    - oppo_round1_test_A_20180929.txt
- preprocessed
    - train_preprocessed.pkl
    - valid_preprocessed.pkl
    - test_A_preprocessed.pkl
- features
    - prefix.pkl
    - ...
- models
    - lgbm.pkl
- images
    - lgbm_feature_importance.png
- submissions
    - lgb_2019-04-14.csv
- `main.py`
- `functions.py`

input包含解压得到的原始数据集,对其进行预处理后保存到preprocessed文件夹中,特征工程阶段得到的特征保存在features文件夹中,训练得到的模型持久化到models中,训练过程中得到的图片保存在images中,模型预测的结果保存在submissions中.