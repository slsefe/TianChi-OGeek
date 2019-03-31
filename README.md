# TianChi-OGeek

## TianChi OGeek competition
[home page](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.6.5ead2784MTFoDV&raceId=231688)
performance: 400/2888, F1=0.7270
### 题目背景

在搜索业务下有一个场景叫实时搜索（Instance Search）,就是在用户不断输入过程中，实时返回查询结果。

此次赛题来自OPPO手机搜索排序优化的一个子场景，并做了相应的简化，意在解决query-title语义匹配的问题。简化后，本次题目内容主要为一个实时搜索场景下query-title的ctr预估问题。本次赛题为开放型算法挑战赛，优秀的解决方案会对我们解决这个场景以及其它场景下的问题带来极大的启发。我们期待优秀的你和你的团队能够投入进来！

### 题目内容

基于百万最新真实用户搜索数据的实时搜索场景下搜索结果ctr预估。

给定用户输入prefix（用户输入，查询词前缀）以及文章标题、文章类型等数据，预测用户是否点击。

文章资源类别非全网资源，属部分垂直领域内容。

初赛后期开放B榜开放时间3小时，请在三小时内提交结果；初赛结束时需要提交完整代码，最终晋级复赛前100名；复赛全程采用线上赛形式。

### 数据描述

此次初赛数据约235万 训练集200万，验证集5万，A榜测试集5万，B榜测试集25万，数据全部来源于OPPO日常搜索真实用户点击数据，数据集内可能存在重复、矛盾、同一搜索词对应多个点击等都是真实存在的。

数据格式： 数据分4列，\t分隔

|字段|	说明	|数据示例|
|---|---|---|
|prefix	|用户输入（query前缀）|	刘德|
|query_prediction	|根据当前前缀，预测的用户完整需求查询词，最多10条；预测的查询词可能是前缀本身，数字为统计概率|	{“刘德华”:  “0.5”, “刘德华的歌”: “0.3”, …}|
|title	|文章标题	|刘德华|
|tag	|文章内容标签	|百科|
|label|	是否点击	|0或1|

文本编码格式：UTF-8

### 评分标准

本次竞赛的评价标准采用F1 score 指标，正样本为1。计算方法参考https://en.wikipedia.org/wiki/F1_score

