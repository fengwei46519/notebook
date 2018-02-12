# tensorflow入门学习
![tensorflow架构](https://github.com/fengwei46519/notebook/blob/master/dnn/images/tensorflow_programming_environment.png)

## (1) High Level APIs
### 1. Estimator
Estimators encapuslate the following actions: **training, evalutation, prediction, export for serving**.
```python
properties:
  config: Configuration object.
  model_dir: Directory to save model parameters, graph and etc.
  model_fn: Model function.
  param: dict of hyper parameters that will be passed into model_fn.
function: 
  __init__:
  evaluate:
  predict:
  train:
  export_savedmodel:
```
- **pre-made estimator** create and manage **Graph** and **Session** objects for you. DNNClassifier, DNNRegressor, LinearClassifier, LinearRegressor, DNNLinearCombinedClassifer, DNNLinearCombinedRegressor。
```python
# Structure of a pre-made Estimators program
#1. inporting data
def input_fn(dataset) :
    return feature_dict, label
#2. 定义特征列
##Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')  #feature name
crime_rate = tf.feature_column.numeric_column('crime_rate')   # feature type
median_education = tf.feature_column.numeric_column('median_education',
                                                    normalizer_fn='lambda x: x - global_education_mean') # scale the raw data
#3. Estimators实例化
#实例化Estimator, name是LinearClassifier，并传入特征列
estimator = tf.estimator.Estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )

#4. 调用training, evaluation, or inference methon
estimator.train(input_fn=my_training_set, steps=2000)
```
- **custom estimator** : 自己实现model function。
![estimator类型](https://github.com/fengwei46519/notebook/blob/master/dnn/images/estimator_types.png)
```python
#自定义实现my_model
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
       'n_classes': 3,
   })
```
### 2. Importing Data
image: 训练样本、测试样本的选取（random）；text:  raw text data->embedding indentifiers。

## (2) Low Level APIs

**graph**: 构建计算网络。
-**operations**: the nodes of the graph;
-**tensors** : the edges in the graph.
```python
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
```
**session**: 运行网络。
```python
sess = tf.Session()
print(sess.run(total))
```

## (3) tensorboard
可视化工具。https://www.cnblogs.com/fydeblog/p/7429344.html

