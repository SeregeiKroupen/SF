### Project #9 
# "Batmoboile"   
**Data Science Specialization course by SkillFactory**   
[description of work and comments](https://github.com/SergeiKroupen/SF/blob/master/module_9/readme_ru.md) in russian. 

Project consisted with several steps. 

1. [EDA and FE](#eda) (*MultyInput_CarPricesPrediction_EDA.ipynb*)
2. [Neural Network for Structured, Tabular Data](#tdnn) (*multyinput-carpricesprediction-nn-v0-ipynb.ipynb*)
3. [Multi-input NN](#dlnn) (*multyinput-carpricesprediction-final-v0-ipynb.ipynb*)
4. [Resume](#estimate)
----
**additional files from module #9**   
5. [Image generator on NN GAN](#gan) (*DL9_IMG_Generate.ipynb*)   
6. [AI for CartPole game](#ai) (*dqn-cartpole.ipynb*)   

Main task of project is to predict price for car using DL instead of [ML](https://github.com/SergeiKroupen/SF/tree/master/module_5) (my module#5 project).

### <a name="eda"></a>EDA and FE

The naive ML model showed MARE=12.5% (best result) for Catboost Regressor. Several other models were tried, of which RandomForest Regr showed the closest result to the best. In the first stage, the data in the table was analyzed, cleaned, and new features were generated. In my opinion, the best result from feature generation is obtained from the 'name' vector, it has the best 'significance' according to the Catboost model.

There are also visible metric improvements from extracting digital data from the textual form of data in vectors, as well as some manipulation with them (mini-max, log normalization)

Applied cleaning of data that is not found in the training database, does not occur in the test. I consider this "noise", which will not be able to help the work of the regressor. As a result of the exclusion, improvements in the metric were noted.

After EDA, I wrote a function that manipulates data in datasets from the download to be clean and ready for ML, to use in subsequent work at other stages.

Further, I carried out an analysis of the cross-correlation of features and excluded several vectors from the work in the final, which made it possible to raise the result by several hundredths of points.

The result at the competition was 11.6153%

Next, I conducted an experiment to test my hypothesis. The main idea is to classify the records by the classification algorithm before the regressor works.

In the first step, I manually split the data by car brand and applied an intra-brand regressor to predict the corresponding results in the test set.
In the second stage, the KMeans algorithm was included. Based on the predicted clusters, the regressor was trained separately and each entry in the test was predicted. I made the final prediction in two ways.
I took the weighted average of the predictions, where the weights were the probabilities of referring this car to the calculated class.
I took the prediction strictly from the regressor who was trained in the class of this particular car.
Unfortunately, I did not manage to improve the metric. But I still think this approach is promising. You just need time and an analytical base for a successful application.

> * **Used libraries**: pandas, numpy, math, re, seaborn, tplotlib.pyplot, tqdm
> * **ML libraries**: sklearn, catboost, xgboost, lightgbm
> * **ML algoritms**: CatBoostRegressor, LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, XGBRegressor, LGBMRegressor, KFold, kneighbors_graph, AgglomerativeClustering
> * **ML methods**: MinMaxScaler, StandardScaler, category_encoders, LabelEncoder, TNSE

### <a name="tdnn"></a> Neural Network for Structured, Tabular Data

The first part is the use of DL for tabular dataset data.

Experiments with the design of neural networks have led to a very simple one with a small number of neurons in each layer of the network.

Next, I implemented the "feature forwarding" technique. Initially, the idea was that we have one very strong feature that can be used directly in the result ('name'), as well as two features with a good level of importance (according to yield from Catboost Regressor). I activated the first feature -  linearly, the rest - with a sigmoid. As a result, it turned out to make the best of all, a prediction. However, the result was still worse than ML.

It should be noted that initially I used a simple scheme, without choosing a specific vector for forwarding within the network, but only making a "wormhole".
Later, I tried the scheme with the choice of a specific vector in the dataset for forwarding. However, the result of the metric turned out to be no better than it was when using the "simple" scheme.

In the future, I continued to use this particular "simple" design of a tabular neural network in a multi-input network.

The second part is the use of DL to parse textual information from the 'description' vector

The work has been done only in terms of lemmatization and cleaning of data in ads from signs, that is, the standard mechanism for working with text has been applied.

As a result, the joint design of the network for the analysis of tabular information and textual information from the "description" field, manipulations with a set of vectors, gave only a slight improvement in the MARE metric, but still worse than the result of the ML model.

Already at this stage, the fact of the absence of computing power had a very strong influence, but more on that below.'

> * **Used libraries**: pandas, numpy, tqdm
> * **ML libraries**: sklearn, catboost,
> * **ML algoritms**: CatBoostRegressor, KFold
> * **ML methods**: category_encoders
> * **DL libraries**: tensorflow, tensorflow.keras, pymorphy2
> * **DL methods**: layers, Model, Sequential, Tokenizer, ModelCheckpoint, EarlyStopping

### <a name="dlnn"></a> Multi-input NN




> * **Used libraries**: pandas, numpy, tqdm
> * **ML libraries**: sklearn, catboost,
> * **ML algoritms**: CatBoostRegressor, KFold
> * **ML methods**: category_encoders
> * **DL libraries**: tensorflow, tensorflow.keras, pymorphy2
> * **DL methods**: layers, Model, Sequential, Tokenizer, ModelCheckpoint, EarlyStopping
### <a name="estimate"></a> Resume.

---

### <a name="gan"></a> Image generator on NN GAN


### <a name="ai"></a>AI for CartPole game

