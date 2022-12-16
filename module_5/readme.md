---
### Here it is project  
# "CAR PRICE'S PREDICTON"  
---
    
Project consisted with several steps. 

1. [Parsing data](#parsing) (*car_prices_pasing.ipynb*)
2. [Normalizing data, EDA and FE](#eda) (*car_prices_eda.ipynb*)
3. [Mashing Learning and submit](#ml) (*car_prices_ML2.ipynb*)
4. [Resume about the estimate on Kaggle](#estimate)
5. [Failures](#failures)
6. [Achievements](#achievements)

### <a name="parsing"></a>Parsing data

The aim was to pars data from any page with auto advertisements. the first thing that comes to mind is the website[auto.ru](auto.ru). 
It was my first time parsing data from an internet webpage. So I started with some fears and in the beginning - tried to find any scrips, 
which will help me to jump there. I found this script from JANE VOYTIK([KAGGLE](https://www.kaggle.com/eugeniavoytik/sf-dst-car-price-prediction-eda-ml),
[Git](https://github.com/EugeniaVoytik/Car_price_prediction/blob/main/%5BSF-DST%20Car%20Price%20Prediction%5D%20Data%20parsing.ipynb)),
and try to understand and use it. Unfortunately, appeared that the code was obsolete. Or, the right words will be "webpage has been updated with HTML markup". So I was forced to make some changes to the code, to reach the goal. My script consists of two parts: parsing URLs of the webpage with car advising, and the second - parsing data for the dataset from each webpage.   

The next difficulty, which I collided with, is that the webpage auto.ru is 'greedy'. If you open a filtered page per brand name, you will see a lot of pages - more than 100. And if you will try to open page number 100, 101, etc. - you will see advertising from some first pages (I mean 1, 2, 3). More of that, when I tried to parse URLs, just consistently opening pages, I realized that more than 75% of URLs were duplicated. So, in a list of 40+ URLs, I received only 10+ unique car webpage URLs. To work around this error, I changed the code to iterate through the page filter not only by the name of the car brand but also by the model. Also, I set sorting on the page "in descending order of prices" and "output in the form of a table". The last thing: was to make the page open faster since there was no need to load images.   

For the first time, I parsed data of all cars, used and new. But then in the second step I knew, that webpage of the new car has different HTMP markup, and do not has all need features (compare with the test dataset). So here I show the updated script, for parsing only used car URLs. As the result, I received a list of nearly 88 000 unique used car advertising URLs. In a week, I made another attempt to receive more new URLs, modified the script to take only the last 7 days of advertisements, and could download over 25000 URLs.   

Another thing, I needed to do - was to make notes and resumes in code, to facilitate understanding. It is considering both parts of the "JANE script".
The second part of parsing took me a lot of time. As I had 120 000 URLs (as I mentioned above - used and new), the total time to download data would be 60 hours. It's been a long time. This became the main difficulty in the work. So I would like to understand how I can improve my program or what other approach to use so as not to waste so much time.

> **Used libraries**: requests, bs4, re, time, json

### <a name="eda"></a> Normalizing data, EDA and FE.

According to the initial conditions of the project, we have the dataset of car advertising without prices. We don't know the price level on those days. But in Kaggle we can find the dataset from auto.ru just at the same time (checked from vector 'unixtime_parsing'). So this helps us to understand the level. But, finally turned out, that the price level today (end November - early December 2021) is much higher. I applied divider 1.6 to current prices to meet the best estimate.   

As I said above, there is difficult to parse data from webpages, because they change the data structure and data filings. So the main work was to reconcile data from the test dataset with the new parsed dataset. I can't parse data for vectors 'model date', 'model info', 'vehicleConfiguration', 'vendor'. I decided to delete vectors 'model date' and 'model info' from features. Vectors 'vehicleConfiguration' and 'vendor' I reconstructed from other vectors.   

On another side, there are a lot of omissions in vectors 'complectation_dict', 'equipment_dict' in the test dataset. After all, they have some differences in structure from now. So, finally, I took from 'equipment dict' information about 'conditioner' to make the new feature, as it is important for our climate zone. Then - I drop both vectors from the dataset.   

Next, I need to reconcile data such as the same case, same structure, and same language. I made the new vector 'model' because the names of the model a year ago and now differ a lot! For example 'model name': Mercerdes 'S-КЛАСС' and Mercedes 'S_KLASSE'. In the dataset, we have more than 160 models, so manual replacement would take a lot of time!   

Some records with omissions I deleted. Several feature vectors were deleted in case they were filled with one-value data - they were obviously unuseful.
To assess the significance of features for predicting the target price value, I wrote a function 'feature_imp'. It helped me to understand that a lot of features even harm prediction and it is better to remove them before ML.   

I tried my function with different ML algorithms and realized that for different cases better to make different feature sets to achieve the best results. I conceived to implement this in future modeling, but unfortunately, I did not succeed in stacking.   

Some of the steps at this stage could be attributed to the ML. But I left them here because the next step - 'car_price_ML' - I show the second version. There, only the modeling itself is already left, without setting hyperparameters. But I did that too - and I want to show it. As the resume of searching the hyperparameters, I can say, that the best score was with the base settings... I didn't understand why. Maybe I need to make GridSearch with found sets of 'good' settings?   

> * **Used libraries**: pandas, numpy, math, re, seaborn, tplotlib.pyplot, tqdm
> * **ML libraries**: sklearn, catboost, xgboost, lightgbm
> * **ML algoritms**: CatBoostRegressor, LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, XGBRegressor, LGBMRegressor, KFold
> * **ML methods**: RandomizedSearchCV, category_encoders, LabelEncoder

### <a name="ml"></a> Mashing Leaning.

In this step, I show my attempts to use different ways for prediction. I tried a simple model like RandomForestRegressor. I tried to use bagging and gradient boosting from the library sklearn and CatBoostRegressor. I tried stacking from the mentioned library and I made my own stacking model (using some scripts from DST). But the best score (19,80% - 166 place) I received was with CatBoostRegressor.   

And one more thing: I always make some validation of my model, just to understand where my feature set and ML model have good or bad prediction power.
As mentioned above, I made two attempts at parsing auto.ru. I decide to use the first part as the train, and the second - as the valid. They are the same structure of brand-model.   

The result with CatBoostRegressor MAPE is 14%. It is not fine, but it is like 65th place on the leaderboard (as of 13/12/2021). So, it is not a bad feature dataset and ML model for prediction.   

> * **Used libraries**: pandas, numpy, tqdm
> * **ML libraries**: sklearn, catboost, xgboost, lightgbm
> * **ML algoritms**: CatBoostRegressor, LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, XGBRegressor, LGBMRegressor, Ridge, AdaBoostRegressor, StackingRegressor, RandomTreesEmbedding, KFold
> * **ML methods**: RandomizedSearchCV, category_encoders, LabelEncoder

### <a name="estimate"></a> Resume about the estimate on Kaggle.

I took part in the competition with the team 'penny drops'.  

Unfortunately, I didn't achieve my personal goal - MAPE=10%. My [result](https://www.kaggle.com/c/sf-dst-car-price-prediction/leaderboard)
is only 19,8%. However, it is much better than the baseline. And my validation shows even better results - 14.3%. After all, the result of my teammate is 15% (on Kaggle).  

As the test set was collected more than a year ago, I need to make some post-production: I divided results on divider 1.6. Maybe it is not good. But it is no way better.   

I have some comments about the task of the project competition. If we consider it as a theoretical problem, then it is quite standard, and suitable for training. However, in reality, the task of determining the price in the market is not a random variable, and to determine it, you need to directly use the data on the already available offers in the market. In other words, use a "data leak". In life, both sellers and buyers do this.   

Also, in reality, the prices in the ads are, in a sense, virtual prices. It is not uncommon that the real price of buying and selling a used car differs from the virtual one. And from a practical point of view, it is the real price that is important, which we do not have. As part of the project, I did not try to build a model based on the behavior of participants in the formation of prices. But in practice, I would definitely use it, instead of a simple "prediction" of any ML.   

### <a name="failures"></a> Failures.

I spent a lot of time parsing data from web pages. I'd love to understand how to speed up this work.   

I am very glad to find opportunities to fast encoding of non-numeric features - library 'categorial encoders'. But, I think, using it had the results of 'data leakage', so I didn't achieve a better estimate on the Kaglle. Wanted to understand if it is.   

Working in the team was the second time. This time was better than the first. But not useful for me. As the fact we have made separate projects but discussed our results.   

My team partner's name is Сергей Фролов [Kaggle](https://www.kaggle.com/serfrol). As for the beginning, I asked him about the project, it seemed he will make it alone and earlier than me. But then after two weeks when I was in the progress of parsing, it turned out, that he is the same stage - parsing too. It turned out to be such a mismatched start.   

Next, we talk about some methods and findings. I gave him my parsed dataset and parsing script and EDA script. We agreed to make EDA and FE separate and then will exchange results, datasets, and good ideas. Then we were going to separate some ML models to find the best hyperparameters... But, in fact, we both made all ML by ourselves. He gave me his dataset but I didn't receive the same result.   

I thought it is a good idea to separate work on EDA and FE, it is like brainstorming. But it's not happened.

### <a name="achievements"></a>My Achivments

I began to better understand the ML library sklearn, among its tools, models, metrics, and others. I began to understand the ML syntax better. I found an excellent IDE software DataSpell (JetBrains), which greatly helps to understand and speed up work on projects and tasks.   

For the first time, I did what I wanted to learn for a long time - I parsed pages from the Internet by the script. But, it is a pity, that very little attention was paid to this topic within the DST units. In particular, the application of the POST method has not been analyzed.   

I am pleased that despite the difficulties I was able to adapt the script from the course to create a stacking model for regression.
