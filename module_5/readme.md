---
### Here it is project  
# "CAR PRICE'S PREDICTON"  
---
    
Project consisted with several steps. 

1. [Parsing data](#parsing)
2. [Normalizing data, EDA and FE](#eda)
3. [Mashing Learning and submit](#ml)
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

### <a name="eda"></a> Normalizing data, EDA and FE.

According to the initial conditions of the project, we have dataset of car advertising 
without prices. We don't know price level in those days. But in Kaggle we can find 
dataset from auto.ru just from the same time (checed from vector 'unixtime_parsing').
So this help us to understand the level. But, finally turned out, that price level 
today (end november - early december 2021) is mush higher. I applied divider 1.6 
to current prices to meet best estimate.

As I said above, there is some difficult to pars data from webpages, because they 
change the data-structure and data-filings. So the main work was to reconcile data 
from test dataset with new parsed dataset. I cant pars data for vectors 'model date',
'model info', 'vehicleConfiguration', 'vendor'. I decided to delete 'model date' and
'nodel info' from features. 'vehicleConfiguration' and 'vendor' I reconstructed from
other vectors.

In another side, there are a lot of empty's in vectors 'complectation_dict', 
'equipment_dict' in test dataset. After all, they have some difference in structure 
from now. So, finally I took from 'equipment dict' information about 'conditioner' 
to make new feature, as it is important for our climate zone. Then - I drop both 
vectors from features.

Next I need to reconcile data such as the same case, same structure, same language. 
I made new vector 'model', because names of model year ago and now is differ a lot! 
For example 'model name': Mercerdes 'S-КЛАСС' and Mercedes 'S_KLASSE'. In dataset 
we have more 160 models, so manual replace would take a lot of time!

Some records with empty fields I deleted. Several features vectors ware deleted 
in case they filled with one-value data - they obviously unuseful.

To assess the significance of features for predicting the target price value, 
I wrote a function 'feature_imp'. It helped my to understand that a lot of features 
even harm prediction and it is better to remove them before ML. 

I tried my function with different ML algorithms, and realised that for different
cases better to make different feature set to achieve best results. I conceived
to implement this in future modeling, but unfortunately, I did not succeed in
stacking.

some of the steps at this stage could be attributed to the ML. But I left them here,
because the next step - 'car_price_ML' - I show the second version. There, only 
the modeling itself is already left, without hyper parameter settings. But I did 
that too - and I want to show it.

As the resume of searching hyper parameters I can say, that best score was with 
the base settings... I didn't understand why. May be I need to make GridSearch 
with found sets of 'good' settings?


### <a name="ml"></a> Mashing Leaning.

On this step I show my attempts to use different ways for prediction. I tried 
simple model like RandomForestRegressor. I tried to use bagging and gradient 
boosting from library *sklearn* and CatBoostRegressor. I tried stacking from 
mentioned library and I made my own stacking model (used some scripts from DST). 
But the best score (19,80% - 166 place) I received from CatBoostRegressor.

One more things: I always make some validation of my model, just to understand
where the my feature set and ML model have good or bad prediction power. 

As mentioned above, I made two attempts of parsing auto.ru. I decide to use the 
first part as train, and the second - as test. They are the same structure of 
brand-model, and the second parsed with setting "last seven days".

The results with CatBoostRegressor MAPE is 14%. Its is not fine, but it is like 
65 place in leaderboard as for 13/12/2021. So, it is not bad features dataset and 
ML-model for prediction.

### <a name="estimate"></a> Resume about the estimate on Kaggle.

I took part in competition with team 'penny drops'.

Unfortunately I didn't achieve my personal goal - 10%. My [result](https://www.kaggle.com/c/sf-dst-car-price-prediction/leaderboard)
is only 19,8%. However, it is much better then baseline. And my validate shown even
better results 14.3%.

After all, the result of my teammate is 15% (on Kaggle).

As the test set was collected more then year ago, I need to make some post production:
I divided results on divider 1.6. May be it is not good. But it is.

I have some comments about formulation of the project competition.

If we consider it as a theoretical problem, then it is quite standard, suitable 
for training. However, in reality, the task of determining the price in the market 
is not a random variable and to determine it, you need to directly use the data 
on the already available offers in the market. In other words, use a "data leak". 
In life, both sellers and buyers do this.

Also, in reality, the prices in the ads are, in a sense, virtual prices. It is 
not uncommon that the real price of buying and selling a used car differs 
from the virtual one. And from a practical point of view, it is the real price 
that is important, which we do not have. As part of the project, I did not try 
to build a model based on the behavior of participants in the formation of prices. 
But in practice, I would definitely use it, instead of a simple "prediction" of 
any ML.

### <a name="failures"></a> Failures.

I spent a lot of time to pars data from webpages. I'd love to understand how to 
speed up this work.

I am very glad to found opportunities to fast encoding of non numeric features - 
library 'cathegorial encoders'. But, I think, using it had the results of 'data 
leakage', so I didn't achieve better estimate on Kaglle. Wanted to understand 
if it is.

Working in team was the second time. This time was better then first. But not 
useful for me. As fact we have made separate project, but discussed our results.

My team partner name Сергей Фролов [Kaggle](https://www.kaggle.com/serfrol). 
As for begin, I asked him about project, it seemed he will make it alone and earlie
then me. But then after two weeks when I was in progress of parsing, it turned out, 
that he is the same stage - parsing too. It turned out to be such a mismatched start.

Next, we talk about some methods and findings. I gave him my parsed dataset 
and parsing script and EDA script. We agreed to make EDA and FE separate and then 
exchange of results, datasets and good ideas. Then we were going to separate some ML 
model to find best hyper parameters...  But in fact we both made all ML by self. And 
finally he receive score 15 on leaderboard. He gave me his dataset but I duidn't 
receive same result.

I think, it is good idea to separate work on EDA and FE, it is like a brainstorm. 
But it's not happened. 

### <a name="achievements"></a>My personal achivments

I began to better understand the ML library *sklearn*, among its tools, models, metrics
and more. I began to understand the ML syntax better. I found an excellent IDE software
DataSpell (JetBrains), which greatly helps to understand and speed up work on projects 
and tasks.

For the first time I did what I wanted to learn for a long time - I parsed pages from 
the Internet by the script. But, it is a pity, that very little attention was paid to 
this topic within the DST units. In particular, the application of the POST method has 
not been analyzed.

I am pleased that despite the difficulties I was able to adapt the script from the 
course to create a stacking model for regression.
