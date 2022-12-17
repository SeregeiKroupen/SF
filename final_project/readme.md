### the Data Science final course project 
the project name:   
# "Make profitable investments by predicting the price of houses!"
[description of work and comments](https://github.com/SergeiKroupen/SF/blob/master/final_project/readme_ru.md) in russian.    

The original plan was generally standard, and the goal was more modest:
1) Clean, analyze, and process data - EDA.
2) Apply ML models for prediction.
3) Apply a neural network for prediction.

However, in the process of analyzing the dataset, it turned out that a large number of records there is information about the price per square foot 
and the area of the property itself. Thus, it turned out,   there is no need to "predict" house prices. And I got the idea, I could build a working 
model for finding profitable sales offers to consider investment opportunities. That is, to find offers with a lower price.
Another point, I wanted to try on the dataset my pre-clustering idea. It means, making clusterization and applying the regression model for each 
cluster separately. I've already tried earlier to apply this scheme in a previous project but did not get significant results.

Thus, the final plan became the following:
1) EDA & FE
2) Classic ML by different models
3) Manual dataset clustering and ML regression within clusters
4) ML dataset clustering and ML regression within clusters
5) Building a Neural Network
6) Applying a Neural Network inside clusters
7) Production: proposal of a mechanism for application in practice

## Notebooks and data

p. 1  Final_project_EDA.ipynb  
pp. 2 - 4 and 7 - Final_project_ML.ipynb  
pp. 5-6 - Final_project[ML+NN] [Kaggle](https://www.kaggle.com/code/sergeikroupen/final-project-ml-nn/edit)  
A pre-processed dataframe after the EDA&FE phase is also posted on [Kaggle](https://www.kaggle.com/datasets/sergeikroupen/housing-preprocessed-data)

## EDA & FE

Two very unpleasant things came up at the stage of processing the initial data.

The first is that there are a lot of duplicate records in the dataset. According to my estimate - from 1/3 to 2/5 of the total volume. 
By duplicate entries, I mean a combination of the full address (house + street + city + state). Moreover, the standard `pd.DataFrame.drop()` 
methods will not be able to clean up, since there are errors in the records: different spellings, extra characters, permutation - a complete set 
of "noise", which is very laborious to work with.    
In addition, when starting the task of cleaning the data twice, and deleting some of the duplicates, I inexplicably encountered a deterioration 
in the metrics of the ML model. Thus, I gave up on this venture.   

The second trouble is a huge number of errors and inaccuracies both in the accompanying information, for example, "the presence of a swimming pool", 
as well as in the critically important one - namely, "real estate area". I've found these errors many times just by googling the addresses from 
the dataset.    
Moreover, what is the saddest thing, among the duplicates, as a rule, there was only one entry with completely correct information, while 
the rest contained incomplete or distorted versions. I think this is why the "simple" (actually not) removal of duplicates did not improve 
the metric - the correct information was mostly lost.   

First of all, the issue of choosing a metric was seen as very important. While cleaning data and generating features, I looked at a wide set. Given the chosen target, a metric was needed to track that the model was predominantly wrong "down", that is, it predicted a smaller value than the actual asking price. This is necessary to insure against a false conclusion when the model's prediction turns out to be higher than the market price and the object turns out to be a bad investment.   
I found the Mean Pinball Loss metric - "a metric of average losses in a certain quantile". It is implemented in `sklearn`, but for some reason, it was not imported locally. So I wrote my function with additional result parameters: in addition to MPL, I also counted the proportion of "overestimated" values, as well as the average of "overestimated" and "underestimated" values. By choosing alpha=0.01, I focused on minimizing the "overestimated" value.   
____
$$Pinball(y, \\widehat{y}) = \frac{1}{n}\sum_{i=0}^{n-1}\alpha * max(y_{i}-\widehat{y}_{i},0)+(1-\alpha) * max(\widehat{y}_{i}-y_{i},0)$$

[Fomula and description](https://scikit-learn.org/stable/modules/model_evaluation.html#pinball-loss)
___

An analysis of the target - price - showed an extremely wide range of values: in fact, from zero to $200 million. The realization of this fact led me to the idea of definitely trying clustering since it is not possible to build a model with acceptable accuracy values for such a range of price fluctuations. In addition, there is a cluster of real estate (very expensive, apparently), the price of which non-trivially depends on many other factors, in addition to simply listing bathrooms, a pool, or even a place on a map.   

In general, I was very caring about information in vectors. That is, when extracting information from vectors (numerical data, classification information), as a rule, I did not delete the origin vector (only at the modeling stage - I simply excluded it) to evaluate the importance of each vector at the end, despite the obvious duplication of data.   

I decide to delete some of the records from the dataset due several means:
- entries indicating that this is a development plan, there are no houses yet (~ 2.5K)
- entries without an address (how can I find them later?) (~1.7K)
- entries with a clear indication of the sale of land, not houses (~ 31K)
- full duplicate entries (address+area+price) (~24K)
- entries with zeros in the columns "area" and "area of the lot with the land" (~ 14K)

Just after the next removal of such records, the MAPE metric improved dramatically.

Once again, I was convinced that to work with data, you need to deeply understand the subject that these data characterize. This a simple example of the knowledge I gleaned from figuring out what the numbers are in the "bathrooms" vector. There were numbers 2.5, 2.75, 2.1... It turned out that this feature, which we can not meet in Russia or Belarus. "Bathroom" Americans consider a room with a shower, bathtub, washbasin, and toilet. Bathrooms that have not all 4 things are considered "incomplete bathrooms". Further, there are two approaches to reflect this information. The old approach considered full bathrooms as whole numbers and not full ones as half of the numbers. For example numbers like 1.5, 2.5, etc. A variant of this approach: accounting for how many of the four things are present in the bathroom so that as many quarters indicate an incomplete bath. This approach implements options 2.75 or 3.25 for example. Alas, often in this approach, the numbers of bathrooms are simply added up, and a completely distorted picture is obtained: 1 + 0.5 + 0.75 = 2.25 - it is not possible to understand by the figure how many and which bathrooms are in the house - one can only guess with probability. The second approach calls for separately counting "full" and "incomplete" bathrooms and writing them through a dot: 3.2, 2.1. This approach, in my opinion, is the most transparent and understandable. But alas, it is not possible to bring all the data to it into a dataset. Therefore, I implemented rounding up, as a kind of compromise.

## ML

ML modeling did not bring any surprises, and CatBoost Regression turned out to be the best model: MAPE=4.09%, Pinball = 17 168

The manual clustering of the dataframe was performed on the target vector with ranges of $500,000, $1,500,000, and at the end, ranges up to $10M, $20M, and $100M. most likely not real estate, but a variant of rent or land, not excluded by obvious signs. As a result of this operation, we have already managed to obtain metric values better than in the usual scheme: for MAPE up to 1.06% (in the vast majority of clusters - better than 3%) for Pinball - even better, "revaluation" averaged less than 2% of the price in the cluster.

But the best results were obtained after ML clustering (K-Means algorithm). He proposed to the algorithm to cluster by 14 clusters (the same number was at the previous stage). It so happened that in several clusters there was a minimum number of records - they had to be excluded from further modeling along the way, in order to avoid errors.

I initially decided to apply two approaches to calculating the price prediction:
+ calculates the weighted average prediction for all models of all clusters for each record, where the weights will be the reciprocals of the distances to the centroids ('price_sumprod')
+ direct calculation of the model prediction calculated for the cluster of the current record ('price_kmeans')

As a result, it turned out that the calculation of the metric for the entire Dataframe using the "price_kmeans" vector gave the result: MARE = 2.13%, Pinball = 12 126
What turned out to be better than simple ML modeling. And the analysis of metrics within clusters in some cases went below 1% error.

Probably, it would be possible to stop here, since I can’t imagine how it is even better to make a forecast if the inaccuracy is only 1% already now. But I decided to build a neural network and try to get at least the same results.

## Neural network
Multilayer, multi-parameter neural networks showed consistently poor results. Any "decent" figures were obtained only for a simple 2-3 layer network of 128-256 neurons per layer. But, the indicator MAPE I never managed to get less than 100%. The neural networks also refused to show a stable result, and after 1000 epochs they overtrained.

To complete the experiment with the neural network, I decided to try using the clusters obtained in the previous step. And it turned out that it worked! I tried only one cluster, but even from the start, a standard picture of learning dynamics (hyperbole) was obtained in it, although the dynamics on the entire dataset were more like a pulse. As a result, after about 1000 epochs, I got MAPE = 3-5%. This is a good result, which further confirms the correctness of the method of pre-clustering data before modeling, although it turned out to be worse than the results of the regression model. But, it is better than many other regression models, except for CatBoost Regressor.

## Business production
So, the idea is to select a cluster with intervening accuracy (K-Means clustered into clusters with a wide range of prices), and get a list of houses where the price is significantly lower than predicted - it means that it was underestimated for sale, compared to similar offers.
I chose the TOP delta between price and predictions. Of course, the most expensive offers came out. To work in this segment, a loan of 60 million dollars is needed.
But you can choose the most popular prices (up to 750 thousand). in this case, investments will require 3.5 million dollars.
Based on the assumption that the return on investment will occur within a year, the yield is 10-12% per annum (typical for all segments). Although the market shows that the turnover is faster, the return of funds may occur within a quarter.
