### the Data Science final course project 
the project name:   
# "Make profitable investments by predicting the price of houses!"

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


p. 1 _ Final_project_EDA.ipynb  
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


