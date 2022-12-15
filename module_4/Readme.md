### Project #4 
# "Computer says: "No!"   
**Data Science Specialization course by SkillFactory**   
[description of work and comments](https://github.com/SergeiKroupen/SF/blob/master/module_4/readme_ru.md) in russian.   

---

Second edition ("*credit_scoring_v2*")

I made the second edition of project #4, as my first yield was not successful. I return to it after finishing 
the study of DS in SkillFactory. So I used more knowledge to achieve a better result.   
I used CatBoostClassifier and TargetEncoder as the base of the ML model. 
And the most useful thing was balancing classes. Initially, we had a near 7:1 balance between the "non-default" 
and the "default" classes, and I made a balance near 4:3.   

The score on the [leaderboard](https://www.kaggle.com/competitions/sf-scoring/leaderboard) is 35.032%, 
it is 12th out of 76 (F1 of default class metric). This competition has changing rules - the evaluation metric is 
F1(default class), instead of ROC AUC, which was previously.

Used libraries: pandas
ML-library: **sklearn, catboost**   
ML models: **RandomForestClassifier, LogisticRegressor, LogisticRegression, SGDClassifier, RidgeClassifier, 
ExtraTreesClassifier, CatBoostClassifier**   
ML methods: **GridSearchCV, RandomizedSearchCV, PCA, StandardScaler, category_encoders**   

---

First edition ("*credit-scoring-penny-drops.ipynb*")

The format of the project is Jupiter Notebook. It is a full-cycle Data Science Machine Learning project. The main task of the project is to predict credit default using the Logistic Regression algorithm. But the Random Forest Classification algorithm is also used. The dataset with personal data has 18th feature vectors and the target vector. The train part has about 74k rows, and the test part has 36k rows. 

The project consists of all parts of the ML cycle: load and cleansing data, exploratory analysis, filling omissions, features engineering, preparing, and sometimes transforming data for the ML model. This project shows working with various types of data. Also, I tried to implement target encoding for features, made myself.

The evaluation metric for this project at first was ROC-AUC. And I reached an 83% result. 

Besides the common pipeline of the ML process, I made the selection of hyperparameters for both ML models (RFC and LR) in several ways: with random search, search by the grid, and by PCA (principal component analysis). The metrics on the validation set grew to 85,67%. But finally, after submission on Kaggle, the result turn out to be 61%. I think my target encoding led to data leakage.

To avoid possible data leakage I was forced to drop my "best vector". But ROC AUC grew only to 72%. I was dissatisfied with this, but to repair or re-make all work at this time couldn't do. Several months later, after finishing the course, I return to the project to make it better.


Used libraries: **pandas, itertools, seaborn, matplotlib, scipy.stats.ttest_ind, random, re, math**
ML-library: **sklearn** 
ML models: **RandomForestClassifier, LogisticRegressor**
ML methods: **GridSearchCV, RandomizedSearchCV, PCA, StandardScaler**
