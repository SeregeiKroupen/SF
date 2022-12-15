
### Project #3.
# "About tasty and healthy food..." 
#### Data Science Specialization course by SkillFactory
--- 
[description of work and comments](https://github.com/SergeiKroupen/SF/blob/master/module_3/readme_ru.md) in russian.
   
The format of the project is Jupiter Notebook. **It is my first full-cycle Data Science Machine Learning project.** 
The task of the project is to predict the target vector "restaurant rating" using Random Forest ML-algorithm.

The dataset is preloaded restaurant data from the webpage of "Tripadvisor". The train part of the dataset has 40k rows 
and 9 vectors (features), and the target vector. The test part has 10k rows.

The project (*TripAdviser_3.ipynb*) consists of all parts of the ML cycle: load and cleansing data, exploratory analysis, filling omissions, 
features engineering, preparing, and sometimes transforming data for the ML model, and making dummy variables. 
This project shows working with various types of data, among them - the extraction of new features from lists of data 
contains in cells. 

The exploratory analysis shows that duplicates in the id vector exist both in the train and the test parts of the dataset. 
This is confusing since the ID must be unique in such kinds of datasets. The prohibition of deleting any row from the test 
part requirement leads to a decrease in the accuracy of the model's results.

Evaluation of the quality of the job is carried out through participation in the Kaggle competition, where is needing to submit 
a prediction for the test part of the dataset. The result of MAE on the spittest of my model = 16,3%, 
on Kaggle - 16,86% (55th place from 692). I aimed to achieve a result better than 18%.


Used libraries: **pandas, itertools, seaborn, matplotlib, scipy.stats.ttest_ind, random, re, math**   
ML-library: **sklearn** train_test_split, metrics, ensemble.RandomForestRegressor

PS. file *"TripAdviser_3-Kaggle.ipynb"* just a copy to load to Kaggle.
