---
### Here it is project  
# "CAR PRICE'S PREDICTON"  
---
    
Project consisted with several steps. 

1. [Parsing data](#parsing)
2. [Normalizing data, EDA and FE](#eda)
3. [Mashing Learning and submit](#ml)

### <a name="parsing"></a>Parsing data

The aim was to pars data from any page with auto advertisings. the first thing that comes to mind is the website [auto.ru](auto.ru).
It was my first time of parsing data from internet webpage. So I started out with some fears and for the begining - tryed to find any scrips, which will help me to jump there. I found this script from JANE VOYTIK ([KAGGLE](https://www.kaggle.com/eugeniavoytik/sf-dst-car-price-prediction-eda-ml), [Git](https://github.com/EugeniaVoytik/Car_price_prediction/blob/main/%5BSF-DST%20Car%20Price%20Prediction%5D%20Data%20parsing.ipynb)). And try to understend and to use it. Unfortunately, appeared that code was obsolete. Or, the right words will be "webpage have been updated with HTML markup". So I forced to make some change to code, to reach the goal.

My script consists with two parts: parsing urls of webpage with car advitising, and the second - parsing data for dataset from each webpage. 

Next difficults, which I collided, that webpage auto.ru is 'greedy'. If you open filtered page per brand-name, you will see a lot of pages - more then 100. And if you will try to open page number 100, 101 atc. - you will see advertising from some first pages (I mean 1, 2, 3). More of that, whten I tried to pars urls, just consistently opening pages, I realised that more then 75% of urls was duplicated. So, in list of 40+ urls I received only 10+ unique car webpage urls.

To work around this error, I changed the code to iterate through the page filter not only by the name of the car brand, but also by the model. Also, I set sorting on the page "in descending order of prices" and "output in the form of a table". The last thing: to make the page open faster, since there was no need to load images.

For the first time, I parsed data of all cars, used and new. But then on the second step I knew, that webpage of new car has different HTMP markup, and do not has all need features (from test dataset). So here I show updated script, for parsing only used car urls.

In the result, I received list of near 88 000 unique used car adertising urls. In a week, I made another attemp to receive more new urls, modifided script to take only last 7days advertising, and could dounload over 25000 urls. 

Anothe thing, which I needed to do - was to make notes and resumes in code, to to facilitate understanding. It is considering both parts of "JANE script".

The second part of parsing took me a lot of time. As I had 120 000 urls (as I mentiond above - used and new), total time to dounload data would be 60 hours. It's been a long time. This became the main difficulty in the work. So I would really like to understand how I can improve my program or what other approach to use so as not to waste so much time. 

### <a name="eda"></a> Normalizing data, EDA and FE.

<a name="ml"></a>
