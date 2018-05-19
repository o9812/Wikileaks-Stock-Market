# Predicting exchange rate change with diplomacy cables

- This project's objective is to measure how confidential information affects the financial market. We know that market efficiency is based on the assumption that all market participants have fair access to the market. More importantly, it includes any kind of information. However, people do possess confidential information in the real world. Most financial markets have prohibited any kind of inside trading. Because of the difficulties to access confidential data, detecting the influence of confidential information affect market is hard. 

- Luckily, we now have [Public Library of US Diplomacy](https://wikileaks.org/plusd/about/) dataset. We scraped documents from 2000 to 2010. With the scrapped data, we built models to detect if we can predict the abnormal change of the exchange rate in different countries. The historical exchange rate is fetched from CRSP datasets and this dataset covers 21 countries, which are Australia, Brazil, Canada, China, Denmark, Hong Kong, India, Japan, Korea, Malaysia, Mexico, New Zealand, Norway, Sweden, South Africa, Singapore, Sri Lanka, Taiwan, Thailand, United Kingdom and Venezuela. 

## Data set

We have two datasets: Public library of US diplomcy and the exchange rate dataset crapped from CRSP. 
For the purpose of error analysis, we split data by country and year. You can access data through the links below. 
  
- [country by country](https://drive.google.com/drive/folders/1uHIfkPc2b-b_3XDnRJn3NO2baRxnrXz5?usp=sharing)
  - date:            the date of wikileaks cable
  - content:         the content of wikileaks cable
  - exchange rate:   the 15 days log return of exchange rate before time t 
  - numerical label: log return of exchange rate at time t
  - dummy label:     abnormal log return or not
- [country by country - negative](https://drive.google.com/drive/folders/1wzG2AGAE3wy-v-GdwZlsixZoV6yVycot?usp=sharing)
ONLY KEEP THE INSTANCE WITH NEGATIVE NUMERICAL LABEL
  - date:            the date of wikileaks cable
  - content:         the content of wikileaks cable
  - exchange rate:   the 15 days log return of exchange rate before time t 
  - numerical label: log return of exchange rate at time t
  - dummy label:     abnormal log return or not

#### We also include a joint table of all conuntries's infromation named as `final_All_countries`. 

- [year by year](https://drive.google.com/drive/folders/1DMejBtKP9QGcnsybepXAuWAlqLqSIahR?usp=sharing)
  - date:            the date of wikileaks cable
  - content:         the content of wikileaks cable
  - exchange rate:   the 15 days log return of exchange rate before time t 
  - numerical label: log return of exchange rate at time t
  - dummy label:     abnormal log return or not
- [year by year - negative](https://drive.google.com/drive/folders/1l8YtosubkGm4T4Wbi2qnv3sFpFKK3ciy?usp=sharing)
ONLY KEEP THE INSTANCE WITH NEGATIVE NUMERICAL LABEL
  - date:            the date of wikileaks cable
  - content:         the content of wikileaks cable
  - exchange rate:   the 15 days log return of exchange rate before time t 
  - numerical label: log return of exchange rate at time t
  - dummy label:     abnormal log return or not

And the following is a joined table, cluding all countries and all year
- [all by all](https://drive.google.com/drive/folders/1gJhyw0p9Ha6C4Yd2yq6P6YYSFU4QNlLD?usp=sharing)
  - date:            the date of wikileaks cable
  - content:         the content of wikileaks cable
  - exchange rate:   the 15 days log return of exchange rate before time t 
  - numerical label: log return of exchange rate at time t
  - dummy label:     abnormal log return or not
- [all by all - negative](https://drive.google.com/drive/folders/1oQkmha0nOgHo6tlEjD9SO1VKJ4f-HtAg?usp=sharing)
ONLY KEEP THE INSTANCE WITH NEGATIVE NUMERICAL LABEL
  - date:            the date of wikileaks cable
  - content:         the content of wikileaks cable
  - exchange rate:   the 15 days log return of exchange rate before time t 
  - numerical label: log return of exchange rate at time t
  - dummy label:     abnormal log return or not

## Build the regression and classfication model
To build the Random Forest Regression Model: 
1. Change working directory to the directory of `RanFrst_regres.py`. 
2. Run the python script as follwoing instructions:

### if user is interested in classifier function in juypter notebook, we have the following two examples
- `All_Countries_Neg_AUC_May17th.ipynb`	: Do the classification and AUC plot
- `Feature_Importance_All_Columns.ipynb`: finding out the feature importance 


#### searching the file name was written in hard code. So, if you want to rename the data, you would need to modify the main function.
- For example, the folder of Country by Country and Year by Year 

### run random forest classifier 
```
python RanFrst_classfy.py 10 ./data_country/ country_10 -country
```
```
python RanFrst_regres.py 10 ./data_year/ country_10 -year
```
Here, the pararmeters:
> - `10`: the number of estimators in random forest model
> - `./data_country/`: input data path, supposed data is stored under `./data_country/`
> - `country_10`: output data path, it would automaticall create a directory called `./output_country_10/`
> - `-country`: let the model know it is searching what kind of data (country level or year)
> - `-year`: let the model know it is searching what kind of data (country level or year)

- Year by Year
### run random forest classifier regression
```
python RanFrst_regres.py 10 ./data_country/ year_10 -all
```
```
python RanFrst_classfy.py 10 ./data_year/ year_10 -all
```
> - `10`: the number of estimators in random forest model
> - `./data_year/`: input data path, supposed data is stored under `./data_year/`
> - `year_10`: output data path, it would automaticall create a directory called `./output_year_10/`
> - `-all`: means run all feature functions with text, exchange rate and mix features. `-text`, `-price` and `-mix` only run the functions. This can help speed up the time computation.

## Result - regression
`RanFrst_regres.py` would automatically generate output under the user defined output directory, ex `./output_country_10/` or `./output_year_10/`. 
- Country by Country
It would generate 22 files named by the country name, ex: the input `final_Single_mexico` file would generate file `mexico` with 
> - mse: mean square error
> - mae: mean absolute error
> - median_absolute_error: median absolute error
> - r2: r squre score
> - (2086, 5): total 2086 instances with 5 features. (it's the size of input dataframe)

| Country                       | mexico       | 
| -------------                 |-------------| 
| mse_text is                   |1.979e-05 |
| mae_text is                   | 1.979e-05  | 
| median_absolute_error stripes |  0.00263002     |
| r2_text                       | -1.20974962      |
| -------------                 |-------------| 
| mse_price is                   |8.6e-07|
| mae_price is                   | 0.0004434 | 
| mdn_ae_price stripes          |  0.00011176    |
| r2_price                       | 0.9034855     |
| -------------                 |-------------| 
| mse_mix                       |6.86e-06|
| mae_mix                       | 0.00144322 | 
| mdn_ae_mix                    | 0.00040706   |
| r2_mix                        | 0.23445956     |
| -------------                 |-------------| 
(2086, 5)

- Year by Year
It would generate 10 files named by the year. ex. the input 2003 files would generate `2003` with 
> - mse: mean square error
> - mae: mean absolute error
> - median_absolute_error: median absolute error
> - r2: r squre score
> - (1233, 5): total 1233 instances with 5 features. (it's the size of input dataframe)

| Year                     | 2003       | 
| -------------                 |-------------| 
| mse_text is                   |6.79e-06|
| mae_text is                   | 6.79e-06  | 
| median_absolute_error stripes |  0.00162758    |
| r2_text                       | -0.03564915     |
| -------------                 |-------------| 
| mse_price is                   |4.17e-06|
| mae_price is                   | 0.00126901| 
| mdn_ae_price stripes          |  0.00069519  |
| r2_price                       | 0.36338027    |
| -------------                 |-------------| 
| mse_mix                       |5.36e-06|
| mae_mix                       | 0.00179364| 
| mdn_ae_mix                    | 0.00137725   |
| r2_mix                        | 0.18312545     |
| -------------                 |-------------| 
(1233, 5)

## Output - classify

`RanFrst_classfy.py` would automatically generate output under the user defined output directory, ex `./output_country_10/` or `./output_year_10/`. What's more, it would automatically create a folder and store three generated AUC figures.
- Country by Country
It would generate 22 files named by the country name, ex: the input `final_Single_australia` file would generate file `australia` with the following table, 
> - fpr: increasing false positive. in this example only has one treshold
> - tpr: increasing true positive. in this example only has one treshold
> - roc_auc: increasing accuracy, computing area under the receiver operating characteristic curve (ROC AUC)

| Country                       | australia       | 
| -------------                 |-------------| 
| fpr is:                       |[ 0.          0.06140351  1.        ]|
| tpr is:                       |[ 0.  0.  1.]  | 
| roc_auc is                    |   0.469298245614   |
| -------------                 |-------------| 
| fpr is:                       |[ 0.          0.00877193  1.        ]|
| tpr is:                       |[ 0.          0.85714286  1.        ]| 
| roc_auc is                    | 0.924185463659 |
| -------------                 |-------------| 
| fpr is:                       |[ 0.          0.06140351  1.        ]|
| tpr is:                       | [ 0.          0.23809524  1.        ]| 
| roc_auc is                    | 0.588345864662|
| -------------                 |-------------| 
(1350, 5)

and three figures showing roc_aucc under `./output_country_10/mexico_figure/`

## Result:
<!---
 We used NYU prince hpc to run experiment
***
 - Random Forest Regression
 > #### year by year:
 > [Random forest regression w/ 10 estimators(https://drive.google.com/drive/folders/1yL5M089DjRimlrfyW15sZ9JEW4zo0tU6?usp=sharing)
***
- Random Forest Classification
> #### year by year:
> [Random forest classify w/ 10 estimators](https://drive.google.com/drive/folders/1w7o3YCzaje1xn_P7dzdQ1ZzVp2GkbLDm?usp=sharing)
> #### country by country:
> [Random forest classify w/ 10 estimators](https://drive.google.com/drive/folders/1jB1Bjm2rpqIjy0ehX5ECw25Fo3fza3Ew?usp=sharing) --->

<!--- # try the negative one
### All countries and years
>[Random forest classify w/ 100 estimators with negative](https://drive.google.com/drive/u/1/folders/1phB7BHdyNfNN0E5x-4CTvkUw-CytBimA)
### All years
>[Random forest classify w/ 100 estimators with negative](https://drive.google.com/drive/folders/1vjVjpDtPU8BEdJPUBRAFgv1DeNNan6_U?usp=sharing)
### All countries
>[Random forest classify w/ 100 estimators with negative](https://drive.google.com/open?id=16SVbkRaDNkB7UcYyJ1x920OHp_2xaSEV)
--->
<!--- 
***
#### Trained Model with 30 trees
>[all_year_neg_30](https://drive.google.com/drive/folders/17QhXMFntKTdx8WR8ZkHed-f5JtDo_dmA?usp=sharing)
<!---
>[all_country_neg_30](https://drive.google.com/drive/folders/1g4DOpf14gy-ApoxJ3juAj5NKFp-iaNwG?usp=sharing)
---> 
***
#### Trained Model and Results with 30 trees after feature selection

- Data Sliced by Year as training and test sets, using top 50 important features in random forest regression to predict negative returns of exchange rate over time. 
> [all_year_neg_30 after feature selection](https://drive.google.com/open?id=1UyVOKKawiKdgQTMcjtchJ0O30TesJGL7)

> The folder contains: 
>  * model output by each years, ex: 2000
> * the folder model contains: 
>    * pickle files with file name suffix 0 -- only text, 1 -- only exchange rate, 2 -- mixed features
  

- Data Sliced by Country as training and test sets, using top 50 important features in random forest regression to predict negative returns of exchange rate over time. 
> [all_country_neg_30 after feature selection](https://drive.google.com/open?id=1RUnsvHBvvJcPUxUMFeLgNKryfIIAklCO)

> The folder contains: 
> * model output by each country, ex: sweden
> * the folder model contains: 
>   * pickle files with file name suffix 0 -- only text, 1 -- only exchange rate, 2 -- mixed features

- Full dataset as training and test sets, using top 50 important features in random forest regression to predict negative returns of exchange rate over time. 
> [Full dataset neg 30 after feature selection](https://drive.google.com/open?id=1uj-BHVLJrC3YrZ74DzAghyU3KfBIGzsv)

> The folder contains: 
>  * model output by full dataset, ex: countries
>  * the folder model contains: 
>    * pickle files with file name suffix 0 -- only text, 1 -- only exchange rate, 2 -- mixed features
