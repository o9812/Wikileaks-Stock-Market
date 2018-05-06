# Wikileaks-Stock-Market
- This project measure how confidential information affects the financial market. It is believed that market efficiency is based on all market participant have fair access to the market. More importantly, it includes any kind of information. However, people do possess confidential information in the real world. Most financial markets have prohibited any kind of inside trading. Detecting how confidential information affect market is hard, due to the difficulties to access confidential data. 
- Luckily, we now have [wikileaks PLUSD](https://wikileaks.org/plusd/about/) dataset. We scrape document from 2000 to 2010 to detect how WikiLeaks cables affect exchange rate by different countries. The historical exchange rate is from CRSP dataset and covers 21 countries.  

## Data set
We have different data set, one is WikiLeak cables and the other is exchange rate. We split data country by country and year by year.
  
- Here is country level data
[country by country](https://drive.google.com/drive/folders/1uHIfkPc2b-b_3XDnRJn3NO2baRxnrXz5?usp=sharing)
#### The countri list is: 'australia', 'brazil', 'canada', 'china', 'denmark', 'hong kong', 'india', 'japan', 'korea', 'malaysia', 'mexico', 'new zealand', 'norway', 'sweden', 'south africa', 'singapore', 'sri lanka', 'switzerland', 'taiwan', 'thailand', 'united kingdom', 'venezuela'
#### it also includes a joint table of all countries call `final_All_countries`
- Here is year level data
[year by year](https://drive.google.com/drive/folders/1DMejBtKP9QGcnsybepXAuWAlqLqSIahR?usp=sharing)


## Run the regression and classfication model
To run the Random Forest Regression Model, firstly change working directory to the directory of `RanFrst_regres_final.py`. Then run the python file `RanFrst_regres_final.py`:
#### searching the file name was written in hard code. So, if you want to rename the data, you would need to modify the main function.
- Country by Country
### run regression
```
python RanFrst_regres.py 10 ./data_country/ country_10 -country
```
### run calssify
```
python RanFrst_classfy.py 10 ./data_country/ country_10 -country
```
Here, the pararmeters:
> - `10`: the number of estimators in random forest model
> - `./data_country/`: input data path, supposed data is stored under `./data_country/`
> - `country_10`: output data path, it would automaticall create a directory called `./output_country_10/`
> - `-country`: let the model know it is searching what kind of data (country level or year)

- Year by Year
### run regression
```
python RanFrst_regres.py 10 ./data_year/ year_10 -year
```
### run calssify
```
python RanFrst_classfy.py 10 ./data_year/ year_10 -year
```
> - `10`: the number of estimators in random forest model
> - `./data_year/`: input data path, supposed data is stored under `./data_year/`
> - `year_10`: output data path, it would automaticall create a directory called `./output_year_10/`
> - `-year`: let the model know it is searching what kind of data (country level or year)

## Output - regression
`RanFrst_regres.py` would automatically generate output under the user defined output directory, ex `./output_country_10/` or `./output_uear_10/`. 
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

`RanFrst_classfy.py` would automatically generate output under the user defined output directory, ex `./output_country_10/` or `./output_uear_10/`. What's more, it would automatically create a folder and store three generated AUC figures.
- Country by Country
It would generate 22 files named by the country name, ex: the input `final_Single_mexico` file would generate file `mexico` with 
| Year                     | 2003       | 
| -------------                 |-------------| 
| mse_text is                   |[ 0.          0.06140351  1.        ]|
| mae_text is                   | 6.79e-06  | 
| median_absolute_error stripes |  0.00162758    |
| r2_text                       | -0.03564915     |
| -------------                 |-------------| 
and `./output_country_10/mexico_figure/`
