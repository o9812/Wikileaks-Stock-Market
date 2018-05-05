# Wikileaks-Stock-Market
- This project measure how confidential information affects the financial market. It is believed that market efficiency is based on all market participant have fair access to the market. More importantly, it includes any kind of information. However, people do possess confidential information in the real world. Most financial markets have prohibited any kind of inside trading. Detecting how confidential information affect market is hard, due to the difficulties to access confidential data. 
- Luckily, we now have [wikileaks PLUSD](https://wikileaks.org/plusd/about/) dataset. We scrape document from 2000 to 2010 to detect how WikiLeaks cables affect exchange rate by different countries. The historical exchange rate is from CRSP dataset and covers 21 countries.  

## Data set
We have different data set, one is WikiLeak cables and the other is exchange rate. We split data country by country and year by year.
  
- Here is country level data
[country by country](https://drive.google.com/drive/folders/1uHIfkPc2b-b_3XDnRJn3NO2baRxnrXz5?usp=sharing)
###### The countri list is: 'australia', 'brazil', 'canada', 'china', 'denmark', 'hong kong', 'india', 'japan', 'korea', 'malaysia', 'mexico', 'new zealand', 'norway', 'sweden', 'south africa', 'singapore', 'sri lanka', 'switzerland', 'taiwan', 'thailand', 'united kingdom', 'venezuela'
###### it also includes a joint table of all countries call `final_All_countries`
- Here is year level data
[year by year](https://drive.google.com/drive/folders/1DMejBtKP9QGcnsybepXAuWAlqLqSIahR?usp=sharing)


## Run the model
To run the Random Forest Regression Model, the python file is `RanFrst_regres_final.py`:
#### searching the file name was written in hard code. So, if you want to rename the data, you would need to modify the main function.
- Country by Country
```
python RanFrst_regres_final.py 10 ./data_country/ country_10 -country
```
Here, the pararmeters:
-- `10`: the number of estimators in random forest model
-- `./data_country/`: input data path, supposed data is stored under `./data_country/`
-- `country_10`: output data path, it would automaticall create a directory called `./output_country_10/`
-- `-country`: let the model know it is searching what kind of data (country level or year)


- Year by Year
```
python RanFrst_regres_final.py 10 ./data_year/ year_10 -year
```

