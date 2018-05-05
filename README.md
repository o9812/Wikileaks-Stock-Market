# Wikileaks-Stock-Market
This project measure how confidential information affects the financial market. It is believed that market efficiency is based on all market participant have fair access to the market. More importantly, it includes any kind of information. However, people do possess confidential information in the real world. Most financial markets have prohibited any kind of inside trading. Detecting how confidential information affect market is hard, due to the difficulties to access confidential data. 
Luckily, we now have [wikileaks PLUSD](https://wikileaks.org/plusd/about/) dataset. We scrape document from 2000 to 2010 to detect how WikiLeaks cables affect exchange rate by different countries.

Sepcifically, we focus on country level. Thus, we measure how exchange rate would be affect by wikileaks.
- /data/: test data
- /script/: store all code and explanation, please read README.md in script folder


## Run the model
To run the Random Forest Regression Model, the python file is `RanFrst_regres_final.py`:
- Country by Country
```
python RanFrst_regres_final.py 10 ./data_country/ country_10 -country
```
- Year by Year
```
python RanFrst_regres_final.py 10 ./data_year/ year_10 -year
```

