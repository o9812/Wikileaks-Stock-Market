# Data and Result Scription
THIS IS THE LIST OF FILE ON THE AZURE SERVER.
## Raw data of wikileaks cable
- `/data/wikileaks/Crawled_Data/Later_than_2000`		: wikileaks data before 2000s
- `/data/wikileaks/Crawled_Data/data matching country`	: wikileaks match country with dummy variable 
- `/data/wikileaks/Crawled_Data/data_crawled`		: wikileaks data after 2000s

    - Json files cluding each cable. Each cable has unique keys, and Includes keys: 
    - Content: content of wikileaks cable
    - Raw content: raw content of wikileaks cable
    - Metadata: metadata of wikileaks cable
    - Date: date of wikileaks cable
    - Canonical ID: id of wikileaks cable in PlusD
    - Original Classification: wikileaks cable origin classification
    - Current Classification: wikileaks cable modified classification by PlusD
    - Handling Restrictions: Not used (the intermediate data from scraping) 
    - Character Count: the count number of words
    - Executive Order: Not used (the intermediate data from scraping) 
    - Locator: Not used (the intermediate data from scraping)
    - TAGS: TAGS in the wikileaks dataset
    - Concepts: Not used (the intermediate data from scraping)
    - Enclosure: Not used (the intermediate data from scraping)
    - Type: Type of the cable
    - Office Origin: Who owns the cable
    - Office Action: Who executes the cable
    - Archive Status: The status of the cable in dataset
    - From: From whom/ which country
    - Markings
    - To: To whom/ which country
    - Linked documents or other documents with the same ID
    - Country name: if true 1, not true zero

    ***
    - Tags example:
    - ECON - Economic Affairs--Economic Conditions, Trends and Potential
    - KMDR - Media Reaction Reporting
    - KPAO - Public Affairs Office
    - KS - Korea (South)
    - MARR - Military and Defense Affairs--Military and Defense Arrangements
    - PGOV - Political Affairs--Government; Internal Governmental Affairs
    - PREL - Political Affairs--External Political Relations
    - US - United States


## Raw data of crsp data (on server)
- `/data/wikileaks/crsp_data/CRSP_data/Company_List`: 
   -  include all companies listed on NYSE, AMEX and NASDAQ as of Mar 31 2018.
   - From the three exchange AMEX, NYSE and NADAQ
- `/data/wikileaks/crsp_data/CRSP_data/all_stock_abnormal_return`:
    - abnormal return: This file includes abnormal returns of all stock list on NYSE, AMEX, NASDAQ.
    - Includes attributes of: PERMNO, return, excess return
    - Excess return: the difference of actual return and expected return.
    This is exactly of the abnormal return. 


- `/data/wikileaks/crsp_data/CRSP_data/allstock_list_crsp`: 
    - all stocks listed on NYSE, AMEX, NASDAQ. From 1980/01/01 to 2010/10/31
    - attributes: price, company name, ticker, premant code

- `/data/wikileaks/crsp_data/CRSP_data/exchange_rate`: 
    - exchange rate from 1971/1/1 to 2010/12/31
    - note: from 2002, the euro appears
    Australia, Brazil, Canada, China, Denmark, Hong Kong, India, Japan, Korea, Malaysia, Mexico, New Zealand, Norway, Sweden, South Africa, Singapore, Sri Lanka, Switzerland, Taiwan, Thailand, United Kingdom, Venezuela, Austria, Belgium, Finland, France, Germany, Greece, Ireland, Italy, Netherlands, Portugal, Spain, Euro, ECU
|
- `/data/wikileaks/crsp_data/CRSP_data/data_country`:
    > `final_All_countries` contain all year and all countris
    - Single country. A data frame with 5 column.
    - date: date of the wikileaks
    - lg_rt_features: 15 days exchange rate before the date
    - num_lable: numerical label, the exchange rate 
    - dummy_lable: the dummy label, 1 if it is an abnormal return 0 if not.
    - Content: the raw content of each cable.
- `/data/wikileaks/crsp_data/CRSP_data/data_year`:
    - Single year. A data frame with 5 column.
    - date: date of the wikileaks
    - lg_rt_features: 15 days exchange rate before the date
    - num_lable: numerical label, the exchange rate 
    - dummy_lable: the dummy label, 1 if it is an abnormal return 0 if not.
    - Content: the raw content of each cable.

- `/data/wikileaks/rsp_data/CRSP_data/data_country_ng`
    > Only contain instance which has negative numerical label. 
    > `final_All_countries_ng` contain all year and all countris
    - Single country. A data frame with 5 column.
    - date: date of the wikileaks
    - lg_rt_features: 15 days exchange rate before the date
    - num_lable: numerical label, the exchange rate 
    - dummy_lable: the dummy label, 1 if it is an abnormal return 0 if not.
    - Content: the raw content of each cable.
    
   
- `/data/wikileaks/crsp_data/CRSP_data/data_year_ng`
    > Only contain instance which has negative numerical label
    - Single year. A data frame with 5 column.
    - date: date of the wikileaks
    - lg_rt_features: 15 days exchange rate before the date
    - num_lable: numerical label, the exchange rate 
    - dummy_lable: the dummy label, 1 if it is an abnormal return 0 if not.
    - Content: the raw content of each cable.

    ***
    ## Resutl
- `/data/wikileaks/data/WorkData/wikileaks/output_result`
    > The result on the Azure server.
    - output_all_neg_30_a       :including all data, with only negative numerical label and only keep top 50 important features. 30 trees.
    - output_year_neg_30_a      :including all data split by year, with only negative numerical label and only keep top 50 important features. 30 trees. 
    - output_country_neg_30_a   :including all data split by country, with only negative numerical label and only keep top 50 important features. 30 trees. 
    ***
    ## models
 -`/data/WorkData/wikileaks/Wikileaks-Stock-Market`
    > store the file folder same as github
