#!/usr/bin/env python
# coding: utf-8

# # Climate change's effect on local food and water resources. 📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# This project will attempt to address if localized climate change has a detrimental effect on local crop production and water resources.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# 1. Is climate change happening in my local area of Southeast Indiana?
# 2. Is climate change affecting the yield production of local crops, e.g., corn and soybean?
# 3. Is climate change affecting the water resources, i.e., is there an abundance or scarcity of water based on the amount of yearly rainfall? 

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# I hypothesize that climate change is occuring in Southeast Indiana, and it has affected local crop yield and reduced a percentage of available water resources.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# 1. United States Department of Agriculture (USDA) - https://quickstats.nass.usda.gov
# 2. Local Climate Analysis Tool (LCAT) - https://lcat.nws.noaa.gov/home
# 3. National Centers for Environmental Information | Local Climatological Data (LCD) - https://www.ncei.noaa.gov/cdo-web/datatools/lcd
# 4. USA Facts - Ripley County, Indiana - https://usafacts.org/issues/climate/state/indiana/county/ripley-county/?endDate=2023-11-18&startDate=2013-02-01#climate/
# 
# I will relate these data sets by geographical locations, either at the state level or preferably at the county level. 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# My approach is to join the climate data set to the agricultural data set on the specified region, Southeast Indiana.  I need to fully understand the NOAA weather data's schema to determine if my county, Ripley, is included in the Wilmington Station, which it should be.
# 
# Once I have identified that Wilmington Weather Station does indeed cover my local area, I will attempt to extrapolate the annual rainfall and temperature data to determine if there is a correlation between weather and crop yield.

# # Import Packages/Libraries

# In[ ]:


# Start your code here

import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv

# Load the project environment variables
load_dotenv(override=True)

import requests
from urllib.request import urlretrieve, urlparse
from bs4 import BeautifulSoup

# Configure pandas to display 500 rows; otherwise it will truncate the output
pd.set_option('display.max_rows', 500)

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns



# # Data Set: USDA National Agricultural Statistics Service (NASS)

# ### Data Source

# "This product uses the NASS API but is not endorsed or certified by NASS."

# In[ ]:


# API KEY obtained from https://quickstats.nass.usda.gov/api/
API_KEY = os.getenv('API_KEY')


# In[ ]:


# URL='https://quickstats.nass.usda.gov/results/5707E545-6B9E-35A4-AF77-DAF0BA7D7A7B'


# In[ ]:


# API documentation: https://quickstats.nass.usda.gov/api
# Example URL = 'https://quickstats.nass.usda.gov/api/api_GET/?key=API_KEY&commodity_desc=CORN&year__GE=2010&state_alpha=VA'

url = 'https://quickstats.nass.usda.gov/api/api_GET/'
params = {
    "key":API_KEY,
    "commodity_desc":"CORN",
    "year__GE":"2010",
    "state_alpha":"IN",
    "county_name":"RIPLEY",
    "sector_desc":"CROPS",
    "source_desc":"SURVEY"
}

response = requests.get(url=url, params=params)
nass_data = str(response.json())


# ### Data Wrangling/Cleaning

# #### Data Dictionary

# |Column or Header Name|Max Length|Definition|
# | --- | --- | --- |
# |The "WHAT" (or Commodity) dimension|||
# |source_desc|60|Source of data (CENSUS or SURVEY). Census program includes the Census of Ag as well as follow up projects. Survey program includes national, state, and county surveys.|
# |(Program)|||
# |sector_desc|60|Five high level, broad categories useful to narrow down choices (CROPS, ANIMALS & PRODUCTS, ECONOMICS, DEMOGRAPHICS, and ENVIRONMENTAL).|
# |(Sector)|||
# |group_desc|80|Subsets within sector (e.g., under sector = CROPS, the groups are FIELD CROPS, FRUIT & TREE NUTS, HORTICULTURE, and VEGETABLES).|
# |(Group)|||
# |commodity_desc|80|The primary subject of interest (e.g., CORN, CATTLE, LABOR, TRACTORS, OPERATORS).|
# |(Commodity)|||
# |class_desc|180|Generally a physical attribute (e.g., variety, size, color, gender) of the commodity.|
# |prodn_practice_desc|180|A method of production or action taken on the commodity (e.g., IRRIGATED, ORGANIC, ON FEED).|
# |util_practice_desc|180|Utilizations (e.g., GRAIN, FROZEN, SLAUGHTER) or marketing channels (e.g., FRESH MARKET, PROCESSING, RETAIL).|
# |statisticcat_desc|80|The aspect of a commodity being measured (e.g., AREA HARVESTED, PRICE RECEIVED, INVENTORY, SALES).|
# |(Category)|||
# |unit_desc|60|The unit associated with the statistic category (e.g., ACRES, $ / LB, HEAD, $, OPERATIONS).|
# |short_desc|512|A concatenation of six columns: commodity_desc, class_desc, prodn_practice_desc, util_practice_desc, statisticcat_desc, and unit_desc.|
# |(Data Item)|||
# |domain_desc|256|Generally another characteristic of operations that produce a particular commodity (e.g., ECONOMIC CLASS, AREA OPERATED, NAICS CLASSIFICATION, SALES). For chemical usage data, the domain describes the type of chemical applied to the commodity. The domain = TOTAL will have no further breakouts; i.e., the data value pertains completely to the short_desc.|
# |(Domain)|||
# |domaincat_desc (Domain Category)|512|Categories or partitions within a domain (e.g., under domain = SALES, domain categories include $1,000 TO $9,999, $10,000 TO $19,999, etc).|
# |The "WHERE" (or Location) dimension|||
# |agg_level_desc|40|Aggregation level or geographic granularity of the data (e.g., STATE, AG DISTRICT, COUNTY, REGION, ZIP CODE).|
# |(Geographic Level)|||
# |state_ansi|2|American National Standards Institute (ANSI) standard 2-digit state codes.|
# |state_fips_code|2|NASS 2-digit state codes; include 99 and 98 for US TOTAL and OTHER STATES, respectively; otherwise match ANSI codes.|
# |state_alpha|2|State abbreviation, 2-character alpha code.|
# |state_name|30|State full name.|
# |(State)|||
# |asd_code|2|NASS defined county groups, unique within a state, 2-digit ag statistics district code.|
# |asd_desc|60|Ag statistics district name.|
# |(Ag District)|||
# |county_ansi|3|ANSI standard 3-digit county codes.|
# |county_code|3|NASS 3-digit county codes; includes 998 for OTHER (COMBINED) COUNTIES and Alaska county codes; otherwise match ANSI codes.|
# |county_name|30|County name.|
# |(County)|||
# |region_desc|80|NASS defined geographic entities not readily defined by other standard geographic levels. A region can be a less than a state (SUB-STATE) or a group of states (MULTI-STATE), and may be specific to a commodity.|
# |(Region)|||
# |zip_5|5|US Postal Service 5-digit zip code.|
# |(Zip Code)|||
# |watershed_code|8|US Geological Survey (USGS) 8-digit Hydrologic Unit Code (HUC) for watersheds.|
# |watershed_desc|120|Name assigned to the HUC.|
# |(Watershed)|||
# |congr_district_code|2|US Congressional District 2-digit code.|
# |country_code|4|US Census Bureau, Foreign Trade Division 4-digit country code, as of April, 2007.|
# |country_name|60|Country name.|
# |location_desc|120|Full description for the location dimension.|
# |The "WHEN" (or Time) dimension|||
# |year|4|The numeric year of the data.|
# |(Year)|||
# |freq_desc|30|Length of time covered (ANNUAL, SEASON, MONTHLY, WEEKLY, POINT IN TIME). MONTHLY often covers more than one month. POINT IN TIME is as of a particular day.|
# |(Period Type)|||
# |begin_code|2|If applicable, a 2-digit code corresponding to the beginning of the reference period (e.g., for freq_desc = MONTHLY, begin_code ranges from 01 (January) to 12 (December)).|
# |end_code|2|If applicable, a 2-digit code corresponding to the end of the reference period (e.g., the reference period of JAN THRU MAR will have begin_code = 01 and end_code = 03).|
# |reference_period_|40|The specific time frame, within a freq_desc.|
# |desc (Period)|||
# |week_ending|10|Week ending date, used when freq_desc = WEEKLY.|
# |load_time|19|Date and time indicating when record was inserted into Quick Stats database.|
# |The Data Value|||
# |value|24|Published data value or suppression reason code.|
# |CV %|7|Coefficient of variation. Available for the 2012 Census of Agriculture only. County-level CVs are generalized.|
# 

# #### Data Loading

# In[ ]:


import io

nass_data = nass_data.replace("'", '"')
nass_df = pd.read_json(io.StringIO(nass_data), orient='records')
nass_df.head()


# In[ ]:


# Reference: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
# Reference: https://stackoverflow.com/questions/20638006/convert-list-of-dictionaries-to-a-pandas-dataframe
# Reference: Bing Chat with GPT-4
# Reference: Github Copilot
# Iterate through df and convert JSON to dataframe
import json

dataframes = []
for index, row in nass_df.iterrows():
    json_data = row.to_json()
    json_clean = json_data[8:-1]
    my_dict = json.loads(json_clean)
    sorted_dict = dict(sorted(my_dict.items()))
    dataframe = pd.DataFrame.from_dict(sorted_dict, orient='index').T
    dataframes.append(dataframe)

# Concatenate all dataframes into a single dataframe
nass_combined_df = pd.concat(dataframes, ignore_index=True)
nass_combined_df.sample(10)


# In[ ]:


"""
Cell generated by Data Wrangler.
"""
def clean_data(result_df):
    # Drop columns: 'congr_district_code', 'region_desc', 'watershed_desc', 'week_ending', 'zip_5'
    result_df = result_df.drop(columns=['congr_district_code', 'region_desc', 'watershed_desc', 'week_ending', 'zip_5'])
    return result_df

nass_combined_df_clean = clean_data(nass_combined_df.copy())
nass_combined_df_clean.head()


# In[ ]:


# Remove commas from the 'Value' column
nass_combined_df_clean['Value'] = nass_combined_df_clean['Value'].str.replace(',', '')
# convert the 'value' column to float dtype
nass_combined_df_clean = nass_combined_df_clean.astype({'Value': 'float64'})
nass_combined_df_clean.info()


# In[ ]:


# Split dataframe into four dataframes: one for each statistical category
area_planted = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'AREA PLANTED']
area_harvested = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'AREA HARVESTED']
yield_per_acre = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'YIELD']
production = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'PRODUCTION']


# In[ ]:


nass_pivoted_df = nass_combined_df_clean.pivot(index='year', columns='statisticcat_desc', values='Value')
nass_pivoted_df


# In[ ]:


print(area_planted.shape)
print(area_harvested.shape)
print(yield_per_acre.shape)
print(production.shape)
print(nass_combined_df_clean.shape)


# In[ ]:


nass_pivoted_df.info()


# ### Data Visualization

# In[ ]:


# Correlation matrix of nass_pivoted_df
corr_matrix = nass_pivoted_df.corr()
corr_matrix


# In[ ]:


# Heatmap of correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# # Data Set: Local Climatological Data (LCD)

# In[ ]:


# Read the CSV file into a dataframe
lcd_df = pd.read_csv('data\LCD_Columbus_Bakalar_Municipal_Airport_Indiana_US.csv', low_memory=False)


# In[ ]:


# Describe the dataframe
lcd_df.describe()


# In[ ]:


# Display the first 5 rows of the dataframe
lcd_df.head()


# In[ ]:


# Display a sample of 10 rows
lcd_df.sample(10)


# In[ ]:


# Display the info of the dataframe
lcd_df.info()


# In[ ]:


# Remove columns that are missing more than 50% of the data
lcd_df = lcd_df.dropna(thresh=0.5*len(lcd_df), axis=1)
lcd_df.info()


# In[ ]:


lcd_df.sample(10)


# # Data Set: USA Facts Dataset

# ### Data Source

# In[ ]:


# Read the CSV file into a dataframe
precip_df = pd.read_csv(r'data\USAFacts_Ripley_County_Indiana_Precipitation.csv', low_memory=False, quotechar='"', encoding='utf-8')


# ### Data Wrangling/Cleaning

# In[ ]:


# Remove the 0 from the column names and trim the column names
precip_df.columns = precip_df.columns.str.replace('0', '').str.strip()
precip_df.columns


# In[ ]:


# Create a new column 'year' from the 'time' column
precip_df['YEAR'] = precip_df['TIME'].str[:4].astype('int64')
precip_df.head()


# In[ ]:


# Rename the 'DATA' column to 'PRECIPITATION'
precip_df.rename(columns={'DATA':'PRECIPITATION'}, inplace=True)
precip_df.head()


# In[ ]:


# Calculate the mean precipitation for each year using the 'PRECIPITATION' and 'YEAR' columns
precip_df_mean = precip_df[['PRECIPITATION', 'YEAR']].groupby('YEAR').mean()
precip_df_mean.head()


# In[ ]:


# Filter the dataframe to include only the years 2013-2023
precip_df_mean = precip_df_mean.loc[2013:2023]
precip_df_mean.head()


# ### Data Visualization

# In[ ]:


# Plot the mean precipitation for each year
plt.figure(figsize=(12, 8))
plt.plot(precip_df_mean.index, precip_df_mean['PRECIPITATION'])
plt.xlabel('Year')
plt.ylabel('Precipitation (inches)')
plt.title('Mean Precipitation for Ripley County, Indiana')
plt.show()


# # Data Set: IndyStar, Ripley County, Indiana Aggregated Weather Data

# ### Data Source

# In[ ]:


indy_star_summary_url = 'https://data.indystar.com/weather-data/ripley-county/18137/2023-07-01/?syear=1895&eyear=2023#summary'
indy_star_table_url = 'https://data.indystar.com/weather-data/ripley-county/18137/2023-07-01/table/'


# In[ ]:


page = requests.get(indy_star_table_url)
soup = BeautifulSoup(page.content, 'html.parser')
#print(soup.prettify())


# In[ ]:


import lxml.html as lh

tables = soup.find_all('table')

# Read the table into a dataframe
indy_star_df = pd.read_html(io.StringIO(str(tables)))[0]
indy_star_df.head()



# ### Data Wrangling/Cleaning

# In[ ]:


# Calculate the 'Year' column from the 'Month' column
indy_star_df['Year'] = indy_star_df['Month'].str[-4:].astype('int64')
indy_star_df.head()


# In[ ]:


# Calculate the mean precipitation for each year using the 'Precipitation' and 'Year' columns
indy_star_df_mean = indy_star_df[['Precipitation', 'Year']].groupby('Year').mean()
# Filter the dataframe to include only the years 2013-2023
indy_star_df_mean = indy_star_df_mean.loc[2013:2023]
indy_star_df_mean.head()


# ### Data Visualization

# In[ ]:


# Compare the mean precipitation for each year from the two dataframes
plt.figure(figsize=(12, 8))
plt.plot(precip_df_mean.index, precip_df_mean['PRECIPITATION'], label='USAFacts')
plt.plot(indy_star_df_mean.index, indy_star_df_mean['Precipitation'], label='Indy Star')
plt.xlabel('Year')
plt.ylabel('Precipitation (inches)')
plt.title('Mean Precipitation for Ripley County, Indiana')
plt.legend()
plt.show()


# In[ ]:


# Display the mean preciptation for each year from the two dataframes side-by-side
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(precip_df_mean.index, precip_df_mean['PRECIPITATION'])
plt.xlabel('Year')
plt.ylabel('Precipitation (inches)')
plt.title('USAFacts')
plt.subplot(1, 2, 2)
plt.plot(indy_star_df_mean.index, indy_star_df_mean['Precipitation'])
plt.xlabel('Year')
plt.ylabel('Precipitation (inches)')
plt.title('Indy Star')
plt.show()


# # Combining the Data Sets

# In[ ]:


# Combine the nass_pivoted_df and precip_df dataframes
combined_df = pd.concat([nass_pivoted_df, precip_df_mean], axis=1)
combined_df.head()


# In[ ]:


# Calculate the correlation matrix
corr_matrix = combined_df.corr()
corr_matrix


# In[ ]:


# Heatmap of correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# # Check Point #2 Questions

# #### Exploratory Data Analysis (EDA)

# What insights and interesting information are you able to extract at this stage?
# - I did not find the correlation between crop yield and annual precipation that I had expected.  I might filter the average precipitation to a "growing" season, e.g., May - October.
# 
# What are the distributions of my variables?
# - I am working with a limited amount of data, aggregated to an annual basis.  The distribution of my variables is difficult to know.
# 
# Are there any correlations between my variables?
# - There is a high correlation between 'ACRE PLANTED' and 'ACRE HARVESTED', but the remaining variables show low correlation.
# 
# What issues can you see in your data at this point?
# - The data doesn't seem to support my hypothesis.
# 
# Are there any outliers or anomalies? are they relevant to your analysis? or should they be removed?
# - Not really, but I have data from 2010 to 2023.  I will remove years 2010, 2011, and 2012 to keep a 10-year range.
# 
# Are there any missing values? how are you going to deal with them?
# - Yes, the Local Climatological Data (LCD) was missing a lot of columns.  After review the remaining data set, I determined that it was unuseful and searched for other data sets.
# 
# Are there any duplicate values? how are you going to deal with them?
# - I did not find any duplication within these data sets.  If I had, I would have deduped that data.
# 
# Are there any data types that need to be changed?
# - Yes, I changed some data types from object or string to Int64 or Float.

# #### Data Cleaning and Transformation

# Each section has a Data Wrangling/Cleaning section.
# 
# Overall, here is my explanation of my process.
# * Missing values
#     * I did not have missing values except for the LCD dataset, which I decided to not use.
# * Duplicate values
#     * I did not have duplicate values.  If I had, I would have deduped the data.
# * Anomalies and Outliers
#     * Technically, I did not see any anomalies or outliers, but I will limit my data to a 10-year range, 2013-2023.
# * Data type transformations
#     * I converted some Object and String variables to Int64 or Float dtypes.

# #### Prior Feedback and Updates

# * Have you received any feedback?
#     * I have not received feedback from others.
# * What changes have you made to your project based on this feedback?
#     * Not applicable at this time.

# # Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# * Data source references listed above
# * Bing Chat with GPT-4
# * https://www.ncei.noaa.gov/data/local-climatological-data/doc/LCD_documentation.pdf
# * https://www.ncdc.noaa.gov/cdo-web/datasets
# * https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
# * https://stackoverflow.com/questions/20638006/convert-list-of-dictionaries-to-a-pandas-dataframe
# * Github Copilot
# * https://stackoverflow.com/questions/14745022/how-to-split-a-column-into-two-columns

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


# 
