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

# In[144]:


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

# In[145]:


# API KEY obtained from https://quickstats.nass.usda.gov/api/
API_KEY = os.getenv('API_KEY')


# In[146]:


# URL='https://quickstats.nass.usda.gov/results/5707E545-6B9E-35A4-AF77-DAF0BA7D7A7B'


# In[147]:


# API documentation: https://quickstats.nass.usda.gov/api
# Example URL = 'https://quickstats.nass.usda.gov/api/api_GET/?key=API_KEY&commodity_desc=CORN&year__GE=2010&state_alpha=VA'

url = 'https://quickstats.nass.usda.gov/api/api_GET/'
params = {
    "key":API_KEY,
    "commodity_desc":"CORN",
    "year__GE":"2013",
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

# In[148]:


import io

nass_data = nass_data.replace("'", '"')
nass_df = pd.read_json(io.StringIO(nass_data), orient='records')
nass_df.head()


# In[149]:


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


# In[150]:


"""
Cell generated by Data Wrangler.
"""
def clean_data(result_df):
    # Drop columns: 'congr_district_code', 'region_desc', 'watershed_desc', 'week_ending', 'zip_5'
    result_df = result_df.drop(columns=['congr_district_code', 'region_desc', 'watershed_desc', 'week_ending', 'zip_5'])
    return result_df

nass_combined_df_clean = clean_data(nass_combined_df.copy())
nass_combined_df_clean.head()


# In[151]:


# Remove commas from the 'Value' column
nass_combined_df_clean['Value'] = nass_combined_df_clean['Value'].str.replace(',', '')
# convert the 'value' column to float dtype
nass_combined_df_clean = nass_combined_df_clean.astype({'Value': 'float64'})
nass_combined_df_clean.info()


# In[152]:


# Split dataframe into four dataframes: one for each statistical category
area_planted = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'AREA PLANTED']
area_harvested = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'AREA HARVESTED']
yield_per_acre = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'YIELD']
production = nass_combined_df_clean[nass_combined_df_clean['statisticcat_desc'] == 'PRODUCTION']


# In[153]:


nass_pivoted_df = nass_combined_df_clean.pivot(index='year', columns='statisticcat_desc', values='Value')
nass_pivoted_df


# In[154]:


print(area_planted.shape)
print(area_harvested.shape)
print(yield_per_acre.shape)
print(production.shape)
print(nass_combined_df_clean.shape)


# In[155]:


nass_pivoted_df.info()


# ### Data Visualization

# In[156]:


# Correlation matrix of nass_pivoted_df
corr_matrix = nass_pivoted_df.corr()
corr_matrix


# In[157]:


# Heatmap of correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# # Data Set: Local Climatological Data (LCD)

# In[158]:


# Read the CSV file into a dataframe
lcd_df = pd.read_csv('data\LCD_Columbus_Bakalar_Municipal_Airport_Indiana_US.csv', low_memory=False)


# In[159]:


# Describe the dataframe
lcd_df.describe()


# In[160]:


# Display the first 5 rows of the dataframe
lcd_df.head()


# In[161]:


# Display a sample of 10 rows
lcd_df.sample(10)


# In[162]:


# Display the info of the dataframe
lcd_df.info()


# In[163]:


# Remove columns that are missing more than 50% of the data
lcd_df = lcd_df.dropna(thresh=0.5*len(lcd_df), axis=1)
lcd_df.info()


# In[164]:


lcd_df.sample(10)


# # Data Set: USA Facts Dataset

# ### Data Source

# In[165]:


# Read the CSV file into a dataframe
precip_df = pd.read_csv(r'data\USAFacts_Ripley_County_Indiana_Precipitation.csv', low_memory=False, quotechar='"', encoding='utf-8')


# ### Data Wrangling/Cleaning

# In[166]:


# Remove the 0 from the column names and trim the column names
precip_df.columns = precip_df.columns.str.replace('0', '').str.strip()
precip_df.columns


# In[167]:


# Create a new column 'year' from the 'time' column
precip_df['YEAR'] = precip_df['TIME'].str[:4].astype('int64')
precip_df.head()


# In[168]:


# Rename the 'DATA' column to 'PRECIPITATION'
precip_df.rename(columns={'DATA':'PRECIPITATION'}, inplace=True)
precip_df.head()


# In[169]:


# Calculate the mean precipitation for each year using the 'PRECIPITATION' and 'YEAR' columns
precip_df_mean = precip_df[['PRECIPITATION', 'YEAR']].groupby('YEAR').mean()
precip_df_mean.head()


# In[170]:


# Filter the dataframe to include only the years 2013-2023
precip_df_mean = precip_df_mean.loc[2013:2023]
precip_df_mean.head()


# ### Data Visualization

# In[171]:


# Plot the mean precipitation for each year
plt.figure(figsize=(12, 8))
plt.plot(precip_df_mean.index, precip_df_mean['PRECIPITATION'])
plt.xlabel('Year')
plt.ylabel('Precipitation (inches)')
plt.title('Mean Precipitation for Ripley County, Indiana')
plt.show()


# # Data Set: IndyStar, Ripley County, Indiana Aggregated Weather Data

# ### Data Source

# In[172]:


indy_star_summary_url = 'https://data.indystar.com/weather-data/ripley-county/18137/2023-07-01/?syear=1895&eyear=2023#summary'
indy_star_table_url = 'https://data.indystar.com/weather-data/ripley-county/18137/2023-07-01/table/'


# In[173]:


page = requests.get(indy_star_table_url)
soup = BeautifulSoup(page.content, 'html.parser')
#print(soup.prettify())


# In[174]:


import lxml.html as lh

tables = soup.find_all('table')

# Read the table into a dataframe
indy_star_df = pd.read_html(io.StringIO(str(tables)))[0]
indy_star_df.head()



# ### Data Wrangling/Cleaning

# In[175]:


# Calculate the 'Year' column from the 'Month' column
indy_star_df['Year'] = indy_star_df['Month'].str[-4:].astype('int64')
indy_star_df.head()


# In[176]:


# Calculate the mean precipitation for each year using the 'Precipitation' and 'Year' columns
indy_star_df_mean = indy_star_df[['Precipitation', 'Year']].groupby('Year').mean()
# Filter the dataframe to include only the years 2013-2023
indy_star_df_mean = indy_star_df_mean.loc[2013:2023]
indy_star_df_mean.head()


# ### Data Visualization

# In[177]:


# Compare the mean precipitation for each year from the two dataframes
plt.figure(figsize=(12, 8))
plt.plot(precip_df_mean.index, precip_df_mean['PRECIPITATION'], label='USAFacts')
plt.plot(indy_star_df_mean.index, indy_star_df_mean['Precipitation'], label='Indy Star')
plt.xlabel('Year')
plt.ylabel('Precipitation (inches)')
plt.title('Mean Precipitation for Ripley County, Indiana')
plt.legend()
plt.show()


# In[178]:


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

# In[179]:


# Combine the nass_pivoted_df and precip_df dataframes
combined_df = pd.concat([nass_pivoted_df, precip_df_mean], axis=1)
combined_df.head()


# In[180]:


# Create a 2023 precipitation variable
precip_2023 = combined_df.loc[2023, 'PRECIPITATION']
precip_2023


# In[181]:


# Create a 2023 dataframe using the median values for each column
# then add the 2023 precipitation value to the dataframe
# Name the dataframe 'predict_2023_df'
predict_2023_df = combined_df.median()
predict_2023_df['PRECIPITATION'] = precip_2023

# Drop the yield column
predict_2023_df = predict_2023_df.drop('YIELD')
predict_2023_df


# In[182]:


# Calculate the correlation matrix
corr_matrix = combined_df.corr()
corr_matrix


# In[183]:


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

# # Checkpoint 3: Machine Learning (Regression/Classification)

# 1. Machine Learning Plan
# 
# * What type of machine learning model are you planning to use?
#     * I plan to use linear regression to predict the yield for 2023.
# * What are the challenges you have identified/are you anticipating in building your machine learning model?
#     * One challenge I have is the almost non-existent correlation between precipitation, yield, and area planted.  Since corn needs moisture/rainfall to help it grow, there must be a correlation that exists.  Another challenge is I do not have the data for 2023.  I can only assume that a prediction will be relevant and statistically signficant with the current data.
# * How are you planning to address these challenges?
#     * I'm unsure at this time.  I need to think about whether this dataset is viable for a machine learning algorthim and predictive model.  There are many factors that are not part of the dataset that factor into this outcome.

# 2. Machine Learning Implementation Process
# 
#     (Ask, Prepare, Process, Analyze, Evaluate, Share)
# 
# * This includes:
#     * EDA process that allows for identifying issues
#     * Splitting the dataset into training and test sets
#     * Data cleaning process using sci-kit learn pipelines
#         * Data imputation
#         * Data Scaling and Normalization
#         * Handling of Categorical Data
#     * Testing multiple algorithms and models
#     * Evaluating the different models and choosing one.

# 3. Prior Feedback and Updates
#     * What feedback did you receive from your peers and/or the teaching team?
#         * I want to thank Christopher for his feedback.  Though the feedback did not provide any constructive critism, I appreciate the encouragement.
#     * What changes have you made to your project based on this feedback?
#         * I did not make any changes based on others' feedback.  However, I am evaluating my project sources and may determine to modify the project based on the usage of machine learning.

# # Machine Learning Process

# ## Exploratory Data Analysis

# In[184]:


# Display a scatter plot of the 'PRECIPITATION' and 'YIELD' columns
plt.figure(figsize=(12, 8))
plt.scatter(combined_df['PRECIPITATION'], combined_df['YIELD'])
plt.xlabel('Precipitation (inches)')
plt.ylabel('Yield (bushels/acre)')
plt.title('Precipitation vs. Yield')
plt.show()


# With the limited amount of data points, approximately 10 years of aggregated data, you can vaguely see a relationship between the annual rainfall and the annual yield of corn in Ripley county, Indiana.

# In[185]:


# Correlation matrix of combined_df
corr_matrix = combined_df.corr()
corr_matrix


# In[186]:


# Display a pairplot of the combined_df dataframe
sns.pairplot(combined_df)
plt.show()


# With the combinded data of: area planted, area harvested, production, yield, and precipitation, no further data imputation or cleansing is required.  This scatter matrix shows the strong correlation between area harvested and area planted, but no correlation between other variables.  However, there is obvious correlation between area harvested and production and/or yield.  Precipitation also plays a factor in the amount of production and/or yield.

# ## Prepare

# In[187]:


# Split the combined_df dataframe for training and testing
from sklearn.model_selection import train_test_split

# Remove 2023 from the dataframe
combined_df = combined_df.drop(index=2023)

# Sort the dataframe by year
combined_df = combined_df.sort_index()

# Create the X and y dataframes
X = combined_df.drop(columns=['YIELD'])

# Replace NaN values with imputed values in Area Harvested, Area Planted, and Production
X['AREA HARVESTED'] = X['AREA HARVESTED'].fillna(X['AREA HARVESTED'].median())
X['AREA PLANTED'] = X['AREA PLANTED'].fillna(X['AREA PLANTED'].median())
X['PRODUCTION'] = X['PRODUCTION'].fillna(X['PRODUCTION'].median())

# Replace NaN values with imputed values for Yield
y = combined_df['YIELD'].fillna(combined_df['YIELD'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')


# In[188]:


print(X)


# ## Process

# In[189]:


# Process pipeline for numeric features
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Process pipeline for categorical features
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Create a preprocessor pipeline
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline for the model
from sklearn.ensemble import RandomForestRegressor

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Fit the model
model.fit(X_train, y_train)

# Predict the yield for the test data
y_pred = model.predict(X_test)

# Calculate the mean absolute error
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate the mean absolute percentage error
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'Mean Absolute Percentage Error: {mape}')

# Calculate the root mean squared error
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error: {rmse}')

# Calculate the coefficient of determination
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'Coefficient of Determination: {r2}')

# Display the feature importances
importances = model.named_steps['regressor'].feature_importances_
feature_names = np.concatenate([numeric_features, categorical_features])
#feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_features)
#feature_names = np.concatenate([numeric_features, feature_names])
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.bar(feature_names[sorted_indices], importances[sorted_indices])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


# ## Analyze

# In[190]:


# Create a Linear Regression model
from sklearn.linear_model import LinearRegression

# Fit the model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict the yield for the test data
y_pred = lin_reg.predict(X_test)

# Calculate the R^2 score for the training and testing data
print(f'Training R^2 score: {lin_reg.score(X_train, y_train)}')
print(f'Testing R^2 score: {lin_reg.score(X_test, y_test)}')

# Calculate the mean absolute error for the training and testing data
from sklearn.metrics import mean_absolute_error

y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

print(f'Training MAE: {mean_absolute_error(y_train, y_train_pred)}')
print(f'Testing MAE: {mean_absolute_error(y_test, y_test_pred)}')

# Calculate the mean squared error for the training and testing data
from sklearn.metrics import mean_squared_error

print(f'Training MSE: {mean_squared_error(y_train, y_train_pred)}')
print(f'Testing MSE: {mean_squared_error(y_test, y_test_pred)}')

# Calculate the root mean squared error for the training and testing data
from sklearn.metrics import mean_squared_error

print(f'Training RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}')
print(f'Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}')

# Calculate the mean absolute percentage error for the training and testing data
from sklearn.metrics import mean_absolute_percentage_error

print(f'Training MAPE: {mean_absolute_percentage_error(y_train, y_train_pred)}')
print(f'Testing MAPE: {mean_absolute_percentage_error(y_test, y_test_pred)}')

# Display the linear regression coefficients
print(f'Intercept: {lin_reg.intercept_}')
print(f'Coefficients: {lin_reg.coef_}')


# In[192]:


# Predict the yield for the 2023 data using the linear regression model
y_pred = lin_reg.predict(predict_2023_df.values.reshape(1, -1))
print(f'Predicted Yield for 2023: {y_pred}')


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

# In[193]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


# 
