#!/usr/bin/env python
# coding: utf-8

# # Global Country Information Dataset 2023

# ## 1. Load data
#
# **Source:** https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023

# In[1]:


import pandas as pd


# In[2]:


world = pd.read_csv("data/world-data-2023.csv") # Read CSV files


# In[3]:


pd.set_option('display.max_columns', 1000) # Display 1000 columns
pd.set_option('display.max_rows', 1000) # Display 1000 columns


# In[4]:


pd.concat([world.head(), world.tail()])


# In[5]:


world.info()


# ## 2. Data treatment
#
# ### 2.1 Select and rename the necessary attributes
#
# - <u>Group 1 - General</u>
#     - **country** <= `Country` - Name of the country.
#     - **area** <= `Land Area(Km2)` - Total land area of the country in square kilometers.
#     - **population** <= `Population` - Total population of the country.
#     - **density** <= `Density \n(P/Km2)`- Population density measured in persons per square kilometer.
#     - **forest** <= `Forested Area (%)` - Percentage of land area covered by forests.
#     - **official_lang** <= `Official language` - Official language(s) spoken in the country.
# - <u>Group 2 - People</u>
#     - **forces_size** <= `Armed Forces size` - Size of the armed forces in the country.
#     - **birth** <= `Birth Rate` - Number of births per 1,000 population per year.
#     - **unemployment** <= `Unemployment rate` - Percentage of the labor force that is unemployed.
#     - **urban** <= `Urban_population` - Percentage of the population living in urban areas.
#     - **inf_mortality** <= `Infant mortality` - Number of deaths per 1,000 live births before reaching one year of age.
#     - **life_exp** <= `Life expectancy` - Average number of years a newborn is expected to live.
#     - **physicians** <= `Physicians per thousand` - Number of physicians per thousand people.
# - <u>Group 3 - Economy</u>
#     - **cpi** <= `CPI` - Consumer Price Index, a measure of inflation and purchasing power.
#     - **cpi_change** <= `CPI Change (%)` - Percentage change in the Consumer Price Index compared to the previous year.
#     - **gdp** <= `GDP` - Gross Domestic Product, the total value of goods and services produced in the country.
#     - **min_wage** <= `Minimum wage` - Minimum wage level in local currency.
#     - **tax_revenue** <= `Tax revenue (%)` - Tax revenue as a percentage of GDP.
#     - **total_tax** <= `Total tax rate` - Overall tax burden as a percentage of commercial profits.
# - <u>Group 4 - Geography</u>
#     - **capital** <= `Capital/Major City` - Name of the capital or major city.
#     - **largest_city** <= `Largest city` - Name of the country's largest city.
#     - **latitude** <= `Latitude` - Latitude coordinate of the country's location.
#     - **longitude** <= `Longitude` - Longitude coordinate of the country's location.

# In[6]:


rename_dict = {
              'Country': 'country',
              'Land Area(Km2)': 'area',
              'Population': 'population',
              'Density\n(P/Km2)': 'density',
              'Forested Area (%)': 'forest',
              'Official language': 'official_lang',
              'Armed Forces size': 'forces_size',
              'Birth Rate': 'birth',
              'Unemployment rate': 'unemployment',
              'Urban_population': 'urban',
              'Infant mortality': 'inf_mortality',
              'Life expectancy': 'life_exp',
              'Physicians per thousand': 'physicians',
              'CPI': 'cpi',
              'CPI Change (%)': 'cpi_change',
              'GDP': 'gdp',
              'Minimum wage': 'min_wage',
              'Tax revenue (%)': 'tax_revenue',
              'Total tax rate': 'total_tax',
              'Capital/Major City': 'capital',
              'Largest city': 'largest_city',
              'Latitude': 'latitude',
              'Longitude': 'longitude'
              }
world.rename(columns=rename_dict, inplace=True)
world = world.loc[:, rename_dict.values()]
world.index = world.country
world.head().T


# ### 2.2 Convert data types

# In[7]:


import warnings

warnings.filterwarnings('ignore')

world["area"]          = pd.to_numeric(world["area"].str.replace(",", ""))/1000000
world["population"]    = pd.to_numeric(world["population"].str.replace(",", ""))/1000000
world["density"]       = pd.to_numeric(world["density"].str.replace(",", ""))
world["forest"]        = pd.to_numeric(world["forest"].str.replace("%", ""))/100
world["forces_size"]   = pd.to_numeric(world["forces_size"].str.replace(",", ""))/1000
world["unemployment"]  = pd.to_numeric(world["unemployment"].str.replace("%", ""))/100
world["urban"]         = pd.to_numeric(world["urban"].str.replace(",", ""))
world["cpi"]           = pd.to_numeric(world["cpi"].str.replace(",", ""))
world["cpi_change"]    = pd.to_numeric(world["cpi_change"].str.replace("%", ""))
world["gdp"]           = pd.to_numeric(world["gdp"].str.replace(",", "").str.replace("$", ""))
world["min_wage"]      = pd.to_numeric(world["min_wage"].str.replace(",", "").str.replace("$", ""))
world["tax_revenue"]   = pd.to_numeric(world["tax_revenue"].str.replace("%", ""))/100
world["total_tax"]     = pd.to_numeric(world["total_tax"].str.replace("%", ""))/100
world["latitude"]      = pd.to_numeric(world["latitude"])
world["longitude"]     = pd.to_numeric(world["longitude"])


# ## 3. Overlook
#
# ### 3.1 Filtering by index
#
# `(Select * from world where country in ("Russia", "Belarus")`
#
# `(Select area, capital from world where country in ("Canada", "United States")`

# In[8]:


world.loc[["Russia", "Belarus"]]


# In[9]:


world.loc[["Canada", "United States"], ["area", "capital"]]


# ### 3.2 Select some columns and sorting
#
# `
# select area, population, density, forest
# from world
# where forest >= 0.5
# order by area desc
# `

# In[10]:


world[["area", "population", "density", "forest"]].sort_values("area", ascending=False).head(6)


# ### 3.3 Complex filtering and select some columns

# In[11]:


world[world["area"] > 3][["area", "population"]]


# In[12]:


world[(world["forest"] >= 0.5) & (world["area"] >= 1)][["area", "population", "forest"]]


# In[13]:


world[lambda x: x["area"] > 5][["area", "population", "forest"]]


# ### 3.4 Group by

# In[14]:


world.groupby("official_lang")["country"].count().sort_values(ascending=False)[lambda x: x!=1]


# In[15]:


world["official_lang"].value_counts()[lambda x: x!=1]


# In[16]:


world.groupby("official_lang")[["area", "population"]].sum().sort_values("population", ascending=False).head(8)


# In[17]:


world.groupby(["official_lang"]).agg({
  "area": "sum",
  "country": "count",
  "density": "mean"
}).reset_index().sort_values("country", ascending=False).head()


# ### 4. Add column

# In[18]:


world["check_density"] = round(world[world["population"] > 1]["population"] / world["area"])
world["diff_density"] = round(abs(world["density"] - world["check_density"]) / world["density"], 2)
world[["density", "check_density", "diff_density"]].dropna().sort_values("diff_density", ascending=False)


# In[19]:


del world["check_density"]
world = world.drop(['diff_density'], axis='columns')


# ### 5. Other

# In[20]:


world.query("life_exp < 60").sample(5)


# In[21]:


world.official_lang.unique()


# In[22]:


"Number of Null Elements in Each Column:"

world.isnull().sum()


# In[23]:


"Percentage of Null Elements in Each Column:"

(world.isnull().mean()*100).round(2)


# ### 6. Pivot table

# In[24]:


short = world.sample(15)
short.pivot_table(values="life_exp", index=["density"], columns=["physicians"]).fillna("-")


# ### 7. Chart

# In[25]:


import matplotlib.pyplot as plt
new_sample = world.set_index("physicians")["life_exp"].sort_values()
new_sample.plot()
plt.show()


# ### 8. Merge

# In[26]:


df1 = world.iloc[0:10][["area", "population"]]
df2 = world.iloc[5:15][["population", "forest"]]
res = df1.merge(df2, 'left', on='country')
res


# ### 9. Copy data

# In[27]:


world_copy = world.iloc[0:15, 0:3].copy(deep=True)
world_copy.area = world_copy.area * 1000000
world_copy.population = world_copy.population * 1000000
world_copy.head()


# ### 10. In

# In[28]:


world[world['official_lang'].isin(['English', 'French'])].head()


# ### 11. Iter

# In[29]:


for idx,row in world[:2].iterrows():
    print(idx, row.area)