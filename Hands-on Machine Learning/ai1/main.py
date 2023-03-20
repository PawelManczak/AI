import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def prepapare_country_stats(oecd, gdp):
    justOECD = oecd[["Country", "Value"]]

    justGDP = gdp[["Country", "2015"]]
    df_cd = pd.merge(justOECD, justGDP, how='left', on='Country')
    df_cd = df_cd.iloc[:-1, :]
    df_cd.rename(columns={"2015": "PKB per capita"}, inplace=True)
    df_cd.rename(columns={"Value": "Satisfaction"}, inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return df_cd.iloc[keep_indices]


# loading data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=",")
gdp_per_Capita = pd.read_csv("gdp_per_capita.csv", thousands=",", delimiter='\t', encoding='latin1', na_values="bd")

# preparing data
country_Stats = prepapare_country_stats(oecd_bli, gdp_per_Capita)
# print(country_Stats)
x = np.c_[country_Stats["PKB per capita"]]
y = np.c_[country_Stats["Satisfaction"]]

# Visualization
country_Stats.plot(kind='scatter', x='PKB per capita', y="Satisfaction" )


model = sklearn.linear_model.LinearRegression()
model.fit(x, y)


y_pred = model.predict(x)
plt.plot(x, y_pred, color="blue", linewidth=3)
plt.show()