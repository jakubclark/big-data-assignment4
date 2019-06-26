

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import numpy as np
from numpy.core._multiarray_umath import dtype
from builtins import str
from _ast import Str



def price_through_years(df: pd.DataFrame):
    groups = df.groupby('AYB')
    years = df['AYB'].drop_duplicates().dropna()
    means = np.zeros(len(years))
    
    for year, props in groups:
        prices: pd.Series = props['PRICE']
        prices = prices.dropna()
        mean_price = np.average(prices)
        means[year] = mean_price
    
    plt.Line2D(years, means)
    None

def plot_price_by_grade(df: pd.DataFrame):
    groups = df.groupby('GRADE')
    
    means = {}
    
    for grade, props in groups:
        prices: pd.Series = props['PRICE']
        prices = prices.dropna()
        mean_price = sum(prices) / len(prices)
        means[grade] = mean_price

    # 'No Data' is omitted
    del means['No Data']
    
    grade_labels = ['Low Quality', 'Fair Quality', 'Average', 'Good Quality', 'Above Average',
                    'Very Good', 'Superior', 'Excellent',
                    'Exceptional-A', 'Exceptional-B', 'Exceptional-C', 'Exceptional-D']
    y_pos = np.arange(len(grade_labels))

    mean_prices = [means[grade] for grade in grade_labels]

    fig, ax = plt.subplots()

    ax.barh(y_pos, mean_prices, align='center')

    ax.set_axisbelow(True)
    ax.grid(linestyle='-', linewidth='0.5', color='black', which='both')

    minor_xticks = np.arange(0, 7000001, 500000)
    major_xticks = np.arange(0, 7000001, 1000000)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(grade_labels)

    ax.set_xlabel('Mean Price ($)')
    ax.set_ylabel('Grade Label')
    ax.set_title('Mean Price By Grade')

    fig.savefig('mean_price_by_grade.pdf')
    plt.show()
    
    None
    
    

def correlate(x, y, test_type = 'pearson'):
    if test_type == 'pearson':
        return stats.pearsonr(x, y)
    
    return stats.kendalltau(x, y)



def regress(target, predictors):
    predictors = sm.add_constant(predictors)
    model = sm.OLS(target, predictors).fit()
    print(model.summary())
    
    None
    
def string_to_int(num):
    if type(num) is str:
        return 0    
    
    return num
    
def num_value_to_grade(grade):
    
    switch = {
        'No Data': 0,
        'Low Quality': 1,
        'Fair Quality': 2,
        'Average': 3,
        'Above Average': 4,
        'Good Quality': 5,
        'Very Good': 6,
        'Superior': 7,
        'Excellent': 8,
        'Exceptional-A': 9,
        'Exceptional-B': 10,
        'Exceptional-C': 11,
        'Exceptional-D': 12
    }
    
    result = switch.get(grade, "Invalid grade")
   
    return result
 
 
def num_value_to_condition(condition):
    
    switch = {
        'Default': 0,
        'Poor': 1,
        'Fair': 2,
        'Average': 3,
        'Good': 4,
        'Very Good': 5,
        'Excellent': 6
        }
    
    return switch.get(condition, "Invalid condition")


   
    
    
def main():
    df = pd.read_csv('DC_Properties.csv')
    #plot_price_by_grade(df)
    df = df.drop(['CMPLX_NUM', 'LIVING_GBA'], axis= 1)
    
    df = df.dropna()
    
    df['SQUARE'].apply(string_to_int)
    df = df.astype({'SQUARE': int})
    
    df['GRADE'].apply(num_value_to_grade)
    print(df['GRADE'])
    df = df.astype({'GRADE': int})
    
    selector = df[['LANDAREA', 'ROOMS', 'SQUARE']]
    
    regress(df['PRICE'], selector)
    
    
    
    y = df['PRICE']
    x = df['LANDAREA']
    cor, p = correlate(y, x)
    print(cor)
    
    x = df['SQUARE']
    cor, p = correlate(y, x)
    print(cor)
    
    x = df['GRADE']
    cor, p = correlate(y, x)
    print(cor)
    
    None

if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    