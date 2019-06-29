

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm
import numpy as np
from builtins import str




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
    
def grade_to_num(grade):
    
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
    #print(result)
    return result
 
 
def condition_to_num(condition):
    
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

def remove_placeholders(df, column):
    df = df[df[column] != 0]
    return df


def scatter_column(df, column):
    plt.scatter(df['PRICE'], df[column])
    plt.title('The relation between Price and ' + column)
    plt.xlabel('Price')
    plt.ylabel(column)
    plt.savefig('scatter.pdf')
    plt.close()
    None

def bar_column(df, column, grade_labels):
    placeholder = 'Default'
    
    if column == 'GRADE':
        placeholder = 'No Data'
    
    
    groups = df.groupby(column)
    
    means = {}
    
    for grade, props in groups:
        prices: pd.Series = props['PRICE']
        prices = prices.dropna()
        mean_price = sum(prices) / len(prices)
        means[grade] = mean_price

    # 'No Data' is omitted
    del means[placeholder]
    
    
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
    ax.set_ylabel(column + ' Label')
    ax.set_title('Mean Price By ' + column)
    
    name = 'mean_price_by_' + column + '.pdf'
    fig.savefig(name)
    plt.show()
    
    None

    
def main():
    df = pd.read_csv('DC_Properties.csv')
    
    struct_labels = ['Multi', 'Row End', 'Row Inside', 'Semi-Detached', 'Single', 
                     'Vacant Land', 'Town Inside', 'Town End']
    condition_labels = ['Poor', 'Fair', 'Average', 'Good', 'Very Good', 'Excellent']
    grade_labels = ['Low Quality', 'Fair Quality', 'Average', 'Good Quality', 'Above Average',
                    'Very Good', 'Superior', 'Excellent',
                    'Exceptional-A', 'Exceptional-B', 'Exceptional-C', 'Exceptional-D']
    bar_column(df, 'GRADE', grade_labels)
    bar_column(df, 'CNDTN', condition_labels)
    #bar_column(df, 'STRUCT', struct_labels)
    
    scatter_column(df, 'LANDAREA')
    
    df = df.drop(['CMPLX_NUM', 'LIVING_GBA'], axis= 1)
    
    df = df.dropna()
    
    square = df['SQUARE'].apply(string_to_int)
    df['Square'] = square
    
    res = df['GRADE'].apply(grade_to_num)
    df['Grade'] = res
    remove_placeholders(df, 'Grade')
    
    res = df['CNDTN'].apply(condition_to_num)
    df['Condition'] = res
    remove_placeholders(df, 'Condition')
    
    selector = df[['LANDAREA', 'Square', 'Grade', 'Condition']]
    
    regress(df['PRICE'], selector)
    
    
    
    y = df['PRICE']
    x = df['LANDAREA']
    cor, p = correlate(y, x)
    print("cor (price, land area):" )
    print(cor)
    
    x = df['Square']
    cor, p = correlate(y, x)
    print("cor (price, square):")
    print(cor)
    
    x = df['Grade']
    cor, p = correlate(y, x)
    print("cor (price, grade):")
    print(cor)
    
    x = df['Condition']
    cor, p = correlate(y, x)
    print("cor (price, condition):")
    print(cor)
    
    x = df['ROOMS']
    cor, p = correlate(y, x)
    print("cor (price, rooms):")
    print(cor)
    
    x = df['ROOMS']
    cor, p = correlate(y, x)
    print("cor (price, ac):")
    print(cor)
    
    
    None

if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    