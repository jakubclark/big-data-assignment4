import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def plot_price_by_quadrant(df: pd.DataFrame):
    groups = df.groupby('QUADRANT')

    means = {}

    for quadrant, props in groups:
        prices: pd.Series = props['PRICE']
        prices = prices.dropna()
        mean_price = sum(prices) / len(prices)
        means[quadrant] = mean_price

    quadrant_labels = ['NW', 'NE', 'SE', 'SW']
    y_pos = np.arange(len(quadrant_labels))

    mean_prices = [means[quadrant] for quadrant in quadrant_labels]

    fig, ax = plt.subplots()

    ax.barh(y_pos, mean_prices, align='center')

    ax.set_axisbelow(True)
    ax.grid(linestyle='-', linewidth='0.5', color='black', which='both')

    minor_xticks = np.arange(0, 1500001, 100000)
    major_xticks = np.arange(0, 1500001, 500000)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(quadrant_labels)

    ax.set_xlabel('Mean Price ($)')
    ax.set_ylabel('Quadrant Label')
    ax.set_title('Mean Price By Quadrant')

    fig.savefig('mean_price_by_quadrant.pdf')
    plt.show()


def main():
    df = pd.read_csv('DC_Properties.csv')
    plot_price_by_grade(df)
    plot_price_by_quadrant(df)


if __name__ == '__main__':
    main()
