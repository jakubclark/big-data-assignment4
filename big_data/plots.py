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


def plot_price_over_time(df):
    subset = df[['PRICE', 'SALEDATE']].dropna()

    subset = subset.sort_values(by=['SALEDATE'])

    year_to_prices = {}

    for _, props in subset.iterrows():
        saledate = props['SALEDATE']
        price = props['PRICE']
        year = int(saledate.split('-')[0])
        if year in year_to_prices.keys():
            year_to_prices[year].append(price)
        else:
            year_to_prices[year] = [price]

    year_to_mean_prices = {}

    for year, prices in year_to_prices.items():
        mean_price = sum(prices) / len(prices)
        year_to_mean_prices[year] = mean_price

    x = list(year_to_mean_prices.keys())
    y = list(year_to_mean_prices.values())

    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.set_title('Year vs Mean Property Price')
    ax.set_xlabel('Saledate')
    ax.set_ylabel('Mean Price ($)')

    fig.savefig('mean_saleprice_over_time.pdf')
    plt.show()


def plot_price_heatmap(df: pd.DataFrame):
    def compute_color(price):
        """
        Go from price (a float) to a color (a string)
        """
        # TODO: Compute the color from a palette
        if 0.0 < price < 500000.0:
            return 'red'
        elif 500000.0 < price < 1000000.0:
            return 'blue'
        elif 1000000.0 < price < 1500000:
            return 'orange'
        else:
            return 'black'

    subset = df[['LONGITUDE', 'LATITUDE', 'PRICE']].dropna()

    data = {}

    for _, props in subset.iterrows():
        lon = props['LONGITUDE']
        lat = props['LATITUDE']
        value = props['PRICE']

        lon = round(lon, 3)
        lat = round(lat, 3)
        key = (lon, lat)

        if key in data.keys():
            data[key].append(value)
        else:
            data[key] = [value]
    x = []
    y = []
    c = []

    for coords, prices in data.items():
        lon, lat = coords
        mean_price = sum(prices) / len(prices)
        x.append(lon)
        y.append(lat)
        color = compute_color(mean_price)
        c.append(color)

    plt.clf()
    plt.title('Mean Price map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.scatter(x, y, s=1, c=c, marker='H')

    plt.savefig('mean_price_by_coordinate.pdf')
    plt.show()


def plot_count_heatmap(df: pd.DataFrame):
    subset = df[['LONGITUDE', 'LATITUDE']].dropna()
    lons = subset['LONGITUDE']
    lats = subset['LATITUDE']

    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=(50, 50))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.title('Number of Properties Heatmap')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.imshow(heatmap, extent=extent, origin='lower')

    plt.show()
    plt.savefig('number_of_properties_heatmap.pdf')


def main():
    df = pd.read_csv('DC_Properties.csv')
    plot_price_by_grade(df)
    plot_price_by_quadrant(df)
    plot_price_over_time(df)
    plot_price_heatmap(df)
    plot_count_heatmap(df)


if __name__ == '__main__':
    main()
