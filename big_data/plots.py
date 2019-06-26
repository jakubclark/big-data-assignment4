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
    def compute_data():
        res = {}

        for _, props in subset.iterrows():
            lon = props['LONGITUDE']
            lat = props['LATITUDE']
            price = props['PRICE']
            lon = round(lon, 3)
            lat = round(lat, 3)
            key = (lon, lat)

            if price > 1500000:
                continue

            if key in res.keys():
                res[key].append(price)
            else:
                res[key] = [price]

        return res

    def compute_data_points():
        data = compute_data()
        x_, y_, c_ = [], [], []
        mean_prices = []

        for coords, prices in data.items():
            lon, lat = coords

            x_.append(lon)
            y_.append(lat)

            mean_price = sum(prices) / len(prices)
            mean_prices.append(mean_price)

        max_price = max(mean_prices)

        for mean_price in mean_prices:
            color = 100 - round((mean_price / max_price) * 100)
            c_.append(color)

        return x_, y_, c_

    def plot_data_points():
        plt.clf()

        fig = plt.figure(figsize=(9, 9))
        ax = fig.gca()
        ax.set_title('Mean Price map')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        ax.scatter(x, y, s=5, c=c, marker='H', cmap='inferno')
        # ax.scatter([-77.0017], [38.8844], c='black', s=500, marker='x')

        ax.set_axisbelow(True)
        ax.grid(linestyle='-', linewidth='0.1', color='black', which='both')

        plt.savefig('mean_price_by_coordinate.pdf')
        plt.show()

    subset = df[['LONGITUDE', 'LATITUDE', 'PRICE']].dropna()

    x, y, c = compute_data_points()
    plot_data_points()


def plot_count_heatmap(df: pd.DataFrame):
    subset = df[['LONGITUDE', 'LATITUDE']].dropna()
    lons = subset['LONGITUDE']
    lats = subset['LATITUDE']

    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=(50, 50))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca()
    ax.set_title('Number of Properties Heatmap')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.imshow(heatmap, extent=extent, origin='lower')

    plt.savefig('number_of_properties_heatmap.pdf')
    plt.show()


def main():
    df = pd.read_csv('DC_Properties.csv')
    plot_price_by_grade(df)
    plot_price_by_quadrant(df)
    plot_price_over_time(df)
    plot_price_heatmap(df)
    plot_count_heatmap(df)


if __name__ == '__main__':
    main()
