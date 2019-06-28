import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


def plot_price_by_grade(df: pd.DataFrame):
    print('Plotting mean price by grade | QUALIFIED == Q')
    subset = df[['GRADE', 'PRICE', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    groups = subset.groupby('GRADE')

    means = {}

    for grade, props in groups:
        prices: pd.Series = props['PRICE']
        prices = prices.dropna()
        mean_price = sum(prices) / len(prices)
        means[grade] = mean_price

    grade_labels = ['Low Quality', 'Fair Quality', 'Average', 'Good Quality', 'Above Average',
                    'Very Good', 'Superior', 'Excellent',
                    'Exceptional-A', 'Exceptional-B', 'Exceptional-C', 'Exceptional-D']
    y_pos = np.arange(len(grade_labels))

    mean_prices = [means[grade] for grade in grade_labels]

    fig, ax = plt.subplots()

    ax.barh(y_pos, mean_prices, align='center')

    ax.set_axisbelow(True)
    ax.grid(linestyle='--', linewidth='0.5', color='black', which='both')

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


def plot_price_over_time(df):
    print('Plotting price over time | QUALIFIED == Q')
    subset = df[['PRICE', 'SALEDATE', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
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
    fig.savefig('mean_saleprice_over_time.png', dpi=300)
    fig.savefig('mean_saleprice_over_time.svg')
    plt.show()


def compute_basic_price_distribution(df):
    def do_computation():
        min_ = prices.min()
        max_ = prices.max()
        median_price = prices.median()
        mean_price = prices.mean()

        parts = {
            'Minimum Price': min_,
            'Maximum Price': max_,
            'Median Price': median_price,
            'Mean Price': mean_price
        }

        s = ' | '.join([f'{k}: ${v}' for k, v in parts.items()])
        print(s)
        return min_, max_

    print('Computing basic price distribution')
    subset = df[['PRICE', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    prices = subset['PRICE']

    min_price, max_price = do_computation()

    print('Removing the max and min prices')
    prices = prices.where(lambda x: min_price < x).where(lambda x: x < max_price)
    do_computation()


def plot_price_histogram(df):
    print('Plotting price histogram | QUALIFIED == Q')
    subset = df[['PRICE', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    prices = subset['PRICE']

    hist = prices.hist(figsize=(8, 4))

    plt.title('Price Histogram')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')

    fig = hist.get_figure()

    fig.savefig('price_histogram.pdf')
    fig.savefig('price_histogram.png', dpi=300)
    fig.savefig('price_histogram.svg')
    plt.show()


def plot_price_by_quadrant(df: pd.DataFrame):
    print('Plotting mean price by quadrant | QUALIFIED == Q')
    groups = df[['PRICE', 'QUADRANT', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"').groupby('QUADRANT')

    final_df = pd.DataFrame(columns=['QUADRANT', 'MEAN PRICE'])

    for quadrant, props in groups:
        mean_price = props['PRICE'].mean()
        final_df = final_df.append(pd.Series({
            'QUADRANT': quadrant,
            'MEAN PRICE': mean_price
        }, name=quadrant))

    x = ['NW', 'NE', 'SE', 'SW']
    y = [final_df.get_value(quad, 'MEAN PRICE') for quad in x]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(x, y, height=0.4)

    ax.set_axisbelow(True)
    ax.grid(linestyle='--', linewidth='0.5', color='black', which='major', axis='x')

    minor_xticks = np.arange(0, 1500001, 20000)
    major_xticks = np.arange(0, 1500001, 200000)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)

    ax.set_xlabel('Mean Price ($)')
    ax.set_ylabel('Quadrant')
    ax.set_title('Mean Price by Quadrant')

    fig.savefig('mean_price_by_quadrant.pdf')
    fig.savefig('mean_price_by_quadrant.png', dpi=300)
    fig.savefig('mean_price_by_quadrant.svg')
    plt.show()
    pass


def plot_boxplots_by_quadrant(df):
    print('Plotting boxplots by grade | QUALIFIED == Q')
    subset = df[['PRICE', 'QUADRANT', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    groups = subset.groupby('QUADRANT')
    quads = {}

    for quadrant, props in groups:
        quads[quadrant] = props['PRICE']

    fig, ax = plt.subplots(figsize=(9, 5))

    labels = ['NW', 'NE', 'SE', 'SW']
    prices = [quads['NW'], quads['NE'], quads['SE'], quads['SW']]

    bp = ax.boxplot(prices,
                    labels=labels,
                    vert=0
                    )

    ax.set_axisbelow(True)
    ax.grid(linestyle='--', linewidth='0.5', color='black', which='both', axis='x')

    ax.set_xscale('log')

    major_xticks = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    major_xticks_labels = ['1e1', '1e2', '1e3', '1e4', '1e5', '1e6', '1e7', '1e8']
    ax.set_xticks(major_xticks, major_xticks_labels)

    ax.set_ylabel('Quadrant')
    ax.set_xlabel('Price ($)')
    ax.set_title('Price Boxplot by Quadrant')

    plt.setp(bp['boxes'], color='blue', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)

    fig.savefig('boxplots_by_quadrant.pdf')
    fig.savefig('boxplots_by_quadrant.png', dpi=300)
    plt.show()


def plot_price_by_coordinate(df: pd.DataFrame):
    print('Plotting mean price by coordinate | QUALIFIED == Q')
    subset = df[['LONGITUDE', 'LATITUDE', 'PRICE', 'QUALIFIED']].dropna() \
        .query('QUALIFIED == "Q"')
    subset['PRICE ($)'] = subset['PRICE']

    lats = subset['LATITUDE'].apply(lambda c: round(c, 3))
    lons = subset['LONGITUDE'].apply(lambda c: round(c, 3))
    prices = subset['PRICE']

    temp = pd.concat([lats, lons, prices], axis=1)

    by_coords = temp.groupby(['LATITUDE', 'LONGITUDE'])

    final_df = pd.DataFrame(columns=['LATITUDE', 'LONGITUDE', 'PRICE ($)'])

    for coords, props in by_coords:
        mean_price = props['PRICE'].mean()
        new_series = pd.Series({
            'LATITUDE': coords[0],
            'LONGITUDE': coords[1],
            'PRICE ($)': mean_price
        }, name=coords)
        final_df = final_df.append(new_series)

    plot: 'FramePlotMethods' = final_df.plot.scatter(
        x='LONGITUDE',
        y='LATITUDE',
        s=0.5,
        c='PRICE ($)',
        colormap='inferno',
        marker='H',
        norm=LogNorm()
    )

    plot.set_axisbelow(True)
    plot.grid(linestyle='--', linewidth='0.5', color='black', which='major')
    plot.set_title('Mean Price, by Coordinate')
    plt.minorticks_on()

    fig = plot.get_figure()
    fig.savefig('mean_price_by_coordinate.pdf')
    fig.savefig('mean_price_by_coordinate.png', dpi=300)
    fig.savefig('mean_price_by_coordinate.svg')
    plt.show()


def plot_count_by_coordinate(df: pd.DataFrame):
    print('Plotting num of properties by coordinate | QUALIFIED == Q')
    subset = df[['LONGITUDE', 'LATITUDE', 'PRICE', 'QUALIFIED']].dropna() \
        .query('QUALIFIED == "Q"')

    lats = subset['LATITUDE'].apply(lambda c: round(c, 3))
    lons = subset['LONGITUDE'].apply(lambda c: round(c, 3))
    prices = subset['PRICE']

    temp = pd.concat([lats, lons, prices], axis=1)

    by_coords = temp.groupby(['LATITUDE', 'LONGITUDE'])

    final_df = pd.DataFrame(columns=['LATITUDE', 'LONGITUDE', 'Number of Properties'])

    for coords, props in by_coords:
        num = len(props['PRICE'])
        new_series = pd.Series({
            'LATITUDE': coords[0],
            'LONGITUDE': coords[1],
            'Number of Properties': num
        }, name=coords)
        final_df = final_df.append(new_series)

    plot: 'FramePlotMethods' = final_df.plot.scatter(
        x='LONGITUDE',
        y='LATITUDE',
        s=0.5,
        c='Number of Properties',
        colormap='viridis',
        marker='H',
        norm=LogNorm()
    )

    plot.set_axisbelow(True)
    plot.grid(linestyle='--', linewidth='0.5', color='black', which='both')
    plot.set_title('Number of Properties, by Coordinate')

    fig = plot.get_figure()
    fig.savefig('num_properties_by_coordinate.pdf')
    fig.savefig('num_properties_by_coordinate.png', dpi=300)
    fig.savefig('num_properties_by_coordinate.svg')
    plt.show()


def main():
    df = pd.read_csv('DC_Properties.csv')

    plot_price_by_grade(df)
    plot_price_over_time(df)

    compute_basic_price_distribution(df)
    plot_price_histogram(df)
    plot_price_by_quadrant(df)
    plot_boxplots_by_quadrant(df)
    plot_price_by_coordinate(df)
    plot_count_by_coordinate(df)
    print('Done creating the plots')


if __name__ == '__main__':
    main()
