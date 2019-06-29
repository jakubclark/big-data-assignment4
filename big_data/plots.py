import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr


def plot_adjusted_price_over_time(df):
    def adjust_for_inflation(prices, year_):
        # Based on
        # https://www.usinflationcalculator.com/inflation/consumer-price-index-and-annual-percent-changes-from-1913-to-2008/
        # and
        # https://www.usinflationcalculator.com/frequently-asked-questions-faqs/#HowInflationCalculatorWorks
        mean = np.mean(prices)
        cpi = cpi_df.get_value(year_, 'Avg_CPI')
        adjusted = mean * (current_cpi / cpi)
        return adjusted

    cpi_df = pd.read_csv('cpi.csv', index_col='Year')
    current_cpi = 256.1

    def make_plot(ignore_outliers=False):
        print(f'Plotting mean sale price over time. ignore_outliers={ignore_outliers}')
        subset = df[['PRICE', 'SALEDATE']].dropna()

        subset['YEAR'] = subset['SALEDATE'].apply(lambda date: int(date.split('-')[0]))
        groups = subset.groupby('YEAR')

        final_df = pd.DataFrame(columns=['YEAR', 'MEAN_PRICE', 'ADJUSTED_MEAN_PRICE', 'MEAN_ADJUSTED_PRICE'])
        for year, props in groups:
            if not ignore_outliers and year in (1991, 2007, 2008, 2014, 2015):
                continue
            mean_price = props['PRICE'].mean()
            adjusted_mean_price = adjust_for_inflation(mean_price, year)
            final_df = final_df.append(pd.Series({
                'YEAR': year,
                'MEAN_PRICE': mean_price,
                'ADJUSTED_MEAN_PRICE': adjusted_mean_price
            }, name=year))

        plot = final_df.plot.line(
            x='YEAR',
            y=['ADJUSTED_MEAN_PRICE', 'MEAN_PRICE'],
        )

        plot.minorticks_on()
        plot.set_axisbelow(True)
        plot.grid(linestyle='--', linewidth='0.5', color='black', which='major')

        if ignore_outliers:
            plot.set_title('Mean Price over Time - Without Outliers')
        else:
            plot.set_title('Mean Price over Time - With Outliers')

        plot.set_xlabel('Year')
        plot.set_ylabel('Price ($)')

        fig = plot.get_figure()
        fig.savefig(f'mean_price_over_time_{ignore_outliers}.pdf')
        fig.savefig(f'mean_price_over_time_{ignore_outliers}.png', dpi=300)
        plt.show()

    make_plot(True)
    make_plot()


def compute_basic_price_distribution(df):
    print('Computing basic price distribution')

    subset = df[['PRICE', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    prices = subset['PRICE']

    min_ = prices.min()
    max_ = prices.max()
    median_price = prices.median()
    mean_price = prices.mean()

    parts = {
        'Minimum': min_,
        'Maximum': max_,
        'Median': median_price,
        'Mean': mean_price
    }
    print(' | '.join([f'{k}={v}' for k, v in parts.items()]))


def plot_price_histogram(df):
    print('Plotting sale-price histogram.')
    subset = df[['PRICE', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    prices = subset['PRICE']

    hist = prices.hist(figsize=(8, 4))

    plt.title('Price Histogram')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')

    fig = hist.get_figure()

    fig.savefig('price_histogram.pdf')
    fig.savefig('price_histogram.png', dpi=300)
    plt.show()


def plot_boxplots_by_quadrant(df):
    print('Plotting sale-price boxplots by grade')
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


def plot_count_by_quadrant(df):
    print('Plotting total number of properties by quadrant')
    subset = df[['PRICE', 'QUADRANT', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    groups = subset.groupby('QUADRANT')

    final_df = pd.DataFrame(columns=['Number of Properties'])
    for quadrant, props in groups:
        final_df = final_df.append(pd.Series({
            'Number of Properties': props['PRICE'].size
        }, name=quadrant))
    pass

    x = ['NW', 'NE', 'SE', 'SW']
    y = [final_df.get_value(quad, 'Number of Properties') for quad in x]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(x, y, height=0.4)

    ax.set_axisbelow(True)
    ax.grid(linestyle='--', linewidth='0.5', color='black', which='major', axis='x')

    minor_xticks = np.arange(0, 50000, 1000)
    major_xticks = np.arange(0, 50000, 5000)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)

    ax.set_xlabel('Number of Properties')
    ax.set_ylabel('Quadrant')
    ax.set_title('Number of Properties by Quadrant')

    fig.savefig('num_properties_by_quadrant.pdf')
    fig.savefig('num_properties_by_quadrant.png', dpi=300)
    plt.show()
    pass


def plot_price_by_coordinate(df: pd.DataFrame):
    print('Plotting mean sale-price by coordinate')
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
    plt.show()


def plot_count_by_coordinate(df: pd.DataFrame):
    print('Plotting num of properties by coordinate')
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
    plt.show()


def compute_correlation_price_vs_longitude(df):
    subset = (df[['LONGITUDE', 'PRICE', 'QUALIFIED']].dropna()
              .query('QUALIFIED == "Q"'))
    lons = subset['LONGITUDE']
    prices = subset['PRICE']

    pear = pearsonr(lons, prices)
    print(f'pearsonr: {pear}')

    plot = subset.plot.scatter(
        x='LONGITUDE',
        y='PRICE',
        s=5
    )

    plot.minorticks_on()
    plot.set_axisbelow(True)
    plot.grid(linestyle='--', linewidth='0.5', color='black', which='major', axis='both')

    plot.set_title('Longitude vs Price')
    plot.set_xlabel('Longitude')
    plot.set_ylabel('Price ($)')

    fig = plot.get_figure()
    fig.savefig('longitude_vs_price.pdf')
    fig.savefig('longitude_vs_price.png', dpi=300)
    plt.show()

    pass


def plots_by_build_date(df):
    def plot_all():
        subset = df[['AYB', 'PRICE']].dropna().groupby('AYB').agg('mean').sort_values('AYB')
        subset['Build Date'] = subset.index

        plot = subset.plot.line(
            x='Build Date',
            y='PRICE',
            figsize=(9, 5),
            title='Build Date vs Mean Sale Price',
            grid=True,
            legend=True
        )
        plot.minorticks_on()

        plot.set_ylabel('Price ($) (in 10-millions)')
        plot.set_xlabel('Build Year')
        fig = plot.get_figure()

        fig.savefig('price_over_time_by_build_date.pdf')
        fig.savefig('price_over_time_by_build_date.png', dpi=300)
        plt.show()

    def plot_nineties():
        df1 = df[df[['PRICE', 'AYB']].apply(np.isclose, b=1990, atol=10).any(1)]
        df_age1 = df1[['AYB', 'PRICE']].groupby('AYB').agg('mean').sort_values('AYB')
        df_age2 = df_age1.drop(df_age1.index[[0, 1]])
        df_age2.reset_index()
        hst2 = df_age2.plot()
        hst2.set_xlabel("Year")
        hst2.set_title('1980-2000')
        fig = hst2.get_figure()
        fig.savefig('price_over_time_by_build_date_1980-2000.png', dpi=300)
        fig.savefig('price_over_time_by_build_date_1980-2000.pdf')
        plt.show()

    plot_all()
    plot_nineties()


def plots_price_frequencies(df):
    def plot_all():
        df_age = df[['AYB', 'PRICE']].groupby('AYB').agg('mean').sort_values('AYB')

        plot = df_age.PRICE.hist()

        plot.set_xlabel('Price')
        plot.set_title('Price frequency in general')
        plot.set_ylabel('Number of properties')

        fig = plot.get_figure()
        fig.savefig('price_freq.png', dpi=300)
        fig.savefig('price_freq.pdf')
        plt.show()

    def plot_nineties():
        df1 = df[df[['PRICE', 'AYB']].apply(np.isclose, b=1990, atol=10).any(1)]
        df_age1 = df1[['AYB', 'PRICE']].groupby('AYB').agg('mean').sort_values('AYB')
        df_age2 = df_age1.drop(df_age1.index[[0, 1]])
        df_age3 = df_age2.reset_index()

        plot = df_age3.PRICE.hist(color=(0.5, 0.1, 0.5, 0.6))

        plot.set_xlabel('Price')
        plot.set_title("Price frequency 1980-2000")
        plot.set_ylabel('Number of properties')

        fig = plot.get_figure()
        fig.savefig('price_freq80.png', dpi=300)
        fig.savefig('price_freq80.pdf')
        plt.show()

    plot_all()
    plot_nineties()


def main():
    df = pd.read_csv('DC_Properties.csv')

    compute_basic_price_distribution(df)
    plot_price_histogram(df)
    plot_boxplots_by_quadrant(df)
    plot_price_by_coordinate(df)
    plot_count_by_quadrant(df)
    plot_count_by_coordinate(df)
    plot_adjusted_price_over_time(df)
    compute_correlation_price_vs_longitude(df)

    plots_by_build_date(df)
    plots_price_frequencies(df)
    print('Done creating the plots')


if __name__ == '__main__':
    main()
