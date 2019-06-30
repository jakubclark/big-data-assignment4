import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.stats import pearsonr

# NOTE: This file depends on a .csv file called `cpi_csv`, which is provided with the uploaded
# Also, the plots are saved to a folder called `plots`, which will need to be created first

matplotlib.rc('font', size=14)


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

    def make_plot():
        print(f'Plotting mean sale price over time.')
        subset = df[['PRICE', 'SALEDATE']].dropna()

        subset['YEAR'] = subset['SALEDATE'].apply(lambda date: int(date.split('-')[0]))
        groups = subset.groupby('YEAR')

        final_df = pd.DataFrame(columns=['YEAR', 'MEAN_PRICE', 'ADJUSTED_MEAN_PRICE'])
        for year, props in groups:
            if year in (1991, 2007, 2008, 2014, 2015):
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
            figsize=(9, 5)
        )

        plot.minorticks_on()
        plot.set_axisbelow(True)
        plot.grid(linestyle='--', linewidth='0.5', color='black', which='major')

        plot.set_title('Mean Price over Time - Without Outliers', fontsize=12)
        plot.set_xlabel('Year', fontsize=12)
        plot.set_ylabel('Price ($)', fontsize=12)

        fig = plot.get_figure()
        fig.savefig('plots/mean_price_over_time.pdf')
        fig.savefig('plots/mean_price_over_time.png', dpi=300)
        plt.show()

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

    hist = prices.hist(
        figsize=(8, 5)
    )

    plt.title('Price Histogram', fontsize=12)
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    fig = hist.get_figure()
    fig.savefig('plots/price_histogram.pdf')
    fig.savefig('plots/price_histogram.png', dpi=300)
    plt.show()


def plot_boxplots_by_quadrant(df):
    print('Plotting sale-price boxplots by grade')
    subset = df[['PRICE', 'QUADRANT', 'QUALIFIED']].dropna().query('QUALIFIED == "Q"')
    groups = subset.groupby('QUADRANT')
    quads = {}

    for quadrant, props in groups:
        quads[quadrant] = props['PRICE']

    fig, ax = plt.subplots()

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
    ax.set_title('Price Boxplot by Quadrant', fontsize=12)

    plt.setp(bp['boxes'], color='blue', linewidth=1)
    plt.setp(bp['whiskers'], color='black', linewidth=1)
    plt.setp(bp['caps'], color='black', linewidth=1)
    plt.setp(bp['medians'], color='red', linewidth=1)

    fig.savefig('plots/boxplots_by_quadrant.pdf')
    fig.savefig('plots/boxplots_by_quadrant.png', dpi=300)
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

    fig, ax = plt.subplots()

    ax.barh(x, y, height=0.4)

    ax.set_axisbelow(True)
    ax.grid(linestyle='--', linewidth='0.5', color='black', which='major', axis='x')

    minor_xticks = np.arange(0, 50000, 1000)
    major_xticks = np.arange(0, 50000, 10000)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)

    ax.set_xlabel('Number of Properties', fontsize=12)
    ax.set_ylabel('Quadrant', fontsize=12)
    ax.set_title('Number of Properties by Quadrant', fontsize=12)

    fig.savefig('plots/num_properties_by_quadrant.pdf')
    fig.savefig('plots/num_properties_by_quadrant.png', dpi=300)
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
        x='LONGITUDE', y='LATITUDE', s=0.5,
        c='PRICE ($)', colormap='inferno', marker='H',
        norm=LogNorm(), fontsize=12, figsize=(8, 5))

    plot.set_axisbelow(True)
    plot.grid(linestyle='--', linewidth='0.5', color='black', which='major')
    plot.set_title('Mean Price, by Coordinate', fontsize=12)
    plt.minorticks_on()

    fig = plot.get_figure()
    fig.savefig('plots/mean_price_by_coordinate.pdf')
    fig.savefig('plots/mean_price_by_coordinate.png', dpi=300)
    plt.show()

def plot_area_by_neighbourhood(df: pd.DataFrame):
    print("Plotting area by neighborhood")
    groups = df.groupby('ASSESSMENT_NBHD')

    mean = {}

    colors = {
        'NW': 'red',
        'NE': 'blue',
        'SW': 'green',
        'SE': 'black'
    }

    red_patch = matplotlib.patches.Patch(color='red', label='NW')
    blue_patch = matplotlib.patches.Patch(color='blue', label='NE')
    green_patch = matplotlib.patches.Patch(color='green', label='SW')
    black_patch = matplotlib.patches.Patch(color='black', label='SE')

    handles = [red_patch, blue_patch, green_patch, black_patch]
    quadrant_legend = ['NW', 'NE', 'SW', 'SE']

    nbhd_labels = []

    area_colors = []

    new_df = pd.DataFrame(columns=['QUADRANT', 'MEAN_LIVING_GBA'])

    for neighbourhood, props in groups:
        quadrant = props['QUADRANT'].iloc[0]
        living_gba: pd.Series = props['LIVING_GBA']
        living_gba = living_gba.dropna()
        mean_living_gba = living_gba.mean()
        if type(mean_living_gba) is not float:
            mean[neighbourhood] = (mean_living_gba, quadrant)
            nbhd_labels.append(neighbourhood)
            area_colors.append(colors[quadrant])

    x_pos = np.arange(len(nbhd_labels))

    mean_areas = [mean[nbhd][0] for nbhd in nbhd_labels]

    mean_price = {}

    for neighbourhood, props in groups:
        price: pd.Series = props['PRICE']
        price = price.dropna()
        sum_price = sum(price)
        len_price = len(price)
        if len_price != 0:
            mean_price[neighbourhood] = sum_price / len_price

    mean_prices = [mean_price[nbhd] for nbhd in nbhd_labels]

    area_price = sorted(list(zip(mean_prices, mean_areas)), key=lambda x: x[1])

    print("Calculating pearson correlation between price and GBA")
    pear = pearsonr(mean_prices, mean_areas)
    print(f'pearsonr: {pear}')

    fig, ax = plt.subplots()

    ax.bar(x_pos, mean_areas, align='center', color=area_colors)

    ax.set_axisbelow(True)
    ax.grid(linestyle='-', linewidth='0.2', color='black', which='both')

    minor_yticks = np.arange(0, 3000, 100)
    major_yticks = np.arange(0, 3000, 100)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(nbhd_labels)

    plt.xticks(rotation=90)

    # Legend
    plt.legend(handles=handles, loc=1)

    ax.set_xlabel('Neighbourhoods')
    ax.set_ylabel('Mean Area (Sq Ft)')
    ax.set_title('Mean Area By Neighbourhood')

    fig.savefig('plots/mean_area_by_neighbourhood.pdf')
    fig.savefig('plots/mean_area_by_neighborhood.png', dpi=300)
    plt.show()

def plot_area_price_correlation(df: pd.DataFrame):
    
    mean_price = {}

    for neighbourhood, props in groups:
        price: pd.Series = props['PRICE']
        price = price.dropna()
        sum_price = sum(price)
        len_price = len(price)
        if len_price != 0:
            mean_price[neighbourhood] = sum_price / len_price

    mean_prices = [mean_price[nbhd] for nbhd in nbhd_labels]

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
        x='LONGITUDE', y='LATITUDE', s=0.5,
        c='Number of Properties', colormap='viridis', marker='H',
        norm=LogNorm(), fontsize=12, figsize=(8, 5))

    plot.set_axisbelow(True)
    plot.grid(linestyle='--', linewidth='0.5', color='black', which='both')
    plot.set_title('Number of Properties, by Coordinate', fontsize=12)

    fig = plot.get_figure()
    fig.savefig('plots/num_properties_by_coordinate.pdf')
    fig.savefig('plots/num_properties_by_coordinate.png', dpi=300)
    plt.show()


def compute_correlation_price_vs_longitude(df):
    subset = (df[['LONGITUDE', 'PRICE', 'QUALIFIED']].dropna()
              .query('QUALIFIED == "Q"'))
    lons = subset['LONGITUDE']
    prices = subset['PRICE']

    pear = pearsonr(lons, prices)
    print(f'pearsonr: {pear}')

    plot = subset.plot.scatter(x='LONGITUDE', y='PRICE', s=5)

    plot.minorticks_on()
    plot.set_axisbelow(True)
    plot.grid(linestyle='--', linewidth='0.5', color='black', which='major', axis='both')

    plot.set_title('Longitude vs Price', fontsize=12)
    plot.set_xlabel('Longitude', fontsize=12)
    plot.set_ylabel('Price ($)', fontsize=12)

    fig = plot.get_figure()
    fig.savefig('plots/longitude_vs_price.pdf')
    fig.savefig('plots/longitude_vs_price.png', dpi=300)
    plt.show()

    pass


def plots_by_build_date(df):
    def plot_all():
        subset = df[['AYB', 'PRICE']].dropna().groupby('AYB').agg('mean').sort_values('AYB')
        subset['Build Date'] = subset.index

        plot = subset.plot.line(x='Build Date', y='PRICE', title='Build Date vs Mean Sale Price', grid=True,
                                legend=True)
        plot.minorticks_on()

        plot.set_ylabel('Price ($) (in 10-millions)', fontsize=12)
        plot.set_xlabel('Build Year', fontsize=12)
        fig = plot.get_figure()

        fig.savefig('plots/price_over_time_by_build_date.pdf')
        fig.savefig('plots/price_over_time_by_build_date.png', dpi=300)
        plt.show()

    def plot_nineties():
        df1 = df[df[['PRICE', 'AYB']].apply(np.isclose, b=1990, atol=10).any(1)]
        df_age1 = df1[['AYB', 'PRICE']].groupby('AYB').agg('mean').sort_values('AYB')
        df_age2 = df_age1.drop(df_age1.index[[0, 1]])
        df_age2.reset_index()
        plot = df_age2.plot(grid=True, title='Prices from 1980 to 2000', fontsize=12)

        plot.minorticks_on()
        plot.set_xlabel('Year', fontsize=12)
        plot.set_ylabel('Price', fontsize=12)

        plot.set_xticks(np.arange(1980, 2001, 5))
        plot.set_xticks(np.arange(1980, 2001, 1), minor=True)

        fig = plot.get_figure()
        fig.savefig('plots/price_over_time_by_build_date_1980-2000.png', dpi=300)
        fig.savefig('plots/price_over_time_by_build_date_1980-2000.pdf')
        plt.show()

    plot_all()
    plot_nineties()


def plots_price_frequencies(df):
    def plot_all():
        df_age = df[['AYB', 'PRICE']].groupby('AYB').agg('mean').sort_values('AYB')

        plot = df_age.PRICE.hist()

        plot.set_title('Price Histogram', fontsize=12)
        plot.set_xlabel('Price ($)', fontsize=12)
        plot.set_ylabel('Number of Properties', fontsize=12)

        fig = plot.get_figure()
        fig.savefig('plots/price_freq.png', dpi=300)
        fig.savefig('plots/price_freq.pdf')
        plt.show()

    def plot_nineties():
        df1 = df[df[['PRICE', 'AYB']].apply(np.isclose, b=1990, atol=10).any(1)]
        df_age1 = df1[['AYB', 'PRICE']].groupby('AYB').agg('mean').sort_values('AYB')
        df_age2 = df_age1.drop(df_age1.index[[0, 1]])
        df_age3 = df_age2.reset_index()

        plot = df_age3.PRICE.hist(color=(0.5, 0.1, 0.5, 0.6), grid=True)
        plot.set_title('Price Histogram from 1980 to 2000', fontsize=12)
        plot.set_xlabel('Price ($)', fontsize=12)
        plot.set_ylabel('Number of Properties', fontsize=12)

        fig = plot.get_figure()
        fig.savefig('plots/price_freq80.png', dpi=300)
        fig.savefig('plots/price_freq80.pdf')
        plt.show()

    plot_all()
    plot_nineties()


def price_vs_grade_and_condition(full_df):
    grade_to_num = {
        'No Data': 0, 'Low Quality': 1, 'Fair Quality': 2,
        'Average': 3, 'Above Average': 4, 'Good Quality': 5,
        'Very Good': 6, 'Superior': 7, 'Excellent': 8,
        'Exceptional-A': 9, 'Exceptional-B': 10, 'Exceptional-C': 11,
        'Exceptional-D': 12}

    condition_to_num = {
        'Default': 0, 'Poor': 1, 'Fair': 2,
        'Average': 3, 'Good': 4, 'Very Good': 5,
        'Excellent': 6}

    def correlate(x_col):
        """
        Correlates `x_col` with PRICE, using `test_type` as the correlation test
        """
        y = subset['PRICE']
        x = subset[x_col]
        res = stats.pearsonr(x, y)
        print(f'Correlation between PRICE and {x_col}: {res}')

    def regress(target_, preds):
        preds = sm.add_constant(preds)
        model = sm.OLS(target_, preds).fit()
        print(model.summary())

    def scatter_column(column):
        plot = subset.plot.scatter(
            x=column,
            y='PRICE',
            title=f'Price vs {column}'
        )
        plot.minorticks_on()
        plot.set_axisbelow(True)
        plot.grid(linestyle='--', linewidth='0.5', color='black', which='major')

        plot.set_xlabel(column, fontsize=12)
        plot.set_ylabel('Price ($)', fontsize=12)

        fig = plot.get_figure()
        fig.savefig(f'plots/{column}_vs_price.pdf')
        fig.savefig(f'plots/{column}_vs_price.png', dpi=300)
        plt.show()

    def bar_column(column, to_ignore=None):
        """
        Plot the mean price, by `column` as a horizontal-bar-graph.
        `labels` determines the order in which the groups appear in.
        """

        groups = subset.groupby(column)

        final_df = pd.DataFrame(columns=[column])

        for col, props in groups:
            if col == to_ignore:
                continue
            mean_price = props['PRICE'].mean()
            final_df = final_df.append(pd.Series({
                'MEAN_PRICE': mean_price,
                column: col
            }, name=col))

        final_df.sort_values(by='MEAN_PRICE', inplace=True)
        plot = final_df.plot.barh(x=column, y='MEAN_PRICE', title=f'Mean Price by {column}', grid=True,
                                  fontsize=12, figsize=(13, 5))
        plot.minorticks_on()
        plot.set_axisbelow(True)
        plot.grid(linestyle='--', linewidth='0.5', color='black', which='major')

        plot.set_xlabel('Mean Price ($)', fontsize=12)

        fig = plot.get_figure()
        fig.savefig(f'plots/mean_price_by_{column}.pdf')
        fig.savefig(f'plots/mean_price_by_{column}.png', dpi=300)
        plt.show()

    # Clean up the data
    subset = full_df[['GRADE', 'CNDTN', 'LANDAREA', 'SQUARE', 'ROOMS', 'PRICE']].dropna().query('SQUARE != "PAR "')
    subset['SQUARE'] = subset['SQUARE'].apply(int)
    subset['Condition'] = subset['CNDTN']
    subset['Grade'] = subset['GRADE']

    # Plot Mean Price by Grade
    bar_column('Grade', to_ignore='No Data')

    # Plot Mean Price by Condition
    bar_column('Condition', to_ignore='Default')

    # Plot Land Area vs Price
    scatter_column('LANDAREA')

    # Plot Square vs Price
    scatter_column('SQUARE')

    # Convert these to numerical values
    subset['GRADE'] = subset['GRADE'].apply(lambda c: grade_to_num[c])
    subset['CNDTN'] = subset['CNDTN'].apply(lambda c: condition_to_num[c])

    # Do a regression
    target = subset['PRICE']
    predictors = subset[['LANDAREA', 'SQUARE', 'GRADE', 'CNDTN']]
    regress(target, predictors)

    # Correlate these variables with the price
    correlate('LANDAREA')
    correlate('SQUARE')
    correlate('GRADE')
    correlate('CNDTN')
    correlate('ROOMS')


def main():
    df = pd.read_csv('DC_Properties.csv')

    compute_basic_price_distribution(df)
    plot_price_histogram(df)
    plot_boxplots_by_quadrant(df)
    plot_price_by_coordinate(df)
    plot_count_by_quadrant(df)
    plot_count_by_coordinate(df)
    plot_area_by_neighbourhood(df)
    plot_adjusted_price_over_time(df)
    compute_correlation_price_vs_longitude(df)

    plots_by_build_date(df)
    plots_price_frequencies(df)

    price_vs_grade_and_condition(df)

    print('Done creating the plots')


if __name__ == '__main__':
    main()
