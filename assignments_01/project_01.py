import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from prefect import flow, task
from prefect.logging import get_run_logger

OUTPUT_PATH = './assignments_01/outputs/'

@task(retries=3, retry_delay_seconds=2)
def load_df():
    dfs = []
    logger = get_run_logger()
    for year in range(2015, 2025):
        df = pd.read_csv(f'./happiness_project/world_happiness_{year}.csv', sep=';')
        if year == 2024:
            df = df.rename({'Ladder score': 'Happiness score'}, axis=1)
        for f in ['Happiness score', 
                  'GDP per capita',
                  'Social support',
                  'Freedom to make life choices',
                  'Generosity',
                  'Perceptions of corruption']:
            df[f] = df[f].str.replace(',', '.').astype(float)
        df['Year'] = year
        dfs.append(df)
    df = pd.concat(dfs)    
    df.to_csv(OUTPUT_PATH + 'merged_happiness.csv')
    logger.info(f'CSV successfully saved.')
    return df

@task
def desc_stat(df):
    logger = get_run_logger()
    logger.info(f"Happiness score mean: {df['Happiness score'].mean()}")
    logger.info(f"Happiness score median: {df['Happiness score'].median()}")
    logger.info(f"Happiness score SD: {df['Happiness score'].std()}")

    score_by_year = df.groupby('Year')['Happiness score'].mean()
    score_by_region = df.groupby('Regional indicator')['Happiness score'].mean().sort_values(ascending=False)
    logger.info(f'Happiness score by year:\n {score_by_year}')
    logger.info(f'Happiness score by region:\n {score_by_region}')
    return score_by_region

@task
def visualization(df):
    logger = get_run_logger()
    # Histogram
    plt.figure()
    plt.hist(df['Happiness score'], edgecolor='black')
    plt.title('Distribution of happiness score across all years')
    plt.xlabel('Happiness score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'happiness_histogram.png')
    logger.info('happiness_histogram.png saved.')
    plt.close()

    # Boxplot
    plt.figure()
    sns.boxplot(df, x='Year', y='Happiness score', hue='Year')
    plt.title('Boxplot of Happiness score by year')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'happiness_by_year.png')
    logger.info('happiness_by_year.png saved.')
    plt.close()

    # Scatterplot
    plt.figure()
    sns.scatterplot(df, x='GDP per capita', y='Happiness score')
    plt.title('Scatterplot of GDP per capita vs Happiness score by year')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'gdp_vs_happiness.png')
    logger.info('gdp_vs_happiness.png saved.')
    plt.close()

    # Correlation heatmap
    plt.figure()
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, annot_kws={"size": 8})
    plt.title('Correlation heatmap')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'correlation_heatmap.png')
    logger.info('correlation_heatmap.png saved.')
    plt.close()

@task
def hyp_test(df):
    logger = get_run_logger()
    # Independent t-test
    scores = {
        2019: df.loc[df['Year'] == 2019, 'Happiness score'],
        2020: df.loc[df['Year'] == 2020, 'Happiness score']
        }
    res = stats.ttest_ind(scores[2019], scores[2020])
    logger.info(f'T-test result:')
    logger.info(f't-statistic: {res.statistic}')
    logger.info(f'p-value: {res.pvalue}')
    logger.info(f'Mean Happiness score in 2019: {scores[2019].mean()}')
    logger.info(f'Mean Happiness score in 2020: {scores[2020].mean()}')

    if res.pvalue < 0.05:
        msg = 'Mean happiness Score in 2020 is statistically significant to be different than 2019.'
    else:
        msg = 'It is not statistically conclusive to say happiness score in 2020 is different than 2019.'
    logger.info(msg)
    
    # Independent t-test between North America and ANZ and South Asia
    region_scores = {
        'NA_ANZ': df.loc[df['Regional indicator'] == 'North America and ANZ', 'Happiness score'],
        'SAsia': df.loc[df['Regional indicator'] == 'South Asia', 'Happiness score']
        }
    res = stats.ttest_ind(region_scores['NA_ANZ'], region_scores['SAsia'])
    logger.info(f'T-test result:')
    logger.info(f't-statistic: {res.statistic}')
    logger.info(f'p-value: {res.pvalue}')
    logger.info(f"Mean Happiness score in North America and ANZ: {region_scores['NA_ANZ'].mean()}")
    logger.info(f"Mean Happiness score in South Asia: {region_scores['SAsia'].mean()}")
    if res.pvalue < 0.05:
        logger.info('Mean happiness Score in North America and ANZ is statistically significant to be different than South Asia.')
    else:
        logger.info('It is not statistically conclusive to say happiness score in North America and ANZ is different than South Asia.')
    
    return msg

@task
def correlations(df):
    logger = get_run_logger()
    num_features = df.select_dtypes(include='number').drop(columns=['Happiness score']).columns
    sig_corr_before = []
    sig_corr_after = []
    for feature in num_features:
        clean_df = df[[feature, 'Happiness score']].dropna()
        res = stats.pearsonr(clean_df[feature], clean_df['Happiness score'])
        logger.info(f'Pearson Coefficient of Correlation between {feature} and Happiness score: {res.statistic}')
        logger.info(f'Pearson p-value between {feature} and Happiness score: {res.pvalue}')
        if res.pvalue < 0.05:
            sig_corr_before.append((feature, res.statistic))
        if res.pvalue < 0.05 / len(num_features):
            sig_corr_after.append((feature, res.statistic))
    logger.info(f'Significant correlation features before corrections: {[feature for feature, _ in sig_corr_before]}')
    logger.info(f'Significant correlation features after corrections: {[feature for feature, _ in sig_corr_after]}')
    return sig_corr_after

@task
def summary(df, score_by_region, msg, sig_var):
    logger = get_run_logger()
    logger.info(f"Total number of countries: {df['Country'].nunique()}")
    logger.info(f"Total number of years: {df['Year'].nunique()}")
    logger.info(f'Top 3 regions by mean happiness score:\n {score_by_region.head(3)}')
    logger.info(f'Bottom 3 regions by mean happiness score:\n {score_by_region.tail(3)}')
    logger.info(f'Result of t-test pre/post-2020: {msg}')
    if strongest_var:
        strongest_var = sorted(sig_var, key=lambda x: abs(x[1]), reverse=True)[0]
        logger.info(f'Variable most strongly correlated to Happiness score: {strongest_var[0]} with correlation coefficient {strongest_var[1]}')
    else:
        logger.info('No variable is significantly correlated to Happiness score after Bonferroni correction.')


@flow
def data_pipeline():
    df = load_df()
    score_by_region = desc_stat(df)    
    visualization(df)
    msg = hyp_test(df)
    sig_var = correlations(df)
    summary(df, score_by_region, msg, sig_var)

if __name__ == '__main__':
    data_pipeline()