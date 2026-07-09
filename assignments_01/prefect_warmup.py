import pandas as pd
import numpy as np

from prefect import flow, task
from prefect.logging import get_run_logger

@task
def create_series(arr):
    return pd.Series(arr, name='values')

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'mode': series.mode()[0]
    }

@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    X = create_series(arr)
    X = clean_data(X)
    summary = summarize_data(X)
    logger = get_run_logger()
    logger.info(f"Summary statistics: {summary}")
    return summary


if __name__ == "__main__":
    pipeline_flow()

'''
1. The pipeline being this simple, running a python script directly is much faster than having to route through local server/database and UI interface.
2. Setting up a framework like Prefect allows us to monitor the data pipeline by decoupling it with execution. We can track whether it succeeded via the UI, and schedule jobs without changing the codes.
'''