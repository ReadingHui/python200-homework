from prefect import task
from prefect.logging import get_run_logger

# --- Prefect Orchestration ---
# Q1
# @task defines a task in the whole pipeline, while @flow orchestrate the whole pipeline. The helper function will not be decorated with @task,
# as there is no I/O, there is no need to log any of the run.



# Q2
@task(retries=2, retry_delay_seconds=30)
def call_api():
    ...
    return

# Q3
# In Runs tab, click on the "Task runs" and look for the transform task, click on it to look at the failed logs. I would be looking for the failed log, and see
# what caused the failed run of the transform task. The load_enriched not running means there is a crashing error happend in transform step.

# --- Production Patterns ---
# Q1
# raise_for_status() makes the prefect flow stop and mark the request step as failed, logging the error and stop downstream steps running. This is better than the
# `if response.status_code != 200: print("error")` because it also catches the other error like 404 or 500, which allow the pipeline continues with empty or malformed responses.
# This leads to corrupted downstream data, misleading results, or a pipeline that appears successful even though the data is wrong.

# Q2
# `upsert` protects the database from having duplicated entry and crash, as it only updates the existing records instead of inserting them, and there will be duplicated `date` 
# entries as we are rerunning from the beginning. Using `insert` will cause an erro on duplicated primary key and crash and fail the task.

# Q3
@task
def upsert_records(enrichment_records: list):
    logger = get_run_logger()
    ...
    logger.info(f"{len(enrichment_records)} enrichment records were upserted.")

# Q4
# The incremental processing check in the transform task make the task idempotent as it allows the task to pick up from where it left off last time, preventing it re-run the data
# it already did, while still producing the full transformed data after the task everytime. If it is removed and it ran the ML and LLM steps on all records every time, the cost and
# time will sky-rocket, as every processed data will be processed again. It will also increase the chance of breaking and producing incorrect data, as running the task completely means
# each data will be processed multiple times, hence a higher chance of running into an error.