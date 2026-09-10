import time
from prefect import task
from prefect.logging import get_run_logger

# --- Prefect Orchestration ---
# Q1
# @task defines a task in the whole pipeline, while @flow orchestrate the whole pipeline. The helper function will not be decorated with @task,
# as there is no I/O, it doesn't touch an external system. There is no orchestration benefit, no retry lofic, no independent observability need 
# and no concurrency benefit. For example, if the temperature helper function also fetches the temperature from the Open-Mateo API, I would promote
# it to a task.



# Q2
@task(name="call_api", retries=3, retry_delay_seconds=30)

# Q3
# In Runs tab, click on the "Task runs" and look for the transform task, click on it to look at the failed logs. I would be looking for the failed log, and see
# what caused the failed run of the transform task. The load_enriched not running means there is a crashing error happend in transform step, it can be seen from
# the timeline dashboard as well.

# --- Production Patterns ---
# Q1
# raise_for_status() makes the prefect flow stop and mark the request step as failed by raising an HTTPError on non-2xx responses, logging the error and stop downstream 
# steps running. This is better than the `if response.status_code != 200: print("error")` because it also catches the other error like 404 or 500, which allow the pipeline 
# continues with empty or malformed responses, leading to corrupted downstream data, misleading results, or a pipeline that appears successful even though the data is wrong.
# More importantly, raising an error stops the task in Prefect and prevents downstream execution, comparing to the `if` clause where the corrupted data flow downstream
# silently.

# Q2
# `upsert` protects the database from having duplicated entry and crash, as it only updates the existing records instead of inserting them, and there will be duplicated `date` 
# entries as we are rerunning from the beginning, allowing a safe re-run and creating an idempotent code. 
# Using `insert` will cause a unique-constraint violation on duplicated primary key and crash and fail the task.

# Q3
@task
def load_enriched(enrichment_records: list):
    logger = get_run_logger()
    logger.info(f"Number of records pending upsert: {len(enrichment_records)}")
    start = time.perf_counter()
    # --- Upsert codes ---
    elapsed = time.perf_counter() - start
    logger.info(f"Upserted {len(enrichment_records)} enrichment records in {elapsed:.2f}s.")    # The len(enrichment_records) should be len(response.data) from response, but this is just a stub for now

# Q4
# The incremental processing check in the transform task make the task idempotent as it allows the task to pick up from where it left off last time, preventing it re-run the data
# it already did, while still producing the full transformed data after the task everytime. If it is removed and it ran the ML and LLM steps on all records every time, the cost and
# time will sky-rocket, as every processed data will be processed again. It will also increase the chance of breaking and producing incorrect data, as running the task completely means
# each data will be processed multiple times, it unnecessarily prolonged the runtime.