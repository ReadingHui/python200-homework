# --- Supabase Connection ---
# Q1
# The two piece of information are the project url and the anon API key. I found them at the `Project Overview`, under the project name, with the `Copy` button.
# They should never be hardcoded in a Python script because when published, bots can scrap them within seconds and have access to the project as if they were you,
# causing major security breach.

# Q2
import os
from dotenv import load_dotenv
from supabase import create_client
def get_client():
    load_dotenv()
    try:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
    except KeyError as e:
        raise RuntimeError(f"Missing required environment variable: {e}")
    supabase = create_client(url, key)
    return supabase

# Q3
# Row Level Security (RLS) is a security measure implemented on row level, like restricting user can only read their own rows. We are disabling them because it adds complexity during development.
# Some real-world application would be better to keep it enabled, like writing and reading bank transaction record, user should not be permitted to read and alter other users record but their own.

# --- supabase-py CRUD ---
# Q1
from datetime import datetime
def insert_test_record(supabase):
    today = str(datetime.now().date())
    row = {
        "date": today,
        "temperature_2m_max": 30.9,
        "temperature_2m_min": 23.3,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 21.1,
    }
    supabase.table("weather_raw").upsert(row, on_conflict="date").execute()

# If the function was written using .insert() instead of .upsert(), running it twice will generate an error
# because there is already an entry with the same primary key `date`. Hence I used .upsert() instead, this 
# function checks if the primary key already exists, if yes, it will just update the entry instead of attempting
# to create a new one and raise an error.

# Q2
def get_record_by_date_range(supabase, start, end):
    response = (supabase
               .table("weather_raw")
               .select("*")
               .gte("date", start)
               .lte("date", end)
               .execute())
    return response.data

# Q3
# insert() insert a new entry into the table, if there is already an entry with the same primary key exists, it will raise an error.
# upsert() updates the row if an entry with the same primary key exists, insert if there is none.
# I would choose insert() if I want to prevent accidental data overwrites, like if my machine did not sync to the new date, I don't want
# script to overwrite the weather of the previous date, it should throw an error instead.
# I would use upsert() when I want to sync data, say a user's profile, we want it to be the most updated, and it has low chance of having
# the same primary key (unless the user has been hacked or major bug in code).
def safe_upsert(supabase, records):
    supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    if isinstance(records, dict):
        print("There is 1 row affected.")
    else:
        print(f"There are {len(records)} rows affected.")

# --- Idempotency ---
# Idempotency is crucial in a data pipeline because the data are being written/read automatically in a large quantity.
# If the result is not the same each time, we will never know if the data is being correctly written or read, this could
# cause the data pipeline to break, literally with errors or crashes, or causing incorrect data to be processed. For example,
# if a script crashes halfway through and restarted with a non-idempotent pipeline, say it didn't update whether a credit card
# transaction has been charged before it was crashed, when it reboot, the credit card will be double charged.

if __name__ == "__main__":
    supabase = get_client()
    insert_test_record(supabase)
    start = "2026-08-26"
    end = "2026-08-28"
    print(f"Records between {start} and {end}:")
    print(get_record_by_date_range(supabase, start, end))
    print()
