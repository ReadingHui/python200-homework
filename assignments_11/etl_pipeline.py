import os
import requests
import json
import pandas as pd
import joblib

from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from prefect import task, flow
from prefect.logging import get_run_logger

# Loading configs
with open("config.json") as f:
    configs = json.load(f)
OPEN_MATEO_URL = configs['OPEN_MATEO_URL']
OPEN_MATEO_PARAMS = configs['OPEN_MATEO_PARAMS']
OPEN_MATEO_FEATURES = configs['OPEN_MATEO_FEATURES']

MODEL_DIR = configs['MODEL_DIR']


# `extract` task
@task(retries=2, retry_delay_seconds=10)
def extract():
    response = requests.get(OPEN_MATEO_URL, params=OPEN_MATEO_PARAMS)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["daily"])
    df["date"] = pd.to_datetime(df["time"]).astype(str)
    df = df.drop("time", axis=1)
    df = df[OPEN_MATEO_FEATURES] # Put the features in correct order
    row_dicts = df.to_dict(orient="records")
    print(f"Raw data extracted. Total of {len(row_dicts)} rows.")
    return row_dicts

# `load_raw` task
def get_supabase_client():
    load_dotenv()
    try:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
    except KeyError as e:
        raise RuntimeError(f"Missing required Supabase credential: {e}")
    supabase = create_client(url, key)
    return supabase

@task(retries=2, retry_delay_seconds=5)
def load_raw(records):
    supabase = get_supabase_client()
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Upserted {len(response.data)} rows.")
    return 

# `transform` task
def incremental_check(supabase, records):
    enriched_response = supabase.table("weather_enriched").select("date").execute()
    already_done = {row["date"] for row in enriched_response.data}
    to_classify = [row for row in records if row["date"] not in already_done]
    print(f"Records to classify: {len(to_classify)} (skipping {len(already_done)} already enriched)")
    return to_classify

def load_model(model_file: str="weather_classifier.pkl", feature_json: str="weather_classifier_metadata.json"):
    # Loading the saved sklearn Pipeline
    clf = joblib.load(os.path.join(MODEL_DIR, model_file))

    # Loads feature names
    with open(os.path.join(MODEL_DIR, feature_json)) as f:
        features = json.load(f)['features']
    return clf, features

def predict(clf, records, features):
    df = pd.DataFrame(records)[features]
    prediction = clf.predict(df)
    probabilities = clf.predict_proba(df)[:, 1]
    return prediction, probabilities


def enrich_records(to_classify, prediction, probabilities):
    enrichment_records = []
    for i, row in enumerate(to_classify):
        enrichment_records.append(
            {
                "date":             row["date"],
                "good_for_running": bool(prediction[i]),
                "confidence":       round(float(probabilities[i]), 4)
            }
        )
    return enrichment_records

def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )

def validate_summary(text):
    text = text.strip() + " " # Added this to avoid accidentally voiding summary with temperature like "27.5C"
    if not text:
        return None
    # Reject if more than one sentence (simple heuristic)
    sentences = [s for s in text.split(". ") if s.strip()]
    if len(sentences) != 1:
        return None
    return text

@task
def transform(records):
    supabase = get_supabase_client()

    # Incremental check
    to_classify = incremental_check(supabase, records)
    if not to_classify:
        print("No new rows to be enriched.")
        return None

    # Load model and features
    clf, features = load_model()

    # Running `predict` and `predict_proba`
    prediction, probabilities = predict(clf, pd.DataFrame(to_classify), features)

    # Make enrichment records
    enrichment_records = enrich_records(to_classify, prediction, probabilities)

    # Calls the OpenAI API to generate a one-sentence recommendation for each record
    system_prompt = (
            "You are writing a one-sentence running recommendation for a daily weather summary app. "
            "You will receive weather conditions for a single day and a machine learning prediction "
            "about whether the day is good for running. "
            "Write exactly one sentence — direct, practical, and specific to the conditions. "
            "Do not use bullet points, headers, or phrases like 'Based on the data'."
        )
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for i, record in enumerate(enrichment_records):
        if i == 0:
            print(f"  Enrichment initiating...")
        raw_row = next(r for r in to_classify if r['date'] == record['date'])
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": make_user_message(
                            raw_row,
                            record["good_for_running"],
                            record["confidence"],
                        ),
                    },
                ],
                max_tokens=100,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error on {record['date']}: {e}")
            summary = "Recommendation unavailable."
        if not validate_summary(summary):
            print(f"Summary validation failed on {record['date']}")
            print(summary)
            summary = "Recommendation unavailable."
        record["llm_summary"] = summary

        if (i + 1) % 50 == 0:
            print(f"  Enriched {i + 1} / {len(enrichment_records)} records...")

        # if i > 10:
        #     break

    return enrichment_records

# `load_enriched` task
@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records):
    if not enrichment_records:
        print("There is nothing to upsert.")
        return
    supabase = get_supabase_client()
    response = supabase.table("weather_enriched").upsert(enrichment_records, on_conflict="date").execute()
    print(f"There are {len(response.data)} row upserted.")

@flow(log_prints=True)
def etl_pipeline():
    raw_records = extract()
    load_raw(raw_records)
    enrichment_records = transform(raw_records)
    load_enriched(enrichment_records)
    print(f"ETL pipeline completed.")

if __name__ == "__main__":
    etl_pipeline()