# Video link: https://youtu.be/9GkYN_Jvoyc

import json
import joblib
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# --- Step 1: Incremental Read ---
def load_meta(file_path: str='assignments_10/models/weather_classifier_metadata.json'):
    with open(file_path, "r") as f:
        meta_data = json.load(f)
    return meta_data

def get_client():
    load_dotenv()
    try:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
    except KeyError as e:
        raise RuntimeError(f"Missing required Supabase credential: {e}")
    supabase = create_client(url, key)
    return supabase

def fetch_raw(supabase):
    response = (supabase
               .table("weather_raw")
               .select("*")
               .execute())
    raw_rows = response.data
    print(f"Fetched {len(raw_rows)} rows from weather_raw")
    return response.data

def fetch_to_classify(supabase, raw_rows):
    enriched_response = supabase.table("weather_enriched").select("date").execute()
    already_done = {row["date"] for row in enriched_response.data}
    to_classify = [row for row in raw_rows if row["date"] not in already_done]
    print(f"Records to classify: {len(to_classify)} (skipping {len(already_done)} already enriched)")
    return to_classify

# --- Step 2: ML Transform ---
def load_model(path: str = "assignments_10/models/weather_classifier.pkl"):
    clf = joblib.load(path)
    return clf

def load_df(data: list, features: list) -> pd.DataFrame:
    if not data:
        return None
    df = pd.DataFrame(data)
    return df[features]

def predict(model, X: pd.DataFrame):
    if not isinstance(X, pd.DataFrame):
        return None, None
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)[:,1]

    print(f"Good days predicted: {prediction.sum()} / {len(prediction)}")
    print(f"Confidence range: {probabilities.min():.2f} - {probabilities.max():.2f}")
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

# --- Step 3: LLM Transform ---
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
    text = text.strip()
    if not text:
        return None
    # Reject if more than two sentences (simple heuristic)
    sentences = [s for s in text.split(".") if s.strip()]
    if len(sentences) > 2:
        return None
    return text

def make_llm_summary(enrichment_records, to_classify):
    SYSTEM_PROMPT = (
        "You are writing a one-sentence running recommendation for a daily weather summary app. "
        "You will receive weather conditions for a single day and a machine learning prediction "
        "about whether the day is good for running. "
        "Write exactly one sentence — direct, practical, and specific to the conditions. "
        "Do not use bullet points, headers, or phrases like 'Based on the data'."
    )
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for i, record in enumerate(enrichment_records):
        raw_row = next(r for r in to_classify if r["date"] == record["date"])

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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
            print(f"API error on {record["date"]}: {e}")
            summary = "Recommendation unavailable."
        record["llm_summary"] = summary

        if (i + 1) % 50 == 0:
            print(f"  Enriched {i + 1} / {len(enrichment_records)} records...")

        # if i > 10:  # For testing
        #     break

    return enrichment_records

# --- Step 4: Load ---
def upsert_records(supabase, enrichment_records):
    if not enrichment_records:
        print(f"No record to upsert.")
        return
    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_enriched")

# --- Step 5: Verify ---
def verify(supabase):
    response = (
        supabase.table("weather_enriched")
        .select("*")
        .execute()
    )
    records = response.data
    features = ['date', 'good_for_running', 'confidence', 'llm_summary']
    print(f"Total number of rows: {len(records)}")
    print(f"Sample rows:")
    for i in range(5):
        print(f"Row {i + 1}: { {k: v for k, v in records[i].items() if k in features} }")
    print(f"Number of days good for running: {sum([r['good_for_running'] for r in records])}")



if __name__ == "__main__":
    # --- Step 1: Incremental Read ---
    meta_data = load_meta()
    supabase = get_client()
    raw_rows = fetch_raw(supabase)
    to_classify = fetch_to_classify(supabase, raw_rows)

    # --- Step 2: ML Transform ---
    clf = load_model()
    df = load_df(to_classify, meta_data['features'])
    prediction, probabilities = predict(clf, df)
    enrichment_records = enrich_records(to_classify, prediction, probabilities)

    # # Sanity check from lesson
    # good_days = [r for r in enrichment_records if r["good_for_running"]]
    # skip_days = [r for r in enrichment_records if not r["good_for_running"]]

    # print(f"Good days: {len(good_days)} ({len(good_days)/len(enrichment_records):.0%})")
    # print(f"Skip days: {len(skip_days)}")

    # # Show a few high-confidence and borderline predictions
    # enrichment_records.sort(key=lambda r: r["confidence"], reverse=True)
    # print("\nHighest confidence (good for running):")
    # for r in enrichment_records[:3]:
    #     print(f"  {r['date']}: {r['confidence']:.3f}")

    # enrichment_records.sort(key=lambda r: abs(r["confidence"] - 0.5))
    # print("\nMost borderline (closest to 0.5 confidence):")
    # for r in enrichment_records[:3]:
    #     print(f"  {r['date']}: {r['confidence']:.3f}")
    
    # --- Step 3: LLM Transform ---
    enrichment_records = make_llm_summary(enrichment_records, to_classify)

    # --- Step 4: Load ---
    upsert_records(supabase, enrichment_records)

    # --- Step 5: Verify ---
    verify(supabase)
    # Most of the summaries reflects the weather features and the model's prediction accurately, but it struggle more when the cofidence is low. 
    # A particularly good one would be "With a high of 25.3°C, low of 18.7°C, minimal precipitation, and light winds, today is an excellent day for running." on 2023-09-17,
    # which gives concrete values and correct analysis. A particularly bad one is "Despite favorable temperatures and no precipitation, the moderate wind speed suggests today may not be ideal for running."
    # on 2023-04-27, as the summary and the choice appears to be contradicting. All the descriptions are saying it should be a good running day, but the result is not.
    # The reason could be this is one of those misclassified case, as the prediction model does not generate 100% accuracy, there could be times where all the weather data seems
    # to be good for running, but the classification says otherwise. In that case, the LLM cannot make sense of the result no matter how hard it tries. Also, this particular example
    # also only has a confidence of 0.4987, which is highly likely that it got misclassified.

    # --- Step 6: Reflect ---
    # 1. Although my data was trained on the same city I chose for this project (Los Angeles) instead of Charlotte, NC, if I were to load another city data, the prediction will not be accurate at all.
    # This is due to the condition on whether a day is good for running or not is linked to that specific city, like 30C in LA and 30C in Thailand feels completely different. Hence, it will not 
    # provide an accurate prediction.
    # 2. The LLM recommendations is purely additive, it cannot "override" the classifier, as it is just generating a summary of why the day is having the prediction, the prediction is already there.
    # This implies the LLM is not involved in the prediction process, which means the prediction is decisive, in contrary to the probabilistic nature of an LLM.
    # 3. The main concern will be the latency. With the cost analysis in the lesson, for 50,000 records it costs roughtly $2.50, which is minimal. However, just running 365 records took a few minutes,
    # so running 50,000 record will cost hours and even days. One way that I can address it is to run the LLM part parallelly, where I can use other services like Google Colab to run the LLM API at the 
    # same time, hence reducing the total time required.
