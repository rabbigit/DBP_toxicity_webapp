"""Public, aggregate usage statistics for the DBP Toxicity Predictor."""

import pandas as pd
import streamlit as st

from analytics import analytics_enabled, fetch_events


@st.cache_data(ttl=300, show_spinner="Loading current usage statistics...")
def load_events() -> pd.DataFrame:
    """Limit repeated Supabase reads while keeping the dashboard current."""
    return fetch_events()


st.title("DBP Toxicity Predictor — Usage Statistics")
st.caption(
    "This public dashboard contains aggregate usage information only. Locations "
    "are approximate, and unique-visitor counts are pseudonymous estimates. "
    "Chemical structures, uploaded files, and prediction results are not collected."
)

if not analytics_enabled():
    st.info(
        "Usage statistics will appear after Supabase analytics is configured in "
        "the Streamlit app secrets. The toxicity predictor remains available."
    )
    st.stop()

refresh_col, status_col = st.columns([1, 4], vertical_alignment="center")
with refresh_col:
    if st.button("Refresh statistics", icon=":material/refresh:"):
        load_events.clear()
with status_col:
    st.caption("Statistics are cached for five minutes to keep the app responsive.")

try:
    events = load_events()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

if events.empty:
    st.info("No analytics events have been recorded yet.")
    st.stop()

visits = events.loc[events["event_type"] == "visit"].copy()
predictions = events.loc[
    events["event_type"].isin(["single_prediction", "batch_prediction"])
].copy()

metric_columns = st.columns(5)
metric_columns[0].metric("Total visits", f"{len(visits):,}")
metric_columns[1].metric(
    "Estimated unique visitors", f"{events['visitor_hash'].nunique():,}"
)
metric_columns[2].metric("Prediction runs", f"{len(predictions):,}")
metric_columns[3].metric(
    "Compounds predicted", f"{int(predictions['item_count'].sum()):,}"
)
metric_columns[4].metric(
    "Countries", f"{visits['country_code'].dropna().nunique():,}"
)

single_runs = int((events["event_type"] == "single_prediction").sum())
batch_runs = int((events["event_type"] == "batch_prediction").sum())
latest_event = events["event_time"].max()
summary = f"Single runs: {single_runs:,} · Batch runs: {batch_runs:,}"
if pd.notna(latest_event):
    summary += f" · Latest activity: {latest_event.strftime('%B %d, %Y %H:%M UTC')}"
st.caption(summary)

trend_col, map_col = st.columns([1, 1], gap="large")

with trend_col:
    st.subheader("Visits over time")
    daily = (
        visits.dropna(subset=["event_time"])
        .assign(date=lambda frame: frame["event_time"].dt.date)
        .groupby("date")
        .size()
        .rename("Visits")
    )
    if daily.empty:
        st.info("No dated visit records are available yet.")
    else:
        st.line_chart(daily)

with map_col:
    st.subheader("Approximate global access")
    map_data = visits.dropna(subset=["latitude", "longitude"]).copy()
    if map_data.empty:
        st.info("No approximate geographic locations are available yet.")
    else:
        map_data = (
            map_data.groupby(
                ["latitude", "longitude", "country_name"], dropna=False
            )
            .size()
            .reset_index(name="visits")
        )
        map_data["marker_size"] = 75_000 * map_data["visits"].pow(0.5)
        st.map(
            map_data,
            latitude="latitude",
            longitude="longitude",
            size="marker_size",
            color="#0F766E",
        )

st.subheader("Visitors by country")
country_table = (
    visits.assign(country=visits["country_name"].fillna("Unknown"))
    .groupby("country")
    .agg(
        Visits=("session_id", "count"),
        **{"Estimated unique visitors": ("visitor_hash", "nunique")},
    )
    .sort_values("Visits", ascending=False)
    .reset_index()
    .rename(columns={"country": "Country"})
)
st.dataframe(country_table, width="stretch", hide_index=True)

st.caption(
    "Approximate locations are rounded to whole-degree coordinates. Raw IP "
    "addresses and individual visitor records are not displayed."
)
