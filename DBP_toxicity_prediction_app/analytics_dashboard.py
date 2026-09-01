"""Private usage dashboard for the DBP Toxicity Predictor."""

import hmac

import streamlit as st

from analytics import analytics_enabled, dashboard_password, fetch_events


st.set_page_config(
    page_title="DBP Predictor Usage Analytics",
    page_icon="📊",
    layout="wide",
)


def authenticate() -> bool:
    """Protect the dashboard when it is deployed with a public URL."""
    expected = dashboard_password()
    if not expected:
        st.error("Set analytics.dashboard_password in Streamlit secrets.")
        return False
    if st.session_state.get("analytics_dashboard_authenticated"):
        return True

    st.title("Usage Analytics")
    supplied = st.text_input("Dashboard password", type="password")
    if st.button("Sign in", type="primary"):
        if hmac.compare_digest(supplied, expected):
            st.session_state.analytics_dashboard_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not analytics_enabled():
    st.error("Analytics is not configured. Add the required Streamlit secrets first.")
    st.stop()

if not authenticate():
    st.stop()

st.title("DBP Toxicity Predictor — Usage Analytics")
st.caption(
    "Locations are approximate and visitor counts are pseudonymous estimates. "
    "No submitted SMILES, uploaded files, or prediction results are collected."
)

try:
    events = fetch_events()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

if events.empty:
    st.info("No analytics events have been recorded yet.")
    st.stop()

visits = events.loc[events["event_type"] == "visit"]
predictions = events.loc[
    events["event_type"].isin(["single_prediction", "batch_prediction"])
]
single_runs = int((events["event_type"] == "single_prediction").sum())
batch_runs = int((events["event_type"] == "batch_prediction").sum())

metric_columns = st.columns(5)
metric_columns[0].metric("Total visits", f"{len(visits):,}")
metric_columns[1].metric(
    "Estimated unique visitors", f"{events['visitor_hash'].nunique():,}"
)
metric_columns[2].metric("Analysis runs", f"{len(predictions):,}")
metric_columns[3].metric(
    "Compounds predicted", f"{int(predictions['item_count'].sum()):,}"
)
metric_columns[4].metric(
    "Countries", f"{visits['country_code'].dropna().nunique():,}"
)

st.caption(f"Single runs: {single_runs:,} · Batch runs: {batch_runs:,}")

st.subheader("Visits over time")
daily = (
    visits.dropna(subset=["event_time"])
    .assign(date=lambda frame: frame["event_time"].dt.date)
    .groupby("date")
    .size()
    .rename("Visits")
)
st.line_chart(daily)

st.subheader("Global access map")
map_data = visits.dropna(subset=["latitude", "longitude"]).copy()
if map_data.empty:
    st.info("No geographic locations are available yet.")
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

st.download_button(
    "Download country summary",
    data=country_table.to_csv(index=False),
    file_name="dbp_predictor_country_analytics.csv",
    mime="text/csv",
)
