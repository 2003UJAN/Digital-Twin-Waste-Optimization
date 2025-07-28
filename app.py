# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit config
st.set_page_config(page_title="Digital Twin Waste Optimization", layout="wide")

# ----------------------------
# 1. Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_waste_management_data.csv")

df = load_data()

# ----------------------------
# 2. Helper Functions
# ----------------------------

def summarize_routes(df):
    route_summary = df.groupby("collection_route").agg(
        total_households=("household_id", "count"),
        total_population=("population", "sum"),
        avg_waste_kg_per_day=("waste_gen_kg_per_day", "mean"),
        avg_recycling_rate=("recycling_rate", "mean"),
        avg_route_length_km=("route_length_km", "mean"),
        avg_route_time_hr=("route_time_hr", "mean"),
        total_distance_km=("route_length_km", "sum"),
        total_time_hr=("route_time_hr", "sum")
    ).reset_index()

    route_summary["fuel_cost_per_km"] = 0.8
    route_summary["labor_cost_per_hour"] = 15
    route_summary["maintenance_cost_per_km"] = 0.1

    route_summary["total_fuel_cost"] = route_summary["total_distance_km"] * route_summary["fuel_cost_per_km"]
    route_summary["total_labor_cost"] = route_summary["total_time_hr"] * route_summary["labor_cost_per_hour"]
    route_summary["total_maintenance_cost"] = route_summary["total_distance_km"] * route_summary["maintenance_cost_per_km"]
    route_summary["total_operational_cost"] = route_summary[["total_fuel_cost", "total_labor_cost", "total_maintenance_cost"]].sum(axis=1)

    route_summary["total_waste_kg_per_day"] = route_summary["total_households"] * route_summary["avg_waste_kg_per_day"]
    route_summary["cost_per_kg_waste"] = route_summary["total_operational_cost"] / route_summary["total_waste_kg_per_day"]

    return route_summary

def simulate_scenario(summary_df, recycling_increase=0.1, fuel_cost_multiplier=1.0):
    sim = summary_df.copy()
    sim["new_recycling_rate"] = np.minimum(sim["avg_recycling_rate"] + recycling_increase, 1.0)
    sim["adjusted_waste_kg_per_day"] = sim["total_households"] * sim["avg_waste_kg_per_day"] * (1 - sim["new_recycling_rate"])
    sim["fuel_cost_per_km"] = 0.8 * fuel_cost_multiplier
    sim["total_fuel_cost"] = sim["total_distance_km"] * sim["fuel_cost_per_km"]
    sim["total_operational_cost"] = sim["total_fuel_cost"] + sim["total_labor_cost"] + sim["total_maintenance_cost"]
    sim["cost_per_kg_waste"] = sim["total_operational_cost"] / sim["adjusted_waste_kg_per_day"]
    return sim[["collection_route", "adjusted_waste_kg_per_day", "new_recycling_rate", "total_operational_cost", "cost_per_kg_waste"]]

def calculate_smart_bin_roi(summary_df, smart_bin_cost_per_route=2000, annual_savings_factor=0.15):
    roi_df = summary_df.copy()
    roi_df["expected_annual_savings"] = roi_df["total_operational_cost"] * annual_savings_factor
    roi_df["roi_years"] = smart_bin_cost_per_route / roi_df["expected_annual_savings"]
    return roi_df[["collection_route", "total_operational_cost", "expected_annual_savings", "roi_years"]]

def calculate_carbon_emissions(summary_df, emission_per_km=2.68):
    summary_df = summary_df.copy()
    summary_df["carbon_emission_kg"] = summary_df["total_distance_km"] * emission_per_km
    return summary_df[["collection_route", "total_distance_km", "carbon_emission_kg"]]

# ----------------------------
# 3. Main Streamlit UI
# ----------------------------

st.title("🚮 Digital Twin-Enabled Waste Management Optimization")
st.markdown("This dashboard uses **generative AI-based synthetic data** to simulate and optimize urban waste management systems.")

route_summary = summarize_routes(df)

st.subheader("📊 Base Summary: Per-Route Operational Analysis")
st.dataframe(route_summary.style.format({"total_operational_cost": "${:,.2f}", "cost_per_kg_waste": "${:,.2f}"}), use_container_width=True)

# ----------------------------
# 4. Interactive Scenario Simulation
# ----------------------------
st.subheader("🔁 Scenario Simulation")

col1, col2 = st.columns(2)
with col1:
    recycle_inc_pct = st.slider("Increase in Recycling Rate (%)", 0, 50, 20)
with col2:
    fuel_multiplier = st.slider("Fuel Cost Multiplier", 0.5, 2.0, 1.0)

sim_result = simulate_scenario(route_summary, recycling_increase=recycle_inc_pct/100, fuel_cost_multiplier=fuel_multiplier)
st.dataframe(sim_result.style.format({"total_operational_cost": "${:,.2f}", "cost_per_kg_waste": "${:,.2f}"}), use_container_width=True)

# ----------------------------
# 5. Smart Bin ROI
# ----------------------------
st.subheader("💰 Smart Bin ROI Estimation")
roi = calculate_smart_bin_roi(route_summary)
st.dataframe(roi.style.format({"total_operational_cost": "${:,.2f}", "expected_annual_savings": "${:,.2f}", "roi_years": "{:.2f} years"}), use_container_width=True)

# ----------------------------
# 6. Carbon Emission Insights
# ----------------------------
st.subheader("🌱 Carbon Emissions per Route")
carbon = calculate_carbon_emissions(route_summary)
st.dataframe(carbon, use_container_width=True)

# ----------------------------
# 7. Visualization (Optional)
# ----------------------------
st.subheader("📈 Operational Cost vs Waste Generated (Bubble Plot)")
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    route_summary["total_waste_kg_per_day"],
    route_summary["total_operational_cost"],
    s=route_summary["total_households"] * 2,
    c=route_summary["avg_recycling_rate"],
    cmap="viridis",
    alpha=0.7
)
ax.set_xlabel("Total Waste Generated per Day (kg)")
ax.set_ylabel("Total Operational Cost ($)")
plt.colorbar(scatter, label="Avg Recycling Rate")
st.pyplot(fig)

