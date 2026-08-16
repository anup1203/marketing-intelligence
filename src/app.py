import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG & THEME STYLING
# ============================================================

st.set_page_config(
    page_title="Marketing Intelligence",
    layout="wide"
)

# Custom color palette (Indigo/Slate Modern Theme)
st.markdown("""
<style>
    /* Dark Base Background */
    .stApp {
        background-color: #0b0b0b !important;
    }
    
    /* Sidebar - Pure Dark */
    section[data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #27272a !important;
    }
    
    /* Metric Cards - Minimal Red/White */
    div[data-testid="stMetric"] {
        background-color: #161619 !important;
        border: 1px solid #2e2e2e !important;
        border-left: 3px solid #ffffff !important;
        padding: 14px 18px !important;
        border-radius: 6px !important;
    }
    
    /* Metric Labels (Subtitle/Category) */
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] p {
        color: #a1a1aa !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.3px;
    }
    
    /* Metric Values (Main Numbers/Text) */
    div[data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Headings - Crisp White */
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f4f4f5 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/marketing_data_features.csv")


df = load_data()


@st.cache_resource
def load_model():
    return joblib.load("models/performance_model.pkl")


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("Marketing Intelligence Dashboard")

st.write(
    "Data-driven analysis and campaign recommendations "
    "for better marketing performance."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Campaign Filters")

industry = st.sidebar.selectbox(
    "Select Industry",
    sorted(df["Industry"].unique())
)

platform = st.sidebar.selectbox(
    "Select Platform",
    sorted(df["Platform"].unique())
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    (df["Industry"] == industry) &
    (df["Platform"] == platform)
]


if filtered_df.empty:
    st.warning("No data available for this selection.")

else:

    # ========================================================
    # RECOMMENDATION LOGIC
    # ========================================================

    best_content = (
        filtered_df.groupby("Content_Type")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    best_topic = (
        filtered_df.groupby("Content_Topic")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    best_day = (
        filtered_df.groupby("Posting_Day")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    best_time = (
        filtered_df.groupby("Posting_Time")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    best_content_type = best_content.index[0]
    best_content_topic = best_topic.index[0]
    best_posting_day = best_day.index[0]
    best_posting_time = best_time.index[0]


    # ========================================================
    # KPI VALUES
    # ========================================================

    avg_performance = filtered_df["Performance_Score"].mean()
    avg_engagement = filtered_df["Engagement_Rate"].mean()
    avg_ctr = filtered_df["CTR"].mean()
    avg_roas = filtered_df["ROAS"].mean()

    total_revenue = filtered_df["Revenue"].sum()
    total_spend = filtered_df["Ad_Spend"].sum()
    total_leads = filtered_df["Leads"].sum()


    # ========================================================
    # RECOMMENDED STRATEGY
    # ========================================================

    st.subheader("Recommended Marketing Strategy")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Content Type", best_content_type)

    with col2:
        st.metric("Content Topic", best_content_topic)

    with col3:
        st.metric("Best Day", best_posting_day)

    with col4:
        st.metric("Best Time", best_posting_time)


    # ========================================================
    # ML PERFORMANCE PREDICTION
    # ========================================================

    st.subheader("ML Performance Prediction")

    prediction_data = filtered_df.copy()

    if not prediction_data.empty:
        prediction_row = prediction_data.iloc[0:1].copy()
        predicted_score = model.predict(prediction_row)[0]

        st.metric(
            "Predicted Performance Score",
            f"{predicted_score:.2f}"
        )

        if predicted_score >= 75:
            st.success("High-performing campaign predicted!")

        elif predicted_score >= 65:
            st.info("Average-to-good campaign performance predicted.")

        else:
            st.warning("Low campaign performance predicted.")


    # ========================================================
    # KPI SECTION
    # ========================================================

    st.subheader("Campaign Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Performance Score", f"{avg_performance:.2f}")

    with col2:
        st.metric("Engagement Rate", f"{avg_engagement:.2f}%")

    with col3:
        st.metric("CTR", f"{avg_ctr:.2f}%")

    with col4:
        st.metric("ROAS", f"{avg_roas:.2f}")


    # ========================================================
    # BUSINESS KPIs
    # ========================================================

    st.subheader("Business Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")

    with col2:
        st.metric("Total Ad Spend", f"₹{total_spend:,.0f}")

    with col3:
        st.metric("Total Leads", f"{total_leads:,}")


    # ========================================================
    # CONTENT TYPE ANALYSIS
    # ========================================================

    st.subheader("Content Type Performance")

    content_chart = (
        filtered_df.groupby("Content_Type")
        .agg(
            Performance_Score=("Performance_Score", "mean"),
            Engagement_Rate=("Engagement_Rate", "mean"),
            CTR=("CTR", "mean"),
            ROAS=("ROAS", "mean")
        )
        .sort_values("Performance_Score", ascending=False)
    )

    st.bar_chart(content_chart["Performance_Score"])

    st.dataframe(
        content_chart.round(2),
        use_container_width=True
    )


    # ========================================================
    # TOPIC ANALYSIS
    # ========================================================

    st.subheader("Content Topic Performance")

    topic_chart = (
        filtered_df.groupby("Content_Topic")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(topic_chart)


    # ========================================================
    # POSTING DAY ANALYSIS
    # ========================================================

    st.subheader("Posting Day Performance")

    day_chart = (
        filtered_df.groupby("Posting_Day")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(day_chart)


    # ========================================================
    # POSTING TIME ANALYSIS
    # ========================================================

    st.subheader("Posting Time Performance")

    time_chart = (
        filtered_df.groupby("Posting_Time")["Performance_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(time_chart)


    # ========================================================
    # ROI / ROAS ANALYSIS
    # ========================================================

    st.subheader("ROI & ROAS Analysis")

    roi_analysis = (
        filtered_df.groupby("Content_Type")
        .agg(
            ROI=("ROI", "mean"),
            ROAS=("ROAS", "mean"),
            Revenue=("Revenue", "mean"),
            Ad_Spend=("Ad_Spend", "mean")
        )
        .sort_values("ROAS", ascending=False)
    )

    st.dataframe(
        roi_analysis.round(2),
        use_container_width=True
    )


    # ========================================================
    # TOP PERFORMING CAMPAIGNS
    # ========================================================

    st.subheader("Top Performing Campaigns")

    top_campaigns = (
        filtered_df.sort_values("Performance_Score", ascending=False)
        [
            [
                "Client",
                "Platform",
                "Content_Type",
                "Content_Topic",
                "Performance_Score",
                "Engagement_Rate",
                "CTR",
                "ROI",
                "ROAS"
            ]
        ]
        .head(10)
    )

    st.dataframe(
        top_campaigns.round(2),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    with st.expander("View Filtered Dataset"):
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Marketing Intelligence System | "
    "Built with Python, Pandas and Streamlit"
)
