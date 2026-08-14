# 📊 Marketing Intelligence Dashboard

A data-driven Marketing Intelligence system built using Python, Pandas, Scikit-learn and Streamlit to analyze marketing campaigns, predict performance and generate actionable recommendations.

## 🚀 Features

- 📊 Interactive marketing performance dashboard
- 🎯 Industry and platform-based campaign filtering
- 🤖 Machine Learning-based performance prediction
- 📈 Performance Score, Engagement Rate, CTR and ROAS analysis
- 💰 Revenue, Ad Spend and Leads tracking
- 📢 Content type and topic performance analysis
- 📅 Best posting day and time recommendations
- 🏆 Top-performing campaign identification
- 💡 Data-driven marketing strategy recommendations

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

## 📁 Project Structure

```text
marketing-intelligence/
│
├── data/
│   ├── marketing_data.csv
│   └── marketing_data_features.csv
│
├── models/
│   └── performance_model.pkl
│
├── src/
│   ├── app.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── generate_dataset.py
│   ├── insights.py
│   ├── model.py
│   └── recommendation.py
│
├── .gitignore
├── requirements.txt
└── README.md'

🤖 Machine Learning Model

A Scikit-learn regression pipeline is used to predict marketing campaign performance.

Model Performance
Metric	Score
MAE	2.86
RMSE	3.71
R² Score	0.9149

The trained model is saved as:

models/performance_model.pkl
📊 Dashboard

The Streamlit dashboard provides:

Campaign Filters
Industry
Platform
Marketing Recommendations
Best Content Type
Best Content Topic
Best Posting Day
Best Posting Time
Performance Metrics
Performance Score
Engagement Rate
CTR
ROAS
Business Metrics
Total Revenue
Total Ad Spend
Total Leads
Analysis
Content Type Performance
Content Topic Performance
Posting Day Performance
Posting Time Performance
ROI & ROAS Analysis
Top Performing Campaigns
🔄 Project Workflow
Raw Marketing Data
        ↓
Data Generation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
Performance Prediction
        ↓
Marketing Insights
        ↓
Campaign Recommendations
        ↓
Streamlit Dashboard
▶️ How to Run
1. Clone the repository
git clone https://github.com/anup1203/marketing-intelligence.git
cd marketing-intelligence
2. Create virtual environment
python -m venv .venv
3. Activate virtual environment

Windows PowerShell:

.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Run the dashboard
streamlit run src/app.py
🎯 Objective

The objective of this project is to transform marketing campaign data into actionable business insights by combining data analysis, machine learning and interactive visualization.

👨‍💻 Author

Anup Yadav

GitHub: anup1203

LinkedIn: Anup Yadav
