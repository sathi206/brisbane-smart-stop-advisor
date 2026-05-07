# Brisbane CBD Smart Stop Advisor

Helps drivers find the best legal parking near their destination in Brisbane CBD using machine learning and time-based rules.

## What it does
- Predicts parking availability using Random Forest trained on 2.8 million records
- Checks if 2-minute loading zones are legal at your arrival time
- Flags bus stops as avoid zones
- Shows results on an interactive map

## Tech Stack
Python, pandas, scikit-learn, Streamlit, Folium

## Model Performance
73% accuracy on 570,000 test records — improved from 45% by adding zone-level historical averages

## Data Sources
Brisbane City Council Open Data Portal

## How to Run
pip install streamlit pandas scikit-learn folium
streamlit run app.py

## Author
Data Science student at QUT Brisbane
