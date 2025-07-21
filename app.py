import streamlit as st
import pickle
import numpy as np

# Load model
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data['model']
le_country = data['le_country']
le_dev = data['le_dev']
le_education = data['le_education']

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #black;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

def show_predict_page():
    st.title("💼 Employee Salary Predictor")
    st.markdown("Uses Machine Learning algoritms like Regression Models to estimate a developer's salary based on survey data.")

    st.markdown("---")

    # Columns for inputs
    col1, col2 = st.columns(2)

    countries = (
        'United States of America', 'Germany', 'United Kingdom of Great Britain and Northern Ireland',
        'India', 'Canada', 'France', 'Brazil', 'Spain', 'Netherlands', 'Poland', 'Australia', 'Italy',
        'Sweden', 'Russian Federation', 'Switzerland', 'Turkey', 'Israel', 'Austria', 'Portugal',
        'Norway', 'Mexico'
    )

    dev_types = (
        'Developer, full-stack', 'Developer, front-end', 'Developer, back-end',
        'Data scientist or machine learning specialist', 'Engineer, data',
        'Developer, mobile', 'Developer, desktop or enterprise applications',
        'Engineer, site reliability', 'Other', 'Other (please specify):',
        'Developer, embedded applications or devices', 'Engineering manager',
        'DevOps specialist', 'Developer, QA or test', 'Academic researcher',
        'Data or business analyst', 'Educator', 'Senior Executive (C-Suite, VP, etc.)',
        'Developer, game or graphics', 'Cloud infrastructure engineer'
    )

    education_levels = (
        "Master's degree", "Bachelor's degree", "Less than a Bachelors", "Post grad"
    )

    with col1:
        country = st.selectbox("🌍 Country", countries)
        education = st.selectbox("🎓 Education Level", education_levels)

    with col2:
        dev_type = st.selectbox("💻 Developer Role", dev_types)
        experience = st.slider("👔 Years of Professional Experience", 0, 50, 3)

    st.markdown("---")

    # Predict button
    if st.button("🔍 Calculate Salary"):
        X = np.array([[country, dev_type, education, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_dev.transform(X[:, 1])
        X[:, 2] = le_education.transform(X[:, 2])
        X = X.astype(float)

        salary = regressor.predict(X)

        st.success("🎯 **Estimated Salary**")
        st.metric(label="💰 Annual Salary (USD)", value=f"${salary[0]:,.2f}")

    with st.expander("ℹ️ About This App"):
        st.write("""
            This app uses a machine learning model trained on the 2022 Stack Overflow Developer Survey.
            The salary is predicted based on role, experience, education, and country.
            \n⚠️ Note: The results are approximate and based on historical data and exchange rates at the time of the survey.
        """)

show_predict_page()
