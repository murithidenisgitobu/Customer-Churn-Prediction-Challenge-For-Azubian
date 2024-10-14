# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from streamlit_extras.colored_header import colored_header

# Set page config
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

# Define hardcoded credentials
username = "admin"
password = "password"

# Initialize session state variables
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = False
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Login form
if not st.session_state['logged_in']:
    st.title("Customer Churn Prediction App")
    st.subheader("Login")

    user_input = st.text_input("Username")
    pass_input = st.text_input("Password", type='password')

    if st.button("Login"):
        if user_input == username and pass_input == password:
            st.session_state['authentication_status'] = True
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password. Please try again.")
else:
    # If logged in, display the dashboard
    st.title("Customer Churn Prediction Dashboard")
    colored_header(
        label="Customer Churn Prediction Dashboard",
        description="Predict the likelihood of a customer churning and plan retention strategies",
        color_name="blue-70"
    )

    # About the app
    st.markdown("""## About This App
    
    This application uses machine learning models to predict customer churn in a telecommunications company. 
    We employ two powerful algorithms:
    
    1. **XGBoost**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.
    2. **Logistic Regression**: A statistical method for predicting binary outcomes.
    
    Our models analyze various customer attributes to determine the likelihood of churn, 
                helping the company take proactive measures to retain customers.
""")

    # Create three columns for 3D visualizations
    st.title("Application Attributes")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Plan Retention Strategies Early")
        
        # Create 3D scatter plot
        x = np.random.randn(100)
        y = np.random.randn(100)
        z = np.random.randn(100)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=z,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Service Usage',
                yaxis_title='Contract Length',
                zaxis_title='Monthly Charges'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Predict Churn and Churn Probability")
        
        # Create 3D surface plot
        x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
        y = x.copy().T
        z = np.cos(x ** 2 + y ** 2)
        
        fig = go.Figure(data=[go.Surface(z=z, colorscale='RdBu')])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Churn Probability'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Choose between Two Models")
        
        # Create 3D line plot
        t = np.linspace(0, 10, 100)
        x = np.cos(t)
        y = np.sin(t)
        z = t
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                width=5,
                color=z,
                colorscale='Jet'
            )
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Precision',
                yaxis_title='Recall',
                zaxis_title='F1 Score'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Add some custom CSS to make the page more colorful
    st.markdown("""<style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        </style>""", unsafe_allow_html=True)
