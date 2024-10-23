import streamlit as st
import joblib
import pandas as pd
import os
from pathlib import Path
from utils import add_logout_button
import time

# Increase Streamlit's server timeout and session settings
st.set_page_config(
    layout='wide',
    page_title='Bulk Prediction Page',
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'prediction_state' not in st.session_state:
    st.session_state['prediction_state'] = None
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'login_timestamp' not in st.session_state:
    st.session_state['login_timestamp'] = time.time()

# Session management
def check_session():
    """Check if session is still valid"""
    if 'login_timestamp' not in st.session_state:
        return False
    
    # Reset timestamp on any activity
    st.session_state['login_timestamp'] = time.time()
    return True

# Authentication check with session management
def authenticate():
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.warning("Please log in from the home page to access this feature.")
        st.stop()
    if not check_session():
        st.warning("Session expired. Please log in again.")
        st.session_state['logged_in'] = False
        st.stop()

# Check authentication
authenticate()

# Add logout button
add_logout_button()

# Page styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .stSpinner {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Page titles
st.title('📊 Bulk Predictions Page')
st.markdown('Upload a CSV file to get churn predictions using different models.')

@st.cache_resource(show_spinner='Models Loading .....')
def load_models():
    """Load and cache the models to prevent reloading on every rerun"""
    try:
        models_dir = Path("Models Directory")
        lr = joblib.load(models_dir / "lr_model.joblib")
        xgb = joblib.load(models_dir / "xgb_model.joblib")
        return {
            "Logistic Regression": lr,
            "XGB Classifier": xgb
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def process_in_batches(df, model, batch_size=1000):
    """Process predictions in batches to prevent memory issues"""
    predictions = []
    probabilities = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        pred = model.predict(batch)
        prob = model.predict_proba(batch)[:, 1]
        predictions.extend(pred)
        probabilities.extend(prob)
        
        # Update progress
        progress = min((i + batch_size) / len(df), 1.0)
        st.progress(progress)
        
    return predictions, probabilities

@st.cache_data(ttl=3600)  # Cache for 1 hour
def predict_churn(model_name, features_dict, batch_size=1000):
    """Make predictions using the selected model with caching and batch processing"""
    try:
        # Convert features dictionary back to DataFrame
        features = pd.DataFrame(features_dict)
        
        # Get the model
        models = load_models()
        model = models[model_name]
        
        # Make predictions in batches
        prediction, probability = process_in_batches(features, model, batch_size)
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None, None

def clean_features(df):
    """Clean and prepare features for prediction"""
    try:
        df_cleaned = df.copy()
        
        if 'user_id' in df_cleaned.columns:
            df_cleaned = df_cleaned.drop('user_id', axis=1)
        
        if df_cleaned.isnull().any().any():
            st.warning("Warning: Your data contains missing values. Handle them if possible or model will handle them.")
        
        return df_cleaned
    except Exception as e:
        st.error(f"Error cleaning features: {str(e)}")
        return None

def convert_prediction_labels(predictions):
    """Convert binary predictions to descriptive labels"""
    return ['Not Churn' if pred == 0 else 'Churn' for pred in predictions]

def format_percentage(value):
    """Format float as percentage with 2 decimal places"""
    return f"{value * 100:.2f}%"

def save_prediction_state():
    """Save current prediction state to session"""
    st.session_state['prediction_state'] = {
        'completed': True,
        'timestamp': time.time()
    }

def main():
    try:
        # Load models
        models = load_models()
        if not models:
            st.error("Failed to load models. Please check if model files exist in the correct directory.")
            return

        # File uploader
        st.subheader('1. Upload your CSV file')
        uploaded_file = st.file_uploader(
            'Choose a CSV file containing the features for prediction',
            type=['csv'],
            help="Make sure your CSV contains all required features for prediction."
        )

        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Show sample of uploaded data
                st.subheader('2. Preview of uploaded data')
                st.dataframe(df.head())
                
                # Clean the features
                cleaned_df = clean_features(df)
                if cleaned_df is None:
                    return
                
                # Model selection
                st.subheader('3. Select a Model')
                model_name = st.selectbox(
                    'Choose the model for prediction',
                    list(models.keys())
                )
                
                # Add batch size selection
                batch_size = st.select_slider(
                    'Select batch size for processing',
                    options=[100, 500, 1000, 2000, 5000],
                    value=1000,
                    help="Larger batch size is faster but uses more memory"
                )
                
                if st.button('Make Predictions', type='primary'):
                    st.session_state['processing_complete'] = False
                    
                    with st.spinner('Processing predictions in batches...'):
                        # Convert DataFrame to dictionary for caching
                        features_dict = cleaned_df.to_dict()
                        
                        # Make predictions with caching and batch processing
                        prediction, probability = predict_churn(model_name, features_dict, batch_size)
                        
                        if prediction is not None and probability is not None:
                            # Convert numeric predictions to labels
                            prediction_labels = convert_prediction_labels(prediction)
                            probability_percentages = [format_percentage(p) for p in probability]
                            
                            # Create results dataframe
                            results_df = pd.DataFrame({
                                'Prediction': prediction_labels,
                                'Churn Probability': probability_percentages
                            })
                            
                            if 'user_id' in df.columns:
                                results_df['User ID'] = df['user_id']
                                results_df = results_df[['User ID', 'Prediction', 'Churn Probability']]
                            
                            # Store results in session state
                            st.session_state['results_df'] = results_df
                            st.session_state['processing_complete'] = True
                            save_prediction_state()
                
                # Show results if processing is complete
                if st.session_state.get('processing_complete', False) and st.session_state.get('results_df') is not None:
                    results_df = st.session_state['results_df']
                    
                    st.subheader('4. Prediction Results')
                    st.dataframe(results_df)
                    
                    # Download button for results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Calculate metrics
                    prediction_series = pd.Series([1 if pred == 'Churn' else 0 for pred in results_df['Prediction']])
                    total_predictions = len(prediction_series)
                    churn_count = sum(prediction_series == 1)
                    not_churn_count = sum(prediction_series == 0)
                    churn_percentage = (churn_count / total_predictions) * 100
                    not_churn_percentage = (not_churn_count / total_predictions) * 100
                    
                    # Show summary statistics
                    st.subheader('5. Summary Statistics')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Predictions", total_predictions)
                        st.metric("Predicted Churns", f"{churn_count} ({churn_percentage:.2f}%)")
                    with col2:
                        st.metric("Predicted Not Churns", f"{not_churn_count} ({not_churn_percentage:.2f}%)")
                        st.metric("Average Churn Probability", 
                                format_percentage(pd.Series([float(p.strip('%')) / 100 for p in results_df['Churn Probability']]).mean()))

                    st.info("""
                    **Prediction Labels:**
                    - 'Churn': Customer predicted to churn (1)
                    - 'Not Churn': Customer predicted to stay (0)
                    
                    The 'Churn Probability' shows the likelihood of churning (0-100% scale).
                    """)

            except Exception as e:
                st.error(f"An error occurred while processing your file: {str(e)}")
                st.info("Please make sure your CSV file is properly formatted and contains all required features.")
                
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, contact support.")

if __name__ == "__main__":
    main()