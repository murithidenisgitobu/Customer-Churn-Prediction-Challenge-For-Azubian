import streamlit as st
import joblib
import pandas as pd
import os
from pathlib import Path
from utils import add_logout_button

# Set page config
st.set_page_config(
    layout='wide',
    page_title='Bulk Prediction Page',
    page_icon="📊"
)

# Check authentication
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the home page to access this feature.")
    st.stop()

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

def predict_churn(model, features):
    """Make predictions using the selected model"""
    try:
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1]
        return prediction, probability
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None, None

def clean_features(df):
    """Clean and prepare features for prediction"""
    try:
        # Make a copy to avoid modifying the original dataframe
        df_cleaned = df.copy()
        
        # Drop user_id column if it exists
        if 'user_id' in df_cleaned.columns:
            df_cleaned = df_cleaned.drop('user_id', axis=1)
        
        # Check for missing values
        if df_cleaned.isnull().any().any():
            st.warning("Warning: Your data contains missing values. Handle them if possible or model will handle them.")
        
        return df_cleaned
    except Exception as e:
        st.error(f"Error cleaning features: {str(e)}")
        return None

def convert_prediction_labels(predictions):
    """Convert binary predictions to descriptive labels"""
    # Corrected mapping: 1 -> 'Churn', 0 -> 'Not Churn'
    return ['Not Churn' if pred == 0 else 'Churn' for pred in predictions]

def format_percentage(value):
    """Format float as percentage with 2 decimal places"""
    return f"{value * 100:.2f}%"

def main():
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
            
            if st.button('Make Predictions', type='primary'):
                with st.spinner('Making predictions...'):
                    # Make predictions
                    prediction, probability = predict_churn(models[model_name], cleaned_df)
                    
                    if prediction is not None and probability is not None:
                        # Convert numeric predictions to labels
                        prediction_labels = convert_prediction_labels(prediction)
                        
                        # Format probabilities as percentages
                        probability_percentages = [format_percentage(p) for p in probability]
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Prediction': prediction_labels,
                            'Churn Probability': probability_percentages
                        })
                        
                        # Add original index if it exists
                        if 'user_id' in df.columns:
                            results_df['User ID'] = df['user_id']
                            # Reorder columns to put User ID first
                            results_df = results_df[['User ID', 'Prediction', 'Churn Probability']]
                        
                        # Show results
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
                        
                        # Calculate percentages for metrics
                        total_predictions = len(prediction)
                        churn_count = sum(prediction == 1)  # Changed from 0 to 1
                        not_churn_count = sum(prediction == 0)  # Changed from 1 to 0
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
                            st.metric("Average Churn Probability", format_percentage(probability.mean()))

                        # Add color coding explanation
                        st.info("""
                        **Prediction Labels:**
                        - 'Churn': Customer predicted to churn (1)
                        - 'Not Churn': Customer predicted to stay (0)
                        
                        The 'Churn Probability' shows the likelihood of churning (0-100% scale).
                        """)

        except Exception as e:
            st.error(f"An error occurred while processing your file: {str(e)}")
            st.info("Please make sure your CSV file is properly formatted and contains all required features.")

if __name__ == "__main__":
    main()