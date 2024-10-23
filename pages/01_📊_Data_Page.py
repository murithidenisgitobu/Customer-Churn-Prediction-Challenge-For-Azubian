# pages/1_Data.py
import streamlit as st
import pandas as pd
from utils import add_logout_button

st.set_page_config(layout='wide',
                   page_icon='🔂', 
                   page_title='Data Page')
# Check authentication
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the home page to access this feature.")
    st.stop()

# Add logout button
add_logout_button()

st.title('Data Page')
st.markdown('200k Rows of the data used to train the models')
st.write('To access the complete training data use the link below:')
st.success('https://drive.google.com/file/d/1e9LdqRG_L4sXseQQrqFGDaZUlmX49XGj/view?usp=drive_link')

# Function to load data

@st.cache_data(show_spinner='Data Loading .....')
def load_data():
    return pd.read_csv('Data\Train_200k.csv')

# Load and display data
train_df = load_data()
st.dataframe(train_df.head(950000))

# Download button
csv = train_df.to_csv(index=False)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='train_data.csv',
    mime='text/csv',
)