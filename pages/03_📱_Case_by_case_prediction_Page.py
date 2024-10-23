import streamlit as st
import joblib
import pandas as pd
import numpy as np
from utils import add_logout_button


# Set page config
st.set_page_config(
    layout="wide",
    page_title="Telecom Churn Prediction",
    page_icon="📱",
)


# Check authentication
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the home page to access this feature.")
    st.stop()

# Add logout button
add_logout_button()


# Define categorical options
REGION_OPTIONS = ['Unknown'] + [
    'DAKAR', 'SAINT-LOUIS', 'THIES', 'LOUGA', 'MATAM', 'FATICK', 'KAOLACK',
    'DIOURBEL', 'TAMBACOUNDA', 'ZIGUINCHOR', 'KOLDA', 'KAFFRINE', 'SEDHIOU',
    'KEDOUGOU'
]

TENURE_OPTIONS = ['Unknown'] + [
    'K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month',
    'I 18-21 month', 'J 21-24 month', 'F 9-12 month', 'D 3-6 month'
]

TOP_PACK_OPTIONS = ['Unknown'] + [
    'On net 200F=Unlimited _call24H', 'Data:490F=1GB,7d', 'All-net 500F=2000F;5d',
    'On-net 500=4000,10d', 'Data:3000F=10GB,30d', 'Data:200F=Unlimited,24H',
    'IVR Echat_Daily_50F', 'Data:1000F=2GB,30d', 'Mixt 250F=Unlimited_call24H',
    'On-net 1000F=10MilF;10d', 'MIXT:500F= 2500F on net _2500F off net;2d',
    'Data: 200 F=100MB,24H', 'All-net 600F= 3000F ;5d', 'On-net 200F=60mn;1d',
    'Twter_U2opia_Daily', 'Data: 100 F=40MB,24H', 'All-net 500F =2000F_AllNet_Unlimited',
    'On net 200F= 3000F_10Mo ;24H', '200=Unlimited1Day', 'Jokko_Daily',
    'Data:1000F=5GB,7d', 'Data:700F=1.5GB,7d', 'All-net 1000=5000;5d',
    'Data:150F=SPPackage1,24H', 'IVR Echat_Monthly_500F', 'VAS(IVR_Radio_Daily)',
    'MIXT: 390F=04HOn-net_400SMS_400 Mo;4h', 'MIXT: 200mnoff net _unl on net _5Go;30d',
    'On-net 500F_FNF;3d', 'MIXT: 590F=02H_On-net_200SMS_200 Mo;24h', 'Data:1500F=3GB,30D',
    'Data:300F=100MB,2d', 'Data:500F=2GB,24H', 'Data: 490F=Night,00H-08H',
    'All-net 1000F=(3000F On+3000F Off);5d', 'New_YAKALMA_4_ALL',
    'MIXT:10000F=10hAllnet_3Go_1h_Zone3;30d', 'Yewouleen_PKG', 'Data:1500F=SPPackage1,30d',
    'WIFI_Family_2MBPS', 'All-net 500F=1250F_AllNet_1250_Onnet;48h', 'On-net 300F=1800F;3d',
    'Twter_U2opia_Weekly', 'Data:50F=30MB_24H',
    'MIXT:1000F=4250 Off net _ 4250F On net _100Mo; 5d', 'WIFI_ Family _4MBPS',
    'Data:700F=SPPackage1,7d', 'Jokko_promo', 'CVM_on-net bundle 500=5000',
    'Pilot_Youth4_490', 'All-net 300=600;2d', 'Twter_U2opia_Monthly',
    'IVR Echat_Weekly_200F', 'TelmunCRBT_daily', 'MROMO_TIMWES_RENEW',
    'MIXT: 500F=75(SMS, ONNET, Mo)_1000FAllNet;24h', 'Pilot_Youth1_290',
    'On-net 2000f_One_Month_100H; 30d', 'Data:DailyCycle_Pilot_1.5GB',
    'Jokko_Monthly', 'Facebook_MIX_2D', 'CVM_200f=400MB',
    'YMGX 100=1 hour FNF, 24H/1 month', 'Jokko_Weekly', 'Internat: 1000F_Zone_1;24H',
    'Data:30Go_V 30_Days', 'SUPERMAGIK_5000', 'FNF2 ( JAPPANTE)', '200F=10mnOnNetValid1H',
    'MIXT: 5000F=80Konnet_20Koffnet_250Mo;30d', 'pilot_offer6', '500=Unlimited3Day',
    'VAS(IVR_Radio_Monthly)', 'MROMO_TIMWES_OneDAY', 'Mixt : 500F=2500Fonnet_2500Foffnet ;5d',
    'Internat: 1000F_Zone_3;24h', 'All-net 5000= 20000off+20000on;30d', 'EVC_500=2000F',
    'Data: 200F=1GB,24H', 'Staff_CPE_Rent', 'SUPERMAGIK_1000', 'All-net 500F=4000F ; 5d',
    '305155009', 'DataPack_Incoming', 'Incoming_Bonus_woma', 'FIFA_TS_daily',
    'VAS(IVR_Radio_Weekly)', '1000=Unlimited7Day', 'Internat: 2000F_Zone_2;24H',
    'FNF_Youth_ESN', 'WIFI_ Family _10MBPS', 'Data_EVC_2Go24H',
    'MIXT: 4900F= 10H on net_1,5Go ;30d', 'EVC_Jokko_Weekly', 'EVC_JOKKO30',
    'Data_Mifi_20Go', 'Data_Mifi_10Go_Monthly', 'CVM_150F_unlimited',
    'CVM_100F_unlimited', 'CVM_100f=200 MB', 'FIFA_TS_weekly',
    '150=unlimited pilot auto', 'CVM_100f=500 onNet', 'GPRS_3000Equal10GPORTAL',
    'EVC_100Mo', 'GPRS_PKG_5GO_ILLIMITE', 'NEW_CLIR_PERMANENT_LIBERTE_MOBILE',
    'EVC_1Go', 'pilot_offer4', 'CVM_500f=2GB', 'pack_chinguitel_24h',
    'Postpaid FORFAIT 10H Package', 'EVC_700Mo', 'CVM_On-net 400f=2200F',
    'CVM_On-net 1300f=12500', 'All-net 500= 4000off+4000on;24H', 'SMS Max',
    'EVC_4900=12000F', 'APANews_weekly', 'NEW_CLIR_TEMPALLOWED_LIBERTE_MOBILE',
    'Data:OneTime_Pilot_1.5GB', 'YMGX on-net 100=700F, 24H', '301765007',
    '1500=Unlimited7Day', 'APANews_monthly', '200=unlimited pilot auto'
]

# Load models
@st.cache_resource(show_spinner='Loading Models...')
def load_model():
    try:
        xgb = joblib.load('Models Directory/xgb_model.joblib')
        lr = joblib.load('Models Directory/lr_model.joblib')
        return xgb, lr
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def create_input_features():
    st.header("Customer Information Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Information")
        region = st.selectbox(
            "Region",
            options=REGION_OPTIONS,
            help="Select customer's region"
        )
        tenure = st.selectbox(
            "Tenure",
            options=TENURE_OPTIONS,
            help="Select customer's tenure period"
        )
        montant = st.number_input("Top-up Amount (MONTANT)", min_value=0.0, key="montant")
        revenue = st.number_input("Monthly Revenue", min_value=0.0, key="revenue")
        
    with col2:
        st.subheader("Usage Metrics")
        frequence_rech = st.number_input("Refill Frequency", min_value=0, key="frequence_rech")
        arpu_segment = st.number_input("ARPU Segment (90-day income/3)", min_value=0.0, key="arpu_segment")
        frequence = st.number_input("Income Frequency", min_value=0, key="frequence")
        data_volume = st.number_input("Number of Connections", min_value=0, key="data_volume")
        regularity = st.number_input("90-day Activity Count", min_value=0, max_value=90, key="regularity")
        
    with col3:
        st.subheader("Call Statistics")
        on_net = st.number_input("Inter Expresso Calls", min_value=0, key="on_net")
        orange = st.number_input("Orange Calls", min_value=0, key="orange")
        tigo = st.number_input("Tigo Calls", min_value=0, key="tigo")
        zone1 = st.number_input("Zone 1 Calls", min_value=0, key="zone1")
        zone2 = st.number_input("Zone 2 Calls", min_value=0, key="zone2")
    
    st.subheader("Package Information")
    col4, col5 = st.columns(2)
    
    with col4:
        mrg = st.selectbox(
            "Client Going Status (MRG)",
            options=['Unknown', 'NO'],
            help="Select client's going status"
        )
        top_pack = st.selectbox(
            "Most Active Packs",
            options=TOP_PACK_OPTIONS,
            help="Select customer's most active package"
        )
    
    with col5:
        freq_top_pack = st.number_input("Top Pack Activation Frequency", min_value=0, key="freq_top_pack")

    # Process categorical variables for model input
    input_data = {
        'REGION': None if region == 'Unknown' else region,
        'TENURE': None if tenure == 'Unknown' else tenure,
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'DATA_VOLUME': data_volume,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'ZONE1': zone1,
        'ZONE2': zone2,
        'MRG': None if mrg == 'Unknown' else mrg,
        'REGULARITY': regularity,
        'TOP_PACK': None if top_pack == 'Unknown' else top_pack,
        'FREQ_TOP_PACK': freq_top_pack
    }
    
    return input_data

def make_prediction(input_data, models):
    xgb_model, lr_model = models
    
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])
    
    try:
        # Make predictions
        xgb_pred = xgb_model.predict_proba(df)[0][1]
        lr_pred = lr_model.predict_proba(df)[0][1]
        
        # Calculate average prediction
        avg_pred = (xgb_pred + lr_pred) / 2
        
        return {
            'XGBoost': xgb_pred,
            'Logistic Regression': lr_pred,
            'Ensemble Average': avg_pred
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title("Telecom Customer Churn Prediction")
    
    # Load models
    models = load_model()
    
    if models[0] is None or models[1] is None:
        st.error("Failed to load models. Please check model files and paths.")
        return
    
    # Get input features
    input_data = create_input_features()
    
    # Add prediction button
    if st.button("Predict Churn Probability"):
        predictions = make_prediction(input_data, models)
        
        if predictions:
            st.success("Churn Predictions")
            
            # Create columns for each prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="XGBoost Prediction",
                    value=f"{predictions['XGBoost']:.2%}"
                )
            
            with col2:
                st.metric(
                    label="Logistic Regression Prediction",
                    value=f"{predictions['Logistic Regression']:.2%}"
                )
            
            with col3:
                st.metric(
                    label="Ensemble Average",
                    value=f"{predictions['Ensemble Average']:.2%}"
                )
            
            # Add interpretation
        st.subheader("Interpretation")
        risk_level = "High" if predictions['Ensemble Average'] > 0.5 else "Low"

        if predictions['Ensemble Average'] > 0.5:
            st.warning(f"This customer has a **{risk_level} risk** of churning based on the provided data.")
        else:
            st.success(f"This customer has a **{risk_level} risk** of churning based on the provided data.")

        # Display input data for verification
        if st.checkbox("Show input data"):
            st.write("Input Data Used for Prediction:", input_data)

if __name__ == "__main__":
    main()