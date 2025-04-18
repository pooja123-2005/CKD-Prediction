import streamlit as st
import pandas as pd
import sqlite3
import bcrypt
import google.generativeai as genai
import joblib
import numpy as np

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="CKD Prediction", layout="wide")
genai.configure(api_key="AIzaSyCfcHaaHNkBLiSiSA4v_UOqOktoxNxE1ag")  # Replace with your key
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# ---------------- FUNCTIONS ---------------- #
def get_health_precautions(patient_data):
    prompt = (f"Patient test results: {patient_data}. Based on these lab values, provide concise health precautions. "
              "Limit to 2-3 key points covering diet, hydration, and lifestyle. Keep it brief and patient-friendly.")
    response = model.generate_content(prompt)
    return response.text

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )""")
    conn.commit()
    conn.close()

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_pw = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user and check_password(password, user[0])

def load_data_reference():
    try:
        data = pd.read_csv("whole_data.csv")
        return data
    except FileNotFoundError:
        st.error("Reference data file 'whole_data.csv' not found.")
        return None

# Function to get min/max values for numerical fields
def get_min_max(data, column):
    if data is not None and column in data.columns:
        min_val = float(data[column].min())
        max_val = float(data[column].max())
        return min_val, max_val
    # Default fallback values
    defaults = {
        'age': (0, 100),
        'bp': (50, 200),
        'sg': (1.005, 1.025),
        'su': (0, 5),
        'bgr': (70, 490),
        'bu': (10, 200),
        'sc': (0.4, 15.0),
        'sod': (110, 160),
        'pot': (2.5, 7.5),
        'hemo': (3.1, 17.8)
    }
    return defaults.get(column, (0, 100))

# Function to get unique values for categorical fields
def get_unique_values(data, column):
    if data is not None and column in data.columns:
        unique_vals = data[column].dropna().unique().tolist()
        if unique_vals:
            return unique_vals
    
    # Default fallback values for categorical fields
    defaults = {
        'rbc': ['normal', 'abnormal'],
        'pc': ['normal', 'abnormal'],
        'pcc': ['present', 'notpresent'],
        'ba': ['present', 'notpresent'],
        'pcv': ['normal', 'abnormal'],
        'wc': ['normal', 'abnormal'],
        'rc': ['normal', 'abnormal'],
        'htn': ['yes', 'no'],
        'dm': ['yes', 'no'],
        'cad': ['yes', 'no'],
        'appet': ['good', 'poor'],
        'pe': ['yes', 'no'],
        'ane': ['yes', 'no']
    }
    return defaults.get(column, ['normal', 'abnormal'])

# ---------------- UI ---------------- #
init_db()

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
    # Load reference data and ML pipeline
    reference_data = load_data_reference()
    
    try:
        pipeline = joblib.load('model_pipeline.joblib')
    except FileNotFoundError:
        pipeline = None
        if st.session_state.authenticated:
            st.error("ML pipeline file 'model_pipeline.joblib' not found.")

    menu = ["Home", "Login", "Sign Up"] if not st.session_state.authenticated else ["Home", "Logout"]
    left, right = st.columns([1, 3])

    # -------- LEFT SIDE -------- #
    with left:
        choice = st.selectbox("🏠 Menu", menu)

        if choice == "Login":
            st.subheader("🔑 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        elif choice == "Sign Up":
            st.subheader("📝 Create an Account")
            new_user = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                if register_user(new_user, new_password):
                    st.success("✅ Account created! You can now log in.")
                else:
                    st.error("❌ Username already taken.")

        elif choice == "Logout":
            st.session_state.authenticated = False
            st.success("✅ Logged out successfully!")
            st.rerun()

        if choice == "Home" and st.session_state.authenticated:
            st.subheader("🔮 Prediction Section")

            if st.button("Predict"):
                if pipeline is None:
                    st.error("ML pipeline not loaded. Please check if the file exists.")
                    st.stop()

                # Create dataframe from inputs
                input_df = pd.DataFrame([st.session_state.inputs])
                
                # Ensure correct data types - convert numerical columns to float
                numerical_cols = ['age', 'bp', 'sg', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
                for col in numerical_cols:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(float)
                
                # Ensure categorical columns are strings
                categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
                for col in categorical_cols:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)
                
                # Convert albumin (al) to float if present
                if 'al' in input_df.columns:
                    input_df['al'] = input_df['al'].astype(float)
                
                # Make prediction using the pipeline
                try:
                    # Debug information
                    
                    prediction = pipeline.predict(input_df)[0]
                    
                    st.subheader("🔍 Prediction Result")
                    if prediction == 1:
                        st.error("🚨 Positive: Chronic Kidney Disease Detected!")
                        st.error("**🚨 Please consult a doctor immediately !!!**")
                    else:
                        st.success("✅ Negative: No Chronic Kidney Disease Detected")

                    st.subheader("🩺 Health Precautions")
                    patient_data_str = ', '.join([f"{k}: {v}" for k, v in st.session_state.inputs.items()])
                    precautions = get_health_precautions(patient_data_str)
                    st.markdown(precautions.strip())
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.write("Please ensure all input fields match what the model expects.")

    # -------- RIGHT SIDE -------- #
    with right:
        if choice == "Home" and st.session_state.authenticated:
            st.title("🔬 Chronic Kidney Disease Prediction")
            # st.image("ckdimg.png", use_container_width=True)

            # Create input form using reference data for ranges and options
            col1, col2 = st.columns(2)
            
            with col1:
                age_min, age_max = get_min_max(reference_data, 'age')
                age = st.number_input('Age', min_value=age_min, max_value=age_max, value=age_min)
                
                bp_min, bp_max = get_min_max(reference_data, 'bp')
                bp = st.number_input('Blood Pressure (mmHg)', min_value=bp_min, max_value=bp_max, value=bp_min)
                
                sg_min, sg_max = get_min_max(reference_data, 'sg')
                sg = st.number_input('Specific Gravity', min_value=sg_min, max_value=sg_max, value=sg_min, format="%.3f", step=0.001)
                
                al = st.slider('Albumin', 0, 5, 0)
                
                su_min, su_max = get_min_max(reference_data, 'su')
                su = st.number_input('Sugar', min_value=su_min, max_value=su_max, value=su_min)
                
                rbc_options = get_unique_values(reference_data, 'rbc')
                rbc = st.selectbox('Red Blood Cells', rbc_options)
                
                pc_options = get_unique_values(reference_data, 'pc')
                pc = st.selectbox('Pus Cell', pc_options)
                
                pcc_options = get_unique_values(reference_data, 'pcc')
                pcc = st.selectbox('Pus Cell Clumps', pcc_options)
                
                ba_options = get_unique_values(reference_data, 'ba')
                ba = st.selectbox('Bacteria', ba_options)
                
                bgr_min, bgr_max = get_min_max(reference_data, 'bgr')
                bgr = st.number_input('Blood Glucose Random (mg/dl)', min_value=bgr_min, max_value=bgr_max, value=bgr_min)
                
                bu_min, bu_max = get_min_max(reference_data, 'bu')
                bu = st.number_input('Blood Urea (mg/dl)', min_value=bu_min, max_value=bu_max, value=bu_min)

            with col2:
                sc_min, sc_max = get_min_max(reference_data, 'sc')
                sc = st.number_input('Serum Creatinine (mg/dl)', min_value=sc_min, max_value=sc_max, value=sc_min, format="%.1f", step=0.1)
                
                sod_min, sod_max = get_min_max(reference_data, 'sod')
                sod = st.number_input('Sodium (mEq/L)', min_value=sod_min, max_value=sod_max, value=sod_min)
                
                pot_min, pot_max = get_min_max(reference_data, 'pot')
                pot = st.number_input('Potassium (mEq/L)', min_value=pot_min, max_value=pot_max, value=pot_min, format="%.1f", step=0.1)
                
                hemo_min, hemo_max = get_min_max(reference_data, 'hemo')
                hemo = st.number_input('Hemoglobin (g/dl)', min_value=hemo_min, max_value=hemo_max, value=hemo_min, format="%.1f", step=0.1)
                
                pcv_options = get_unique_values(reference_data, 'pcv')
                pcv = st.selectbox('Packed Cell Volume', pcv_options)
                
                wc_options = get_unique_values(reference_data, 'wc')
                wc = st.selectbox('White Blood Cell Count', wc_options)
                
                rc_options = get_unique_values(reference_data, 'rc')
                rc = st.selectbox('Red Blood Cell Count', rc_options)
                
                htn_options = get_unique_values(reference_data, 'htn')
                htn = st.selectbox('Hypertension', htn_options)
                
                dm_options = get_unique_values(reference_data, 'dm')
                dm = st.selectbox('Diabetes Mellitus', dm_options)
                
                cad_options = get_unique_values(reference_data, 'cad')
                cad = st.selectbox('Coronary Artery Disease', cad_options)
                
                appet_options = get_unique_values(reference_data, 'appet')
                appet = st.selectbox('Appetite', appet_options)
                
                pe_options = get_unique_values(reference_data, 'pe')
                pe = st.selectbox('Pedal Edema', pe_options)
                
                ane_options = get_unique_values(reference_data, 'ane')
                ane = st.selectbox('Anemia', ane_options)

            # Store all inputs in session state
            st.session_state.inputs = {
                'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba, 'bgr': bgr, 
                'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo,
                'pcv': pcv, 'wc': wc, 'rc': rc, 'htn': htn, 'dm': dm,
                'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
            }
            
        elif choice == "Home" and not st.session_state.authenticated:
            st.warning("🔒 Please log in to access the CKD prediction form.")

    # -------- Footer -------- #
    st.markdown("""<hr style="margin-top: 2em;">""", unsafe_allow_html=True)
    st.markdown("Mentors: <b>Sumana Das</b> |[🔗 LinkedIn](https://www.linkedin.com/search/results/all/?heroEntityKey=urn%3Ali%3Afsd_profile%3AACoAAAIwVBYBJTLkL5coXl4BYqE3spJc2ey2xV8&keywords=SUMANA%20DAS&origin=ENTITY_SEARCH_HOME_HISTORY&sid=_Yb) ,<b> Aparna Tanam</b> | 🔗 [LinkedIn](https://in.linkedin.com/in/aparna-tanam-42532929)",unsafe_allow_html=True)
    st.markdown("Made by : <b>Methuku Divyasri</b> | [🔗 LinkedIn](https://www.linkedin.com/in/methuku-divyasri-834b64278/) , <b>Garlapad Akshara</b> | [🔗 LinkedIn](https://www.linkedin.com/search/results/all/?fetchDeterministicClustersOnly=true&heroEntityKey=urn%3Ali%3Afsd_profile%3AACoAAEjBC1MB9SooWJfOxV0bOlDjZ6HKPPhrwz0&keywords=akshara%20garlapad&origin=RICH_QUERY_TYPEAHEAD_HISTORY&position=0&searchId=501d8acb-0621-4cac-beb0-94dba8d9e5e1&sid=0Mo&spellCorrectionEnabled=true) ,<b> Poojitha Kottam</b> | [🔗 LinkedIn](https://in.linkedin.com/in/poojitha-kottam-2064b5255?original_referer=https%3A%2F%2Fwww.google.com%2F)", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
