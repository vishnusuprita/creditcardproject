import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open(r"C:\Users\Admin\Desktop\python_project\models\Descisiontree_model.pkl", 'rb'))

st.title("💳 Credit Card Fraud Detection App")

# Input method selector
input_method = st.radio("Select Input Method:", ("Slider Input", "Text Input (Paste values)"))

columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

if input_method == "Slider Input":
    st.markdown("### Use sliders below to enter feature values:")

    time = st.slider("Time", min_value=0.0, max_value=172792.0, step=100.0)
    amount = st.slider("Amount", min_value=0.0, max_value=25691.16, step=1.0)

    feature_ranges = {
        "V1": (-56.407510, 2.454930),
        "V2": (-72.715728, 22.057729),
        "V3": (-48.325589, 9.382558),
        "V4": (-5.683171, 16.875344),
        "V5": (-113.743307, 34.801666),
        "V6": (-26.160506, 73.301626),
        "V7": (-43.557242, 120.589494),
        "V8": (-73.216718, 20.007208),
        "V9": (-13.434066, 15.594995),
        "V10": (-24.588262, 23.745136),
        "V11": (-4.797473, 12.018913),
        "V12": (-18.683715, 7.848392),
        "V13": (-5.791881, 7.126883),
        "V14": (-19.214325, 10.526766),
        "V15": (-4.498945, 8.877742),
        "V16": (-14.129855, 17.315112),
        "V17": (-25.162799, 9.253526),
        "V18": (-9.498746, 5.041069),
        "V19": (-7.213527, 5.591971),
        "V20": (-54.497720, 39.420904),
        "V21": (-34.830382, 27.202839),
        "V22": (-10.933144, 10.503090),
        "V23": (-44.807735, 22.528412),
        "V24": (-2.836627, 4.584549),
        "V25": (-10.295397, 7.519589),
        "V26": (-2.604551, 3.517346),
        "V27": (-22.565679, 31.612198),
        "V28": (-15.430084, 33.847808)
    }

    v_features = []
    for i in range(1, 29):
        key = f"V{i}"
        min_val, max_val = feature_ranges[key]
        value = st.slider(key, min_value=float(min_val), max_value=float(max_val), step=0.1, value=0.0)
        v_features.append(value)

    input_data = pd.DataFrame([[time] + v_features + [amount]], columns=columns)

    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.error(f"⚠️ Fraudulent Transaction! (Confidence: {prediction_proba[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {prediction_proba[0][0]*100:.2f}%)")

else:
    st.markdown("### Paste a row of values (Time, V1–V28, Amount):")
    example = "406, -2.31, 1.95, ..., -0.14, 0"  # example shortened for display
    st.markdown(f"Example:\n\n`{example}`")

    user_input = st.text_area("Paste your comma-separated values below:", height=150)

    if st.button("Submit Text Input"):
        try:
            values = [float(x.strip()) for x in user_input.strip().split(",") if x.strip() != '']
            if len(values) != 30:
                st.error(f"⚠️ Expected 30 values, but got {len(values)}.")
            else:
                input_data = pd.DataFrame([values[:30]], columns=columns)

                # Predict and show result
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)

                if prediction[0] == 1:
                    st.error(f"⚠️ Fraudulent Transaction! (Confidence: {prediction_proba[0][1]*100:.2f}%)")
                else:
                    st.success(f"✅ Legitimate Transaction (Confidence: {prediction_proba[0][0]*100:.2f}%)")
        except Exception as e:
            st.error(f"Error parsing input: {e}")
