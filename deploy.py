import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Disable GPU to prevent compatibility issues
tf.config.set_visible_devices([], 'GPU')

# Load and compile the model
model = keras.models.load_model("CNN_eeg_model.h5")
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc', tf.keras.metrics.Recall(), 
                       tf.keras.metrics.AUC(), tf.keras.metrics.Precision()])

# Sidebar Information
st.sidebar.subheader('About the App')
st.sidebar.write('EEG Classification App with Streamlit using a trained CNN model')
st.sidebar.write('This app will classify EEG signal and determine whether the subject is a Good counter or Bad counter (for whom the mental task required excessive efforts).')

# Main UI
st.title("EEG Classification App")
st.write("Upload a CSV file containing EEG signal with **19 channels**: [Fp1, Fp2, F3, F4, F7, F8, T3, T4, C3, C4, T5, T6, P3, P4, O1, O2, Fz, Cz, Pz]")
st.write("Example input file: [s00.csv](https://drive.google.com/file/d/1wrWdREzw4z6rSK0kO3zkcYuHqjCz21Tp/view?usp=sharing)")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV file
    signal = pd.read_csv(uploaded_file, header=None)
    signal = signal.transpose().to_numpy()
    signal = signal.reshape(760, 775)

    # Select EEG channel
    channel_index = st.selectbox("Select EEG Channel", options=[i+1 for i in range(signal.shape[0])])
    
    # Visualize EEG Signal
    plt.plot(signal[channel_index - 1])
    plt.title(f'EEG Signal - Channel {channel_index}')
    plt.xlabel('Time')
    plt.ylabel('EEG Signal Amplitude')
    st.pyplot(plt.gcf())

if st.button("🧠 Classify EEG Signals\n(Brainwave Analysis)", key='classify_button'):
    if uploaded_file is not None:      
        # Ensure correct input shape for the model
        x_test = np.expand_dims(signal, axis=0)  # (1, 760, 775)
        x_test = np.expand_dims(x_test, axis=1)  # (1, 1, 760, 775)

        # Model Prediction
        predict = model.predict(x_test)

        # Output Result
        if predict[0][0] > 0.5:  # Assuming a binary classification threshold of 0.5
            st.success("✅ The subject is a **Good Counter**")
        else:
            st.error("❌ The subject is a **Bad Counter**")
    else:
        st.warning("⚠️ Please upload a CSV file before classifying.")

