import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# Set English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Feature name mapping
FEATURE_NAME_MAPPING = {
    'ECWTBW': 'ECW/TBW ratio',
    'Age': 'Age',
    'Duration_of_diabetes': 'Duration of diabetes',
    'Creatinine': 'Scr',
    'TSH': 'TSH',
    'Platelet': 'Platelet',
    'Weight': 'Weight',
    'Height': 'Height'
}

# Use SHAP library's built-in force plot
def create_shap_force_plot(model, input_df, risk_score):
    """
    Create SHAP force plot
    """
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Determine prediction class
        predicted_class = 1 if risk_score > 0.5 else 0
        
        # Extract SHAP values and base value
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_array = shap_values[predicted_class][0]
            base_value = explainer.expected_value[predicted_class]
        elif len(shap_values.shape) == 3:
            shap_array = shap_values[0, :, predicted_class]
            base_value = explainer.expected_value[predicted_class]
        else:
            shap_array = shap_values[0]
            base_value = explainer.expected_value
        
        # Prepare feature names
        feature_names = []
        for feature in input_df.columns:
            display_name = FEATURE_NAME_MAPPING.get(feature, feature)
            feature_names.append(display_name)
        
        # Create SHAP force plot
        plt.figure(figsize=(16, 6))
        
        # Use SHAP force plot
        shap.force_plot(
            base_value, 
            shap_array, 
            input_df.values[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        # Get current figure
        fig = plt.gcf()
        
        # Remove title
        plt.title('')
        
        # Adjust layout
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.85)
        
        return fig
        
    except Exception as e:
        # If failed, create error message plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"SHAP visualization failed", 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return fig

# Model loading function
def load_model(model_path):
    """
    Load model file
    """
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file does not exist: {model_path}")
            return None
            
        model = joblib.load(model_path)
        return model
        
    except Exception as e:
        st.error(f"Model loading failed")
        return None

# Get model feature list
def get_model_features(model):
    """
    Extract feature list from model
    """
    try:
        if hasattr(model, 'feature_name_'):
            return model.feature_name_
        elif hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_
        elif hasattr(model, 'features_name'):
            return model.features_name
        else:
            return ['ECWTBW', 'Age', 'Duration_of_diabetes', 'Creatinine', 'TSH', 'Platelet', 'Weight', 'Height']
    except Exception as e:
        return ['ECWTBW', 'Age', 'Duration_of_diabetes', 'Creatinine', 'TSH', 'Platelet', 'Weight', 'Height']

# Prediction function
def predict_dpn_risk(model, input_df):
    """
    Use model for prediction
    """
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)
            risk_score = probabilities[0, 1]
        else:
            prediction = model.predict(input_df)
            risk_score = float(prediction[0])
            risk_score = max(0, min(1, risk_score))
        
        return risk_score
        
    except Exception as e:
        st.error(f"Prediction failed")
        return 0.5

# Main application logic
def main():
    st.title("DPN Risk Prediction Model")
    
    # Model file path
    model_path = "extra_trees_model.pkl"
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = load_model(model_path)
        
    if 'model_features' not in st.session_state:
        if st.session_state.model is not None:
            st.session_state.model_features = get_model_features(st.session_state.model)
        else:
            st.session_state.model_features = [
                'ECWTBW', 'Age', 'Duration_of_diabetes', 'Creatinine', 
                'TSH', 'Platelet', 'Weight', 'Height'
            ]
    
    # Display model status
    if st.session_state.model is None:
        st.error("Model loading failed, please check model file path")
        st.info("Please ensure extra_trees_model.pkl file is in the current directory")
        return
    
    # Create input form
    st.header("Patient Information Input")
    
    input_data = {}
    for feature in st.session_state.model_features:
        # Get display name
        display_name = FEATURE_NAME_MAPPING.get(feature, feature)
        
        input_data[feature] = st.number_input(
            f"{display_name}",
            value=0.0,
            min_value=0.0,
            max_value=500.0,
            step=0.1,
            key=f"input_{feature}"
        )
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button("Predict", use_container_width=True)
    
    if predict_button:
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        risk_score = predict_dpn_risk(st.session_state.model, input_df)
        
        # Display prediction result text
        st.markdown("---")
        st.write(f"**Based on feature values, predicted possibility of DPN is {risk_score:.2%}**")
        
        # Create SHAP visualization
        shap_fig = create_shap_force_plot(
            st.session_state.model, 
            input_df,
            risk_score
        )
        
        # Display SHAP plot
        st.pyplot(shap_fig)

# Run main program
if __name__ == "__main__":
    main()