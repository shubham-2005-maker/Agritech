import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# =============================================
# CUSTOM CSS STYLING
# =============================================
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 12px 28px;
            font-weight: bold;
            transition: all 0.3s;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.02);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stNumberInput, .stSelectbox {
            margin-bottom: 1.5rem;
        }
        .success-box {
            padding: 1.5rem;
            background-color: #dff0d8;
            color: #3c763d;
            border-radius: 8px;
            margin: 1.5rem 0;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .red-box {
            padding: 1.5rem;
            background-color: #f08080;
            color: white;
            border-radius: 8px;
            margin: 1.5rem 0;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .title {
            color: #2e7d32;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.2rem;
        }
        .header {
            color: #4CAF50;
            font-size: 1.2rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .tab-container {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-top: 1.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 24px;
            background-color: #E8F5E9;
            border-radius: 8px 8px 0 0;
            transition: all 0.3s;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50;
            color: white;
        }
        .water-icon {
            font-size: 1.5rem;
            vertical-align: middle;
            margin-right: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================
# MODEL LOADING AND PREPROCESSING FUNCTIONS
# =============================================
@st.cache_resource
def load_models():
    """Load all required models and transformers"""
    model = load_model("irrigation_lstm.h5")
    
    with open("label_encoder.pkl", "rb") as file:
        le = pickle.load(file)
    
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    
    return model, le, scaler

def preprocess_input(moisture, temperature, weather):
    """Preprocess input data to match model requirements"""
    weather_encoded = le.transform([weather])[0]
    input_data = np.array([[moisture, temperature, weather_encoded]], dtype=np.float32)
    input_data[:, :2] = scaler.transform(input_data[:, :2])
    return input_data.reshape(1, input_data.shape[1], 1)

# =============================================
# VISUALIZATION FUNCTIONS
# =============================================
def create_gauge_chart(confidence):
    """Create confidence gauge chart"""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgreen"},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 50], 'color': "#f8f9fa"},
                {'range': [50, 100], 'color': "#e9f5e9"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence}}))

def create_radar_chart(moisture, temperature, weather):
    """Create input features radar chart"""
    weather_numeric = le.transform([weather])[0]/len(le.classes_)*100
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[moisture, temperature, weather_numeric],
        theta=['Moisture','Temperature','Weather'],
        fill='toself',
        name='Input Features',
        line_color='#4CAF50'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,100]
            )),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_moisture_indicator(moisture):
    """Create soil moisture status indicator"""
    status = "🟢 Optimal" if moisture > 50 else "🟠 Dry" if moisture > 30 else "🔴 Critical"
    return go.Figure(go.Indicator(
        mode="number+gauge",
        value=moisture,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current Soil: {status}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 30], 'color': "lightcoral"},
                {'range': [30, 50], 'color': "moccasin"},
                {'range': [50, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': moisture}
        }))

def simulate_moisture_trend(current_moisture, days=7):
    """Generate realistic moisture trend based on current value"""
    base_trend = np.linspace(current_moisture * 0.7, current_moisture * 1.3, days)
    noise = np.random.normal(0, 5, days)  # Small random fluctuations
    simulated_moisture = (base_trend + noise).clip(0, 100)
    return simulated_moisture.round(1)

def show_soil_timeline(moisture):
    """Display soil moisture timeline (real or simulated)"""
    historical_data_exists = False  # Change this when you have real data
    
    if historical_data_exists:
        days = list(range(1, 8))
        moisture_levels = np.random.normal(loc=moisture, scale=10, size=7).clip(0, 100)
        title = "Soil Moisture Trend (Last 7 Days)"
    else:
        days = list(range(1, 8))
        moisture_levels = simulate_moisture_trend(moisture)
        title = "Simulated Moisture Trend"
    
    fig = px.line(
        x=days, 
        y=moisture_levels,
        title=title,
        labels={'x': 'Days', 'y': 'Moisture %'},
        markers=True,
        line_shape='spline',
        color_discrete_sequence=['#4CAF50']
    )
    
    # Add moisture zones
    fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="red", opacity=0.1, 
                 annotation_text="Dry Zone", annotation_position="bottom left")
    fig.add_hrect(y0=30, y1=50, line_width=0, fillcolor="orange", opacity=0.1)
    fig.add_hrect(y0=50, y1=100, line_width=0, fillcolor="green", opacity=0.1,
                 annotation_text="Optimal Zone", annotation_position="top left")
    
    # Highlight current moisture
    fig.add_hline(y=moisture, line_dash="dot", line_color="blue",
                 annotation_text=f"Current: {moisture}%", annotation_position="top right")
    
    if not historical_data_exists:
        fig.add_annotation(
            text="<i>Simulated data for demonstration</i>",
            xref="paper", yref="paper", x=0.5, y=-0.2, showarrow=False
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if not historical_data_exists:
        with st.expander("🔍 How to get real historical data?"):
            st.markdown("""
            **To track actual soil moisture history:**
            1. Connect IoT soil sensors (e.g., Xiaomi Flora, Arduino)
            2. Install our data logger software
            3. Enable cloud sync for continuous monitoring
            
            [Download sensor setup guide](#) | [View compatible devices](#)
            """)

def create_impact_chart(moisture, temp, weather):
    """Calculate and visualize irrigation impact factors"""
    # Calculate impacts (science-based)
    moisture_impact = (50 - moisture) * 1.5  # 50% is ideal (0 impact)
    temp_impact = (temp - 25) * 0.8          # 25°C is ideal (0 impact)
    weather_impact = {
        "Rainy": -45,
        "Cloudy": -20,
        "Sunny": +45,
    }.get(weather, 0)
    
    # Create figure
    fig = px.bar(
        x=["Soil Moisture", "Temperature", "Weather"],
        y=[moisture_impact, temp_impact, weather_impact],
        color=["Soil", "Temp", "Weather"],
        color_discrete_map={
            "Soil": "#EF553B",  # Red
            "Temp": "#00CC96",  # Green
            "Weather": "#636EFA" # Blue
        },
        labels={"y": "Impact Score", "x": ""},
        text=[f"{v:+.0f}" for v in [moisture_impact, temp_impact, weather_impact]]
    )
    
    # Add critical annotations
    annotations = []
    if moisture_impact > 40:
        annotations.append(dict(x=0, y=moisture_impact+5, text="🚨 Critical", showarrow=False))
    elif moisture_impact < -30:
        annotations.append(dict(x=0, y=moisture_impact-5, text="⚠️ Alert", showarrow=False))
    
    # Customize layout
    fig.update_layout(
        title="Irrigation Impact Factors",
        yaxis_range=[-60, 60],
        annotations=annotations,
        showlegend=False
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig

# =============================================
# STREAMLIT UI DESIGN AND LOGIC
# =============================================
model_lstm, le, scaler = load_models()

st.markdown('<h1 class="title">🌱 AI-Powered Irrigation Prediction</h1>', unsafe_allow_html=True)
st.markdown('<div class="header">Smart irrigation Classification based on soil conditions</div>', unsafe_allow_html=True)

# Input columns
col1, col2 = st.columns(2)
with col1:
    moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0, 
                             help="Current soil moisture percentage")
    weather = st.selectbox("Weather Condition", options=le.classes_,
                         help="Select current weather situation")
with col2:
    temperature = st.number_input("Temperature (°C)", value=25.0,
                                help="Current air temperature in Celsius")

if st.button("Get Irrigation Status", key="predict"):
    processed_input = preprocess_input(moisture, temperature, weather)
    prediction = model_lstm.predict(processed_input)
    proba = prediction[0][0]
    predicted_class = "Irrigation Needed" if proba > 0.5 else "No Irrigation Needed"
    confidence = abs(proba-0.5)*200

    # Results container
    with st.container():
        if(proba > 0.5):
         st.markdown(f"""
         <div class="red-box">
            <span class="water-icon">💧</span> <b>Status:</b> {predicted_class}<br>
            <small>Model confidence: {confidence:.1f}%</small>
         </div>
         """, unsafe_allow_html=True)
        else:
          st.markdown(f"""
          <div class="success-box">
            <span class="water-icon">💧</span> <b>Status:</b> {predicted_class}<br>
            <small>Model confidence: {confidence:.1f}%</small>
          </div>
          """, unsafe_allow_html=True)    
        
        # Visualization tabs (now with 5 tabs)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Confidence", "Input Analysis", "Soil Status", "Moisture History", "Impact Factors"])
        
        with tab1:
            st.plotly_chart(create_gauge_chart(confidence), use_container_width=True)
            st.markdown("""
            **Confidence Interpretation:**
            - <80%: Low confidence (check sensor values)
            - 80-90%: Moderate confidence
            - >90%: High confidence
            """)
        
        with tab2:
            st.plotly_chart(create_radar_chart(moisture, temperature, weather), use_container_width=True)
            st.markdown("""
            **Radar Chart Guide:**
            - Balanced triangle = ideal input distribution
            - Large weather spike = weather dominant factor
            - Small moisture = potential drought risk
            """)
        
        with tab3:
            st.plotly_chart(create_moisture_indicator(moisture), use_container_width=True)
            if moisture < 40:
                st.warning("**Soil is drying** - consider increasing irrigation frequency")
            elif moisture > 80:
                st.warning("**Soil is very wet** - reduce irrigation to prevent root rot")
            else:
                st.success("**Soil moisture is in optimal range**")
        
        with tab4:
            show_soil_timeline(moisture)
            
        with tab5:
            st.plotly_chart(create_impact_chart(moisture, temperature, weather), use_container_width=True)
            st.markdown("""
            **Impact Score Guide:**
            - Positive → Increases irrigation need
            - Negative → Decreases irrigation need
            - Critical Thresholds:
              • Soil Moisture: >|40| = Emergency
              • Temperature: >|15| = Significant
              • Weather: Rain/Sunny = Major Influence
            """)
        
        # Contextual feedback
        if predicted_class == "Irrigation Needed":
            cols = st.columns([3, 1])
            with cols[0]:
                st.warning("""
                **Recommended Actions:**
                - Water plants within 24 hours
                - Check deeper soil layers
                - Monitor after irrigation
                """)
        else:
            cols = st.columns([3, 1])
            with cols[0]:
                st.info("""
                **Maintenance Tips:**
                - Check again in 2-3 days
                - Monitor weather changes
                - Ensure proper drainage
                """)
            with cols[1]:
                st.image("https://media.giphy.com/media/3o7TKsrfldgW9UXxWM/giphy.gif", width=150)

        # Input summary
        with st.expander("📋 View Detailed Input Analysis"):
            st.json({
                "Soil Moisture": f"{moisture}%",
                "Temperature": f"{temperature}°C", 
                "Weather Condition": weather,
                "Prediction Probability": f"{proba:.4f}",
                "Confidence Score": f"{confidence:.1f}%"
            })
            st.markdown("""
            **Threshold Information:**
            - Irrigation Status is 1(nedded) when probability > 0.5
            - Confidence measures prediction certainty
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🌻 Sustainable farming through AI | v1.2.0</p>
    <p>Note: Predictions are estimates. Always verify field conditions.</p>
</div>
""", unsafe_allow_html=True)