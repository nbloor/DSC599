"""
Mission Success Predictor 
"""

import gradio as gr
import pandas as pd
import json
import os
import sys
from datetime import datetime

# Import the predictor module
try:
    from mission_predictor import MissionAnalysisAgent
except ImportError as e:
    print(f"Error importing mission_predictor: {e}")
    sys.exit(1)

# Initialize the agent
def initialize_agent():
    """Initialize the agent with fallback options"""
    try:
        agent = MissionAnalysisAgent(model_name="meta-llama/Llama-3.2-3B-Instruct")
        
        if os.path.exists("mission.csv"):
            agent.load_historical_data("mission.csv")
            print("Historical data loaded successfully")
        else:
            print("Warning: mission.csv not found")
            
        return agent
        
    except Exception as e:
        print(f"Warning during initialization: {e}")
        agent = MissionAnalysisAgent(model_name="meta-llama/Llama-3.2-3B-Instruct")
        if os.path.exists("mission.csv"):
            agent.load_historical_data("mission.csv")
        return agent

# Initialize agent and load data
agent = initialize_agent()

# Load the dataset for display
if os.path.exists("mission.csv"):
    df_missions = pd.read_csv("mission.csv")
    # Create mission options for dropdown
    mission_options = [f"Mission {i+1}: {row['Company']} - {row['Vehicle Type']} ({row['Launch Date']})" 
                      for i, row in df_missions.iterrows()]
else:
    df_missions = pd.DataFrame()
    mission_options = []

def get_mission_details(mission_selection):
    """Extract mission details from selection"""
    if not mission_selection:
        return None
    
    try:
        mission_idx = int(mission_selection.split(":")[0].replace("Mission", "").strip()) - 1
        if 0 <= mission_idx < len(df_missions):
            return df_missions.iloc[mission_idx]
    except:
        return None
    
    return None

def analyze_mission(mission_selection):
    """Analyze selected mission"""
    
    if not mission_selection:
        return "", "", ""
    
    # Get mission details
    mission_data = get_mission_details(mission_selection)
    if mission_data is None:
        return "Error loading mission data", "", ""
    
    # Mission info (simplified)
    mission_info = f"""**{mission_data['Company']} | {mission_data['Vehicle Type']} | {mission_data['Launch Date']}**
Temperature: {mission_data['Temperature (° F)']}°F | Wind: {mission_data['Wind speed (MPH)']} MPH | Humidity: {mission_data['Humidity (%)']}%
Payload: {mission_data['Payload Mass (kg)']} kg {mission_data['Payload Type']}
"""
    
    # Prepare mission data for prediction
    mission_dict = {
        'company': mission_data['Company'],
        'vehicle_type': mission_data['Vehicle Type'],
        'launch_site': mission_data['Launch Site'],
        'temperature': mission_data['Temperature (° F)'],
        'wind_speed': mission_data['Wind speed (MPH)'],
        'humidity': mission_data['Humidity (%)'],
        'payload_type': mission_data['Payload Type'],
        'payload_mass': mission_data['Payload Mass (kg)'],
        'payload_orbit': mission_data['Payload Orbit'],
        'rocket_height': mission_data['Rocket Height (m)'],
        'liftoff_thrust': mission_data['Liftoff Thrust (kN)']
    }
    
    # Get prediction
    try:
        features = agent._extract_features(mission_dict)
        prediction = agent.predict(mission_dict)
    except Exception as e:
        prediction = {'SUCCESS_PROBABILITY': 50, 'PREDICTED_STATUS': 'Unknown', 'CONFIDENCE': 'Low'}
    
    # Prediction result
    is_correct = prediction['PREDICTED_STATUS'] == mission_data['Mission Status']
    prediction_text = f"**Predicted: {prediction['PREDICTED_STATUS']}** ({'✓' if is_correct else '✗'})"
    
    # Ground truth
    actual_status = mission_data['Mission Status']
    failure_reason = mission_data.get('Failure Reason', '')
    
    ground_truth_text = f"**Actual: {actual_status}**"
    if actual_status == 'Failure' and failure_reason and pd.notna(failure_reason):
        ground_truth_text += f" - {failure_reason}"
    
    return mission_info, prediction_text, ground_truth_text

def predict_custom_mission(company, vehicle_type, launch_site, temperature, wind_speed, humidity, 
                          payload_type, payload_mass, payload_orbit, rocket_height, liftoff_thrust):
    """Predict success/failure for a custom mission"""
    
    # Build mission dictionary
    mission_dict = {
        'company': company,
        'vehicle_type': vehicle_type,
        'launch_site': launch_site,
        'temperature': temperature,
        'wind_speed': wind_speed,
        'humidity': humidity,
        'payload_type': payload_type,
        'payload_mass': payload_mass,
        'payload_orbit': payload_orbit,
        'rocket_height': rocket_height,
        'liftoff_thrust': liftoff_thrust
    }
    
    try:
        prediction = agent.predict(mission_dict)
    except Exception as e:
        prediction = {
            'SUCCESS_PROBABILITY': 50,
            'PREDICTED_STATUS': 'Unknown',
            'CONFIDENCE': 'Low',
            'KEY_RISK_FACTORS': ['Analysis error'],
            'RECOMMENDATION': 'CAUTION',
            'REASONING': f'Error: {str(e)}'
        }
    
    # Format the prediction result
    success_prob = prediction.get('SUCCESS_PROBABILITY', 50)
    status = prediction.get('PREDICTED_STATUS', 'Unknown')
    confidence = prediction.get('CONFIDENCE', 'Low')
    recommendation = prediction.get('RECOMMENDATION', 'CAUTION')
    risk_factors = prediction.get('KEY_RISK_FACTORS', [])
    reasoning = prediction.get('REASONING', '')
    
    # Create formatted output
    result_text = f"""
## Prediction Results

**Status:** {status} ({'🚀 Success' if status == 'Success' else '❌ Failure'})
**Success Probability:** {success_prob}%
**Confidence Level:** {confidence}

### Recommendation: {recommendation}

### Key Risk Factors:
{(chr(10).join([f'• {factor}' for factor in risk_factors])) if risk_factors else '• No significant risk factors identified'}

### Analysis:
{reasoning}
"""
    
    return result_text

# Create Gradio interface
with gr.Blocks(title="Mission Success Predictor") as demo:
    
    gr.Markdown("""
    # 🚀 Mission Success Predictor - Agentic AI
    **Advanced AI-powered prediction system using Llama 3.2 for space mission analysis**
    """)
    
    with gr.Tabs():
        
        # Tab 1: Historical Mission Analysis
        with gr.TabItem("📊 Historical Analysis"):
            gr.Markdown("### Analyze missions from your historical dataset")
            
            with gr.Row():
                with gr.Column():
                    mission_dropdown = gr.Dropdown(
                        choices=mission_options,
                        label="Select Historical Mission",
                        value=mission_options[0] if mission_options else None
                    )
                    analyze_btn = gr.Button("Analyze Mission", variant="primary", scale=1)
            
            with gr.Row():
                mission_info = gr.Markdown()
            
            with gr.Row():
                with gr.Column():
                    prediction_result = gr.Markdown()
                with gr.Column():
                    ground_truth = gr.Markdown()
            
            # Set up event handler for historical analysis
            analyze_btn.click(
                fn=analyze_mission,
                inputs=[mission_dropdown],
                outputs=[mission_info, prediction_result, ground_truth]
            )
        
        # Tab 2: Custom Mission Prediction
        with gr.TabItem("🎯 Custom Mission Prediction"):
            gr.Markdown("### Enter mission parameters to predict success probability")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Mission Organization & Type")
                    company = gr.Textbox(label="Company/Organization", placeholder="e.g., SpaceX, NASA, Blue Origin")
                    vehicle_type = gr.Textbox(label="Vehicle Type", placeholder="e.g., Falcon 9, Atlas V, New Shepard")
                    launch_site = gr.Textbox(label="Launch Site", placeholder="e.g., Kennedy Space Center, Cape Canaveral")
                
                with gr.Column():
                    gr.Markdown("#### Weather Conditions")
                    temperature = gr.Number(label="Temperature (°F)", value=65)
                    wind_speed = gr.Number(label="Wind Speed (MPH)", value=10)
                    humidity = gr.Number(label="Humidity (%)", value=60)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Payload Information")
                    payload_type = gr.Textbox(label="Payload Type", placeholder="e.g., Satellite, Crew, Cargo, Science")
                    payload_mass = gr.Number(label="Payload Mass (kg)", value=5000)
                    payload_orbit = gr.Textbox(label="Payload Orbit", placeholder="e.g., LEO, GEO, Lunar, Suborbital")
                
                with gr.Column():
                    gr.Markdown("#### Vehicle Specifications")
                    rocket_height = gr.Number(label="Rocket Height (m)", value=70)
                    liftoff_thrust = gr.Number(label="Liftoff Thrust (kN)", value=7000)
            
            predict_btn = gr.Button("🔮 Get Prediction", variant="primary", size="lg")
            prediction_output = gr.Markdown()
            
            # Set up event handler for custom prediction
            predict_btn.click(
                fn=predict_custom_mission,
                inputs=[company, vehicle_type, launch_site, temperature, wind_speed, humidity,
                       payload_type, payload_mass, payload_orbit, rocket_height, liftoff_thrust],
                outputs=[prediction_output]
            )
        
        # Tab 3: Model Information
        with gr.TabItem("ℹ️ Model Info"):
            gr.Markdown("### About the Prediction System")
            
            model_info = agent.get_model_insights()
            
            gr.Markdown(f"""
## Model: {model_info['model']}

### Historical Data
- **Total Missions Analyzed:** {model_info['total_historical_missions']}

### Model Capabilities
{chr(10).join([f'✓ {cap}' for cap in model_info['capabilities']])}

### How It Works
This agentic AI system:
1. **Extracts** relevant features from mission parameters
2. **Analyzes** historical patterns from your dataset
3. **Assesses** weather, vehicle, and company-specific risks
4. **Predicts** mission outcomes with confidence levels
5. **Provides** actionable recommendations

### Key Features
- **LLM-Based Analysis:** Uses Llama 3.2 for nuanced decision-making
- **Historical Pattern Recognition:** Learns from past successes and failures
- **Multi-Factor Risk Scoring:** Considers weather, equipment, and organizational factors
- **Confidence Levels:** Indicates prediction reliability (High/Medium/Low)
- **Actionable Recommendations:** PROCEED, CAUTION, or POSTPONE guidance

### Weather Risk Factors
- Temperature extremes (< 40°F or > 95°F)
- High wind speeds (> 20 MPH)
- Excessive humidity (> 90%)
- Combination of adverse conditions
""")

if __name__ == "__main__":
    print("🚀 Launching Mission Success Predictor...")
    print("📊 Gradio interface will open at http://localhost:7860")
    demo.launch(share=False)
