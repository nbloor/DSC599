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

# Create Gradio interface
with gr.Blocks(title="Mission Predictor") as demo:
    
    gr.Markdown("""
    # Mission Success Predictor - Agentic AI (Nicholas-Bloor)
    """)
    
    with gr.Row():
        with gr.Column():
            mission_dropdown = gr.Dropdown(
                choices=mission_options,
                label="Select Mission",
                value=mission_options[0] if mission_options else None
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
    
    with gr.Row():
        mission_info = gr.Markdown()
    
    with gr.Row():
        with gr.Column():
            prediction_result = gr.Markdown()
        with gr.Column():
            ground_truth = gr.Markdown()
    
    # Set up event handler
    analyze_btn.click(
        fn=analyze_mission,
        inputs=[mission_dropdown],
        outputs=[mission_info, prediction_result, ground_truth]
    )

if __name__ == "__main__":
    demo.launch()
