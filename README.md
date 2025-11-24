# Agentic AI-Based Forecasting for Enhanced Space Mission Safety

A neural network–based predictive framework for improving the safety and reliability of space launch operations by forecasting short-term atmospheric conditions such as wind shear, lightning, cloud cover, and precipitation.

---

## Overview

This project investigates the use of **agentic AI systems** and **neural network architectures** to analyze and predict launch-critical environmental constraints.  
By fusing multi-source meteorological data — including radiosonde soundings, radar, satellite imagery, and surface observations — the system produces **probabilistic, high-resolution forecasts** that enhance decision-making under launch commit criteria (LCCs).
The project also analyzes historical launches and creates an agentic AI system that predicts whether a launch would be a success or a failure.

It includes:
- A Mission Analysis Agent (MissionAnalysisAgent)
- A Gradio UI (app.py) for interactive prediction
- A historical launch dataset (mission.csv)
- Several meteorological datasets for future model development

---

## Features

- **Multi-Source Data Fusion**: Integrates atmospheric data from radar, satellite, radiosonde, and numerical weather prediction (NWP) models  
- **Probabilistic Forecasting**: Generates calibrated risk estimates for key hazards  
- **Model Interpretability**: Explains reasoning using feature attribution and counterfactual analyses  
- **Safety-Driven Evaluation**: Quantifies impact on launch decision safety and efficiency  
- **Reproducible Research**: Version-controlled pipelines and reproducible experiments  

---

## Project Objectives

1. **Develop** a neural forecasting framework that fuses heterogeneous meteorological data  
2. **Generate** short-term (0–6 hour) forecasts of wind shear, lightning, and precipitation  
3. **Quantify and validate** uncertainty using probabilistic metrics  
4. **Interpret model behavior** to ensure operational transparency  
5. **Evaluate** the model against historical launch weather and decision outcomes  

---

## Project Structure
``` bash
├── app.py                 # Gradio UI for mission selection + prediction
├── mission_predictor.py   # Agentic AI logic and prediction system
├── mission.csv            # Historical mission dataset for pattern analysis
└── data/
      └── mission_launches.csv            # Primary data set for the agentic AI protion of the project
      └── PTER_NEMCC_040824_.1729.xlsx    # Weather balloon data taken from site A
      └── PTER_NEMCC_101423_.1452.xlsx    # Weather balloon data taken from site B
      └── spaceMissions.csv               # Alternate Space launch fdata including various weather phenomena readings
      └── Stratostar EclipseFlight.csv    # Weather balloon data taken from site C
└── README.md              # Project documentation
└── requirements.txt       # List of required Python packages for execution
```

---

## Quick Start (Installation Instructions)

### Prerequisites

- **Python 3.9+**
- GPU-enabled environment recommended
- HuggingFace API access (for Llama 3.2 InferenceClient)
- (Future) Framework support: PyTorch or TensorFlow

### 1. Clone Repository

```bash
git clone https://github.com/nbloor/DSC599
cd agentic-ai-launch-forecasting
```

### 2. Create Environment

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install gradio
pip install scikit-learn
pip install transformers
pip install plotly
```

## Usage Instructions

### Run the Gradio App
```bash
python app.py
```

Access the UI at:
```arduino
http://localhost:7860
```
!(images/outputScreenshot.png "Image of the UI")

What the UI does
- Loads mission.csv
- Allows selecting a mission from a dropdown
- Displays:
  - Mission Parameters
  - Predicted Status (Success/failure)
  - Actual Status (from dataset)
  - Confidence and risk explanation


## Preprocessing Steps (Planned)
A future module (preprocess_data.py) will:
1. Spatially align and normalize meteorological inputs
2. Extract derived features (shear magnitudes, ckoud-top temperatures)
3. Label training samples based on launch constreaint thresholds
4. Split data into train/validate/tes

## Results and Discussion (Work in Progress)
The following will be aded after experimentation:
- Accuracy of mission success prediction
- Error analyssi by:
   - Weather domain
   - Company
   - Vehicle type
- Comparison against heuristic baselines
- Confusion matrix and classification report

## Computing Resources
Environment: Python code created in Colab
Hardware: Recommended using a GPU runtime due to vast amount of data
Reproducibility: GitHub version history and use of random-seed control
   - GitHub versioning
   - Fixed random seeds
   - Documented preprocessing pipeline

## Acknowledgements
This work is supported by NASA Nebrasks Space Grant Fellowship. Special thanks to Dr. Steven Fernandes for guidance in AI model design and Dr. Amelia Tangeman for operational insights and data coordination

## References
 - Choucair, C (2025), How Much Does It Cost to Launch a Rocket? [By Type and Size], SpaceInsider. https://spaceinsider.tech/2023/08/16/how-much-does-it-cost-to-launch-a-rocket/
 - Dalal, S. R., Fowlkes, E. B., & Hoadley, B. (1989). Risk Analysis of the Space Shuttle: Pre-Challenger Prediction of Failure. Journal of the American Statistical Association, 84(408), 945–957. https://doi.org/10.1080/01621459.1989.10478858
 - Kreitzberg, C. W. (1979), Observing, analyzing, and modeling mesoscale weather phenomena, Rev. Geophys., 17(7), 1852–1871, doi:10.1029/RG017i007p01852.

