# Agentic AI-Based Forecasting for Enhanced Space Mission Safety

A neural network–based predictive framework for improving the safety and reliability of space launch operations by forecasting short-term atmospheric conditions such as wind shear, lightning, cloud cover, and precipitation.

---

## Overview

This project investigates the use of **agentic AI systems** and **neural network architectures** to predict launch-critical environmental constraints.  
By fusing multi-source meteorological data — including radiosonde soundings, radar, satellite imagery, and surface observations — the system produces **probabilistic, high-resolution forecasts** that enhance decision-making under launch commit criteria (LCCs).

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
└── README.md              # Project documentation
```

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- GPU-enabled environment recommended
- (Future) Framework support: PyTorch or TensorFlow

### 1. Clone Repository

```bash
git clone https://github.com/<username>/agentic-ai-launch-forecasting.git
cd agentic-ai-launch-forecasting
```

### 2. Create Environment

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate
```

### 3. Install Dependencies
Once the model framework has been selected, the dependencies will be finalized.

### 4. Run Data Preprocessing
Once the preprocessing Python has been completed, it will be updated here. The preprocessing is currently a work in progress.
The file will be named preprocess_data.py

### 5. Train Model
Once the training code is finalized, it will be updated here.
The file will be named train_model.py

### 6. Generate Forecasts
Once the final code is done, it will be placed here. 
The file will be named predict.py

## Preprocessing Steps

1. Spatially align and normalize meteorological inputs
2. Extract derived features (shear magnitudes, ckoud-top temperatures)
3. Label training samples based on launch constreaint thresholds
4. Split data into train/validate/tes

## Results and Discussion
This will be updated once results have been identified. This is currently a work in progress.

## Computing Resources
Environment: Python code created in Colab
Hardware: Recommended using a GPU runtime due to vast amount of data
Reproducibility: GitHub version history and use of random-seed control

## Acknowledgements
This work is supported by NASA Nebrasks Space Grant Fellowship. Special thanks to Dr. Steven Fernandes for guidance in AI model design and Dr. Amelia Tangeman for operational insights and data coordination

## References
 - Choucair, C (2025), How Much Does It Cost to Launch a Rocket? [By Type and Size], SpaceInsider. https://spaceinsider.tech/2023/08/16/how-much-does-it-cost-to-launch-a-rocket/
 - Dalal, S. R., Fowlkes, E. B., & Hoadley, B. (1989). Risk Analysis of the Space Shuttle: Pre-Challenger Prediction of Failure. Journal of the American Statistical Association, 84(408), 945–957. https://doi.org/10.1080/01621459.1989.10478858
 - Kreitzberg, C. W. (1979), Observing, analyzing, and modeling mesoscale weather phenomena, Rev. Geophys., 17(7), 1852–1871, doi:10.1029/RG017i007p01852.

