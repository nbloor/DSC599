"""
Mission Success/Failure Prediction Agent using Llama 3.2
An agentic AI system that analyzes mission parameters and predicts outcomes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import re
from huggingface_hub import InferenceClient

class MissionAnalysisAgent:
    """
    Agentic AI system for mission success/failure prediction using Llama 3.2
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize the agent with Llama 3.2 model
        """
        self.model_name = model_name
        self.client = InferenceClient(model=model_name)
        self.historical_data = None
        self.failure_patterns = {}
        self.success_patterns = {}
        
    def load_historical_data(self, csv_path: str):
        """
        Load and preprocess historical mission data
        """
        self.historical_data = pd.read_csv(csv_path)
        self._analyze_patterns()
        return self
    
    def _analyze_patterns(self):
        """
        Analyze historical patterns for failures and successes
        """
        if self.historical_data is None:
            return
        
        # Separate success and failure cases
        failures = self.historical_data[self.historical_data['Mission Status'] == 'Failure']
        successes = self.historical_data[self.historical_data['Mission Status'] == 'Success']
        
        # Analyze failure patterns
        self.failure_patterns = {
            'common_reasons': failures['Failure Reason'].value_counts().to_dict() if 'Failure Reason' in failures.columns else {},
            'companies': failures['Company'].value_counts().to_dict(),
            'vehicle_types': failures['Vehicle Type'].value_counts().to_dict(),
            'weather_conditions': self._analyze_weather_patterns(failures),
            'payload_types': failures['Payload Type'].value_counts().to_dict()
        }
        
        # Analyze success patterns
        self.success_patterns = {
            'companies': successes['Company'].value_counts().to_dict(),
            'vehicle_types': successes['Vehicle Type'].value_counts().to_dict(),
            'weather_conditions': self._analyze_weather_patterns(successes),
            'payload_types': successes['Payload Type'].value_counts().to_dict()
        }
    
    def _analyze_weather_patterns(self, df):
        """
        Analyze weather patterns in missions
        """
        weather_stats = {}
        
        # Clean numeric columns
        for col in ['Temperature (° F)', 'Wind speed (MPH)', 'Humidity (%)']:
            if col in df.columns:
                cleaned = pd.to_numeric(df[col], errors='coerce')
                weather_stats[col] = {
                    'mean': cleaned.mean(),
                    'std': cleaned.std(),
                    'min': cleaned.min(),
                    'max': cleaned.max()
                }
        
        return weather_stats
    
    def _extract_features(self, mission_data: Dict) -> Dict:
        """
        Extract relevant features from mission data
        """
        features = {
            'company': mission_data.get('company', ''),
            'vehicle_type': mission_data.get('vehicle_type', ''),
            'launch_site': mission_data.get('launch_site', ''),
            'temperature': self._safe_float(mission_data.get('temperature')),
            'wind_speed': self._safe_float(mission_data.get('wind_speed')),
            'humidity': self._safe_float(mission_data.get('humidity')),
            'payload_mass': self._safe_float(mission_data.get('payload_mass')),
            'payload_type': mission_data.get('payload_type', ''),
            'payload_orbit': mission_data.get('payload_orbit', ''),
            'rocket_height': self._safe_float(mission_data.get('rocket_height')),
            'liftoff_thrust': self._safe_float(mission_data.get('liftoff_thrust'))
        }
        return features
    
    def _safe_float(self, value):
        """
        Safely convert value to float
        """
        if value is None or value == 'NA' or value == '':
            return None
        try:
            return float(value)
        except:
            return None
    
    def _generate_risk_assessment_prompt(self, features: Dict) -> str:
        """
        Generate a comprehensive prompt for Llama 3.2 to assess mission risk
        """
        prompt = f"""You are an expert aerospace mission analyst. Analyze this upcoming space mission and predict its likelihood of success.
MISSION PARAMETERS:
- Company: {features['company']}
- Vehicle Type: {features['vehicle_type']}
- Launch Site: {features['launch_site']}
- Weather Conditions:
  * Temperature: {features['temperature']}°F
  * Wind Speed: {features['wind_speed']} MPH
  * Humidity: {features['humidity']}%
- Payload:
  * Type: {features['payload_type']}
  * Mass: {features['payload_mass']} kg
  * Target Orbit: {features['payload_orbit']}
- Vehicle Specifications:
  * Height: {features['rocket_height']} m
  * Liftoff Thrust: {features['liftoff_thrust']} kN
HISTORICAL CONTEXT:
- This company has had {self._get_company_history(features['company'])}
- Common failure reasons in similar missions include: {self._get_relevant_failure_reasons()}
- Weather-related risks: {self._assess_weather_risk(features)}
Based on this analysis, provide:
1. SUCCESS_PROBABILITY: (0-100)
2. KEY_RISK_FACTORS: (list top 3)
3. RECOMMENDATION: (PROCEED/CAUTION/POSTPONE)
4. REASONING: (brief explanation)
Respond in JSON format."""

        return prompt
    
    def _get_company_history(self, company: str) -> str:
        """
        Get historical performance of the company
        """
        if self.historical_data is None or not company:
            return "no historical data available"
        
        company_data = self.historical_data[self.historical_data['Company'] == company]
        if company_data.empty:
            return "no previous missions"
        
        total = len(company_data)
        successes = len(company_data[company_data['Mission Status'] == 'Success'])
        failures = total - successes
        
        return f"{successes} successes and {failures} failures out of {total} missions"
    
    def _get_relevant_failure_reasons(self) -> str:
        """
        Get most common failure reasons
        """
        if not self.failure_patterns.get('common_reasons'):
            return "No historical failure data"
        
        top_reasons = list(self.failure_patterns['common_reasons'].keys())[:3]
        return ", ".join(top_reasons) if top_reasons else "Various technical issues"
    
    def _assess_weather_risk(self, features: Dict) -> str:
        """
        Assess weather-related risks
        """
        risks = []
        
        if features['temperature'] is not None:
            if features['temperature'] < 40 or features['temperature'] > 95:
                risks.append("Extreme temperature")
        
        if features['wind_speed'] is not None:
            if features['wind_speed'] > 20:
                risks.append("High wind speed")
        
        if features['humidity'] is not None:
            if features['humidity'] > 90:
                risks.append("Very high humidity")
        
        return ", ".join(risks) if risks else "Weather conditions appear normal"
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse the LLM response to extract prediction data
        """
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                result = {
                    'SUCCESS_PROBABILITY': 50,
                    'KEY_RISK_FACTORS': ['Unable to parse response'],
                    'RECOMMENDATION': 'CAUTION',
                    'REASONING': response[:200]
                }
                return result
        except Exception as e:
            return {
                'SUCCESS_PROBABILITY': 50,
                'KEY_RISK_FACTORS': ['Analysis error'],
                'RECOMMENDATION': 'CAUTION',
                'REASONING': f'Error parsing response: {str(e)}'
            }
    
    def predict(self, mission_data: Dict) -> Dict:
        """
        Main prediction method using the agentic approach
        """
        # Extract features
        features = self._extract_features(mission_data)
        
        # Generate risk assessment prompt
        prompt = self._generate_risk_assessment_prompt(features)
        
        try:
            # Get prediction from Llama 3.2
            response = self.client.text_generation(
                prompt,
                max_new_tokens=500,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Parse the response
            prediction = self._parse_llm_response(response)
            
            # Add binary prediction based on probability
            prediction['PREDICTED_STATUS'] = 'Success' if prediction['SUCCESS_PROBABILITY'] >= 60 else 'Failure'
            
            # Add confidence level
            prob = prediction['SUCCESS_PROBABILITY']
            if prob >= 80 or prob <= 20:
                prediction['CONFIDENCE'] = 'High'
            elif prob >= 65 or prob <= 35:
                prediction['CONFIDENCE'] = 'Medium'
            else:
                prediction['CONFIDENCE'] = 'Low'
            
            return prediction
            
        except Exception as e:
            # Fallback prediction using simple heuristics
            return self._fallback_prediction(features, str(e))
    
    def _fallback_prediction(self, features: Dict, error_msg: str) -> Dict:
        """
        Fallback prediction using simple heuristics when LLM fails
        """
        risk_score = 50  # Start with neutral score
        risk_factors = []
        
        # Company risk assessment
        if features['company'] in self.failure_patterns.get('companies', {}):
            company_failures = self.failure_patterns['companies'][features['company']]
            company_successes = self.success_patterns['companies'].get(features['company'], 0)
            if company_failures > company_successes:
                risk_score -= 15
                risk_factors.append("Company has higher failure rate")
        
        # Weather risk assessment
        if features['temperature'] is not None:
            if features['temperature'] < 40 or features['temperature'] > 95:
                risk_score -= 10
                risk_factors.append("Extreme temperature conditions")
        
        if features['wind_speed'] is not None and features['wind_speed'] > 20:
            risk_score -= 10
            risk_factors.append("High wind speed")
        
        # Vehicle type assessment
        if features['vehicle_type'] in self.failure_patterns.get('vehicle_types', {}):
            vehicle_failures = self.failure_patterns['vehicle_types'][features['vehicle_type']]
            vehicle_successes = self.success_patterns['vehicle_types'].get(features['vehicle_type'], 0)
            if vehicle_failures > vehicle_successes:
                risk_score -= 10
                risk_factors.append("Vehicle type has reliability issues")
        
        return {
            'SUCCESS_PROBABILITY': max(0, min(100, risk_score)),
            'KEY_RISK_FACTORS': risk_factors[:3] if risk_factors else ['Standard mission parameters'],
            'RECOMMENDATION': 'PROCEED' if risk_score >= 60 else ('CAUTION' if risk_score >= 40 else 'POSTPONE'),
            'REASONING': f'Fallback heuristic analysis due to: {error_msg}',
            'PREDICTED_STATUS': 'Success' if risk_score >= 50 else 'Failure',
            'CONFIDENCE': 'Low'
        }
    
    def batch_predict(self, missions: List[Dict]) -> List[Dict]:
        """
        Predict multiple missions
        """
        predictions = []
        for mission in missions:
            prediction = self.predict(mission)
            predictions.append(prediction)
        return predictions
    
    def get_model_insights(self) -> Dict:
        """
        Get insights about the model's understanding
        """
        return {
            'model': self.model_name,
            'total_historical_missions': len(self.historical_data) if self.historical_data is not None else 0,
            'failure_patterns': self.failure_patterns,
            'success_patterns': self.success_patterns,
            'capabilities': [
                'Weather risk assessment',
                'Company reliability analysis',
                'Vehicle type evaluation',
                'Payload compatibility check',
                'Historical pattern recognition',
                'Multi-factor risk scoring'
            ]
        }
