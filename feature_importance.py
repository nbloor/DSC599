"""
Feature Importance Analysis for Mission Success Prediction
Analyzes which variables have the largest impact on mission outcomes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for mission success prediction
    """
    
    def __init__(self, csv_path: str = "mission.csv"):
        """
        Initialize the analyzer with mission data
        """
        self.csv_path = csv_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Models
        self.lr_model = None
        self.rf_model = None
        
        # Results
        self.importance_results = {}
        self.model_metrics = {}
        
        self._load_and_preprocess_data()
    
    def _load_and_preprocess_data(self):
        """Load and preprocess mission data"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✓ Loaded {len(self.data)} missions")
            
            # Create a copy for preprocessing
            df = self.data.copy()
            
            # Target variable
            if 'Mission Status' not in df.columns:
                raise ValueError("'Mission Status' column not found")
            
            # Encode target
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(df['Mission Status'])
            
            # Select features
            feature_columns = [
                'Temperature (° F)', 'Wind speed (MPH)', 'Humidity (%)',
                'Payload Mass (kg)', 'Rocket Height (m)', 'Liftoff Thrust (kN)',
                'Company', 'Vehicle Type', 'Payload Type', 'Payload Orbit'
            ]
            
            # Keep only existing columns
            available_features = [col for col in feature_columns if col in df.columns]
            df_features = df[available_features].copy()
            
            self.feature_names = available_features
            
            # Handle missing values
            for col in df_features.columns:
                if df_features[col].dtype == 'object':
                    df_features[col].fillna('Unknown', inplace=True)
                else:
                    df_features[col].fillna(df_features[col].median(), inplace=True)
            
            # Encode categorical features
            categorical_cols = df_features.select_dtypes(include='object').columns
            for col in categorical_cols:
                le = LabelEncoder()
                df_features[col] = le.fit_transform(df_features[col].astype(str))
                self.label_encoders[col] = le
            
            # Convert to numeric
            X = df_features.astype(float).values
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            print(f"✓ Preprocessed {len(self.feature_names)} features")
            print(f"  Training set: {len(self.X_train)} samples")
            print(f"  Test set: {len(self.X_test)} samples")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def train_logistic_regression(self):
        """Train logistic regression model"""
        try:
            self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
            self.lr_model.fit(self.X_train, self.y_train)
            
            # Evaluate
            y_pred = self.lr_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Get feature importance (coefficients)
            importance = np.abs(self.lr_model.coef_[0])
            importance = importance / importance.sum()  # Normalize
            
            self.importance_results['logistic_regression'] = {
                'scores': importance,
                'model': self.lr_model
            }
            
            self.model_metrics['logistic_regression'] = {
                'accuracy': accuracy,
                'model_type': 'Logistic Regression'
            }
            
            print(f"✓ Logistic Regression trained (Accuracy: {accuracy:.3f})")
            return True
            
        except Exception as e:
            print(f"✗ Error training Logistic Regression: {e}")
            return False
    
    def train_random_forest(self):
        """Train random forest model"""
        try:
            self.rf_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=15
            )
            self.rf_model.fit(self.X_train, self.y_train)
            
            # Evaluate
            y_pred = self.rf_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Get feature importance
            importance = self.rf_model.feature_importances_
            
            self.importance_results['random_forest'] = {
                'scores': importance,
                'model': self.rf_model
            }
            
            self.model_metrics['random_forest'] = {
                'accuracy': accuracy,
                'model_type': 'Random Forest'
            }
            
            print(f"✓ Random Forest trained (Accuracy: {accuracy:.3f})")
            return True
            
        except Exception as e:
            print(f"✗ Error training Random Forest: {e}")
            return False
    
    def calculate_permutation_importance(self):
        """Calculate permutation importance using Random Forest"""
        try:
            if self.rf_model is None:
                print("✗ Random Forest model not trained yet")
                return False
            
            perm_importance = permutation_importance(
                self.rf_model, 
                self.X_test, 
                self.y_test, 
                n_repeats=10, 
                random_state=42,
                n_jobs=-1
            )
            
            # Normalize scores
            importance = perm_importance.importances_mean
            importance = importance / importance.sum()
            
            self.importance_results['permutation'] = {
                'scores': importance,
                'stds': perm_importance.importances_std / perm_importance.importances_std.sum()
            }
            
            print(f"✓ Permutation importance calculated")
            return True
            
        except Exception as e:
            print(f"✗ Error calculating permutation importance: {e}")
            return False
    
    def calculate_shap_importance(self):
        """Calculate SHAP importance values"""
        try:
            if not SHAP_AVAILABLE:
                print("⚠ SHAP not available, skipping")
                return False
            
            if self.rf_model is None:
                print("✗ Random Forest model not trained yet")
                return False
            
            # Create explainer
            explainer = shap.TreeExplainer(self.rf_model)
            shap_values = explainer.shap_values(self.X_test)
            
            # For binary classification, take the shap values for the positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute shap values
            importance = np.abs(shap_values).mean(axis=0)
            importance = importance / importance.sum()
            
            self.importance_results['shap'] = {
                'scores': importance,
                'explainer': explainer,
                'shap_values': shap_values
            }
            
            print(f"✓ SHAP importance calculated")
            return True
            
        except Exception as e:
            print(f"✗ Error calculating SHAP importance: {e}")
            return False
    
    def calculate_correlation_analysis(self):
        """Analyze correlation with mission success"""
        try:
            df = self.data.copy()
            
            # Encode target
            target_encoded = self.target_encoder.transform(df['Mission Status'])
            
            # Select numeric features
            numeric_cols = [col for col in self.feature_names if col in df.columns]
            df_numeric = df[numeric_cols].copy()
            
            # Handle missing and convert to numeric
            for col in df_numeric.columns:
                if df_numeric[col].dtype == 'object':
                    le = LabelEncoder()
                    df_numeric[col] = pd.to_numeric(
                        le.fit_transform(df_numeric[col].astype(str)), 
                        errors='coerce'
                    )
                else:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
            
            # Calculate correlations
            correlations = []
            for col in df_numeric.columns:
                corr = np.corrcoef(df_numeric[col].fillna(df_numeric[col].mean()), target_encoded)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            # Normalize
            importance = np.array(correlations)
            importance = importance / importance.sum()
            
            self.importance_results['correlation'] = {
                'scores': importance,
                'raw_correlations': correlations
            }
            
            print(f"✓ Correlation analysis completed")
            return True
            
        except Exception as e:
            print(f"✗ Error in correlation analysis: {e}")
            return False
    
    def train_all_models(self):
        """Train all models"""
        print("\n" + "="*50)
        print("Training Feature Importance Models")
        print("="*50)
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.calculate_permutation_importance()
        self.calculate_shap_importance()
        self.calculate_correlation_analysis()
        
        print("="*50)
    
    def get_feature_importance_summary(self) -> Dict:
        """Get summary of feature importances from all models"""
        summary = {}
        
        for model_name, result in self.importance_results.items():
            if model_name == 'shap':
                continue  # Skip SHAP for summary
            
            scores = result['scores']
            ranked = sorted(
                zip(self.feature_names, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            summary[model_name] = {
                'top_features': [feat for feat, _ in ranked[:5]],
                'top_scores': [score for _, score in ranked[:5]],
                'all_features': [feat for feat, _ in ranked],
                'all_scores': [score for _, score in ranked]
            }
        
        return summary
    
    def get_ranked_features(self) -> Dict:
        """Get features ranked by average importance across models"""
        if not self.importance_results:
            return {}
        
        # Average scores across available models
        all_scores = {}
        model_count = 0
        
        for model_name, result in self.importance_results.items():
            if model_name == 'shap':
                continue
            
            scores = result['scores']
            for feat, score in zip(self.feature_names, scores):
                if feat not in all_scores:
                    all_scores[feat] = 0
                all_scores[feat] += score
            model_count += 1
        
        # Average
        for feat in all_scores:
            all_scores[feat] /= model_count
        
        # Rank
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'features': [feat for feat, _ in ranked],
            'scores': [score for _, score in ranked],
            'models_used': model_count
        }
    
    def create_importance_visualization(self, figsize=(14, 10)) -> plt.Figure:
        """Create comprehensive visualization of feature importance"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Feature Importance Analysis - Mission Success Prediction', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Logistic Regression
        if 'logistic_regression' in self.importance_results:
            ax = axes[0, 0]
            lr_scores = self.importance_results['logistic_regression']['scores']
            ranking = np.argsort(lr_scores)[-10:]
            
            ax.barh(np.array(self.feature_names)[ranking], lr_scores[ranking], color='steelblue')
            ax.set_xlabel('Normalized Coefficient')
            ax.set_title('Logistic Regression Coefficients')
            ax.grid(axis='x', alpha=0.3)
        
        # Plot 2: Random Forest
        if 'random_forest' in self.importance_results:
            ax = axes[0, 1]
            rf_scores = self.importance_results['random_forest']['scores']
            ranking = np.argsort(rf_scores)[-10:]
            
            ax.barh(np.array(self.feature_names)[ranking], rf_scores[ranking], color='seagreen')
            ax.set_xlabel('Feature Importance')
            ax.set_title('Random Forest Importance')
            ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: Permutation Importance
        if 'permutation' in self.importance_results:
            ax = axes[1, 0]
            perm_scores = self.importance_results['permutation']['scores']
            ranking = np.argsort(perm_scores)[-10:]
            
            stds = self.importance_results['permutation']['stds'][ranking]
            ax.barh(np.array(self.feature_names)[ranking], perm_scores[ranking], 
                   xerr=stds, color='coral', capsize=5)
            ax.set_xlabel('Importance')
            ax.set_title('Permutation Importance (with std)')
            ax.grid(axis='x', alpha=0.3)
        
        # Plot 4: Average Ranking
        ax = axes[1, 1]
        ranked = self.get_ranked_features()
        if ranked:
            top_n = min(10, len(ranked['features']))
            top_features = ranked['features'][:top_n]
            top_scores = ranked['scores'][:top_n]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            ax.barh(top_features, top_scores, color=colors)
            ax.set_xlabel('Average Importance Score')
            ax.set_title('Top 10 Features (Averaged)')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_model_comparison_text(self) -> str:
        """Get text summary of model accuracies"""
        if not self.model_metrics:
            return "No models trained yet"
        
        text = "## Model Accuracy Comparison\n\n"
        for model_name, metrics in self.model_metrics.items():
            text += f"**{metrics['model_type']}**: {metrics['accuracy']:.1%}\n"
        
        return text
    
    def export_results(self, output_path: str = "importance_analysis.csv"):
        """Export importance analysis to CSV"""
        try:
            ranked = self.get_ranked_features()
            
            df_export = pd.DataFrame({
                'Feature': ranked['features'],
                'Importance_Score': ranked['scores']
            })
            
            df_export.to_csv(output_path, index=False)
            print(f"✓ Results exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error exporting results: {e}")
            return False


# Main execution for testing
if __name__ == "__main__":
    print("Initializing Feature Importance Analyzer...")
    analyzer = FeatureImportanceAnalyzer()
    
    print("\nTraining models...")
    analyzer.train_all_models()
    
    print("\nFeature Importance Summary:")
    summary = analyzer.get_feature_importance_summary()
    for model_name, info in summary.items():
        print(f"\n{model_name.upper()}:")
        for feat, score in zip(info['top_features'], info['top_scores']):
            print(f"  {feat}: {score:.4f}")
    
    print("\nOverall Top Features:")
    ranked = analyzer.get_ranked_features()
    for feat, score in zip(ranked['features'][:5], ranked['scores'][:5]):
        print(f"  {feat}: {score:.4f}")
    
    print("\nModel Accuracies:")
    print(analyzer.get_model_comparison_text())
    
    print("\nCreating visualization...")
    fig = analyzer.create_importance_visualization()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'feature_importance_analysis.png'")
    
    print("\nExporting results...")
    analyzer.export_results()
