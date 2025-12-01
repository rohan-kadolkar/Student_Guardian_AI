"""
MODEL EXPLAINABILITY
SHAP values for explaining predictions
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


class ModelExplainer:
    def __init__(self, model_dir='models'):
        self.model = joblib.load(f'{model_dir}/dropout_model.pkl')
        self.feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
        
    def explain_model(self, X, sample_size=100):
        """Generate SHAP explanations for model"""
        print("\n" + "="*80)
        print("🔍 MODEL EXPLAINABILITY (SHAP)")
        print("="*80)
        
        # Sample for faster computation
        X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
        
        print(f"\n📊 Computing SHAP values for {len(X_sample)} samples...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Global feature importance
        self._plot_global_importance(shap_values, X_sample)
        
        # Summary plot
        self._plot_summary(shap_values, X_sample)
        
        return explainer, shap_values
    
    def explain_student(self, student_data, explainer=None):
        """Explain prediction for individual student"""
        if explainer is None:
            explainer = shap.TreeExplainer(self.model)
        
        shap_values = explainer.shap_values(student_data)
        
        # Force plot
        shap.force_plot(
            explainer.expected_value[1],  # Medium Risk class
            shap_values[1],
            student_data,
            matplotlib=True,
            show=False
        )
        plt.savefig('models/student_explanation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Individual explanation saved: models/student_explanation.png")
    
    def _plot_global_importance(self, shap_values, X):
        """Plot global feature importance"""
        # Average absolute SHAP values across all classes
        mean_shap = np.abs(shap_values).mean(axis=0).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Top 20 Features (SHAP Importance)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('models/shap_importance.png', dpi=300)
        plt.close()
        
        print(f"💾 SHAP importance plot saved: models/shap_importance.png")
    
    def _plot_summary(self, shap_values, X):
        """Plot SHAP summary"""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[1], X, show=False)  # Medium Risk class
        plt.tight_layout()
        plt.savefig('models/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 SHAP summary plot saved: models/shap_summary.png")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('processed_data.csv')
    
    # Prepare features (use same preprocessing as training)
    from train_model import DropoutModel
    model = DropoutModel()
    X, y = model.prepare_data(df)
    
    # Explain model
    explainer_obj = ModelExplainer(model_dir='models')
    explainer, shap_values = explainer_obj.explain_model(X, sample_size=200)
    
    # Explain first student
    print("\n🔬 Explaining first student prediction...")
    explainer_obj.explain_student(X.iloc[:1], explainer)
    
    print("\n✅ Explainability analysis complete!")