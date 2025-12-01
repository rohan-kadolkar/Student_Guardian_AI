"""
PREDICTION & ANALYTICS
Generate predictions and comprehensive student analytics
"""

import pandas as pd
import numpy as np
import joblib
import json


class StudentAnalytics:
    def __init__(self, model_dir='models'):
        """Load trained model and preprocessors"""
        self.model = joblib.load(f'{model_dir}/dropout_model.pkl')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        self.encoders = joblib.load(f'{model_dir}/encoders.pkl')
        self.feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
        self.label_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        
    def predict_student(self, student_data):
        """
        Predict dropout risk and generate analytics for a single student
        
        Args:
            student_data: dict or DataFrame row with student features
            
        Returns:
            dict with predictions and analytics
        """
        # Convert to DataFrame if dict
        if isinstance(student_data, dict):
            df = pd.DataFrame([student_data])
        else:
            df = pd.DataFrame([student_data.to_dict()])
        
        student_id = df['student_id'].values[0]
        
        # Prepare features (same as training)
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        engineer.encoders = self.encoders
        engineer.scaler = self.scaler
        
        df = engineer.engineer_features(df)
        
        # Select and encode features
        X = df[self.feature_names]
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col in self.encoders:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        X.fillna(0, inplace=True)
        
        # Scale
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        dropout_risk = self.label_mapping[prediction]
        risk_confidence = probabilities[prediction] * 100
        
        # Generate analytics
        analytics = self._generate_analytics(df.iloc[0])
        
        # Compile results
        result = {
            'student_id': int(student_id),
            'dropout_risk': dropout_risk,
            'risk_confidence': round(risk_confidence, 2),
            'risk_probabilities': {
                'Low Risk': round(probabilities[0] * 100, 2),
                'Medium Risk': round(probabilities[1] * 100, 2),
                'High Risk': round(probabilities[2] * 100, 2)
            },
            'learning_style': analytics['learning_style'],
            'strengths': analytics['strengths'],
            'weaknesses': analytics['weaknesses'],
            'interests': analytics['interests'],
            'recommendations': self._generate_recommendations(dropout_risk, analytics)
        }
        
        return result
    
    def _generate_analytics(self, student):
        """Generate individual student analytics"""
        
        # 1. Learning Style (inferred from scores)
        learning_scores = {
            'Visual': student.get('learning_visual_score', 0),
            'Reading/Writing': student.get('learning_reading_score', 0),
            'Kinesthetic': student.get('learning_kinesthetic_score', 0),
            'Auditory': student.get('learning_auditory_score', 0)
        }
        learning_style = max(learning_scores, key=learning_scores.get)
        
        # 2. Academic Strengths (top performing subjects)
        subject_cols = [col for col in student.index if col.startswith('marks_subject_')]
        if subject_cols:
            subject_scores = {col.replace('marks_subject_', '').replace('_', ' ').title(): 
                            student[col] for col in subject_cols}
            sorted_subjects = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
            strengths = [subj for subj, score in sorted_subjects[:3] if score > 60]
        else:
            strengths = []
        
        # Add other strengths
        if student.get('attendance_percentage', 0) >= 85:
            strengths.append('Excellent Attendance')
        if student.get('assignment_submission_rate', 0) >= 90:
            strengths.append('Assignment Completion')
        if student.get('extra_leadership_roles', 0) > 0:
            strengths.append('Leadership')
        
        # 3. Weaknesses
        weaknesses = []
        
        # Low performing subjects
        if subject_cols:
            weak_subjects = [subj for subj, score in sorted_subjects[-2:] if score < 50]
            weaknesses.extend(weak_subjects)
        
        # Low attendance
        if student.get('attendance_percentage', 100) < 75:
            weaknesses.append('Low Attendance')
        
        # Low assignment completion
        if student.get('assignment_submission_rate', 100) < 70:
            weaknesses.append('Assignment Delays')
        
        # Fee issues
        if student.get('fee_pending_count', 0) > 0:
            weaknesses.append('Pending Fees')
        
        # No extracurricular
        if student.get('extra_participates', 1) == 0:
            weaknesses.append('No Extracurricular Activities')
        
        # 4. Interests (from extracurricular activities)
        interests = []
        
        # Check activity categories
        activity_cats = ['Technical', 'Sports', 'Cultural', 'Social', 'Academic']
        for cat in activity_cats:
            col = f'extra_category_{cat}'
            if student.get(col, 0) > 0:
                interests.append(cat)
        
        # Check library usage
        if student.get('library_visits', 0) > 10:
            interests.append('Reading/Research')
        
        # Check specific subjects
        if student.get('marks_subject_computer_science', 0) > 75:
            interests.append('Computer Science')
        
        return {
            'learning_style': learning_style,
            'strengths': strengths[:5],  # Top 5
            'weaknesses': weaknesses[:5],  # Top 5
            'interests': interests[:5]  # Top 5
        }
    
    def _generate_recommendations(self, risk_level, analytics):
        """Generate personalized recommendations"""
        recommendations = []
        
        if risk_level == 'High Risk':
            recommendations.append("🚨 URGENT: Schedule immediate counseling session")
            recommendations.append("👥 Assign dedicated faculty mentor")
            recommendations.append("📞 Contact family regarding support")
        
        if risk_level in ['High Risk', 'Medium Risk']:
            recommendations.append("📚 Recommend tutoring for weak subjects")
            recommendations.append("⏰ Set weekly check-in meetings")
        
        # Specific recommendations based on weaknesses
        if 'Low Attendance' in analytics['weaknesses']:
            recommendations.append("📅 Address attendance issues - provide flexible options")
        
        if 'No Extracurricular Activities' in analytics['weaknesses']:
            recommendations.append("🎯 Encourage joining 1-2 clubs/activities")
        
        if 'Pending Fees' in analytics['weaknesses']:
            recommendations.append("💰 Discuss financial aid/scholarship options")
        
        # Leverage strengths
        if analytics['strengths']:
            recommendations.append(f"💪 Leverage strengths: {', '.join(analytics['strengths'][:2])}")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def batch_predict(self, df):
        """Predict for multiple students"""
        results = []
        
        for idx, student in df.iterrows():
            try:
                result = self.predict_student(student)
                results.append(result)
            except Exception as e:
                print(f"⚠️ Error processing student {student.get('student_id', idx)}: {e}")
                continue
        
        return results


if __name__ == "__main__":
    # Load test data
    df = pd.read_csv('processed_data.csv')
    
    # Initialize analytics
    analytics = StudentAnalytics(model_dir='models')
    
    # Predict for all students
    print("🔮 Generating predictions for all students...")
    results = analytics.batch_predict(df)
    
    # Save results
    with open('student_analytics_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Predictions complete: {len(results)} students")
    print(f"💾 Results saved: student_analytics_results.json")
    
    # Display sample
    print("\n📋 Sample Results:")
    for i in range(min(3, len(results))):
        print(f"\n{'-'*60}")
        print(json.dumps(results[i], indent=2))