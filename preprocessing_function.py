import pandas as pd
import json

class PreprocessingFunction:
    def __init__(self, df=None):
        self.df = df
        self.feature_columns = None
        
    def preprocess(self):
        """Preprocess training data"""
        df = self.df.copy()  # Work with a copy
        
        # Binary encoding
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)
        df['employment_type'] = df['employment_type'].apply(lambda x: 1 if x == 'Full-time' else 0)
        df['admin_support'] = df['admin_support'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['resource_availability'] = df['resource_availability'].apply(lambda x: 1 if x == 'Low' else 2 if x == 'Medium' else 3 if x == 'High' else 0)

        # One-Hot encoding
        df = df.join(pd.get_dummies(df['education_level'], prefix='Education_Level'))
        df = df.join(pd.get_dummies(df['subject'], prefix='Subject'))

        # Date processing
        df['date_of_hire'] = pd.to_datetime(df['date_of_hire'], format='%m/%d/%Y', errors='coerce')
        df['date_of_last_eval'] = pd.to_datetime(df['date_of_last_eval'], format='%m/%d/%Y', errors='coerce')

        # Calculate tenure in years
        df['tenure_years'] = (df['date_of_last_eval'] - df['date_of_hire']).dt.days / 365

        # Feature engineering
        df['performance_per_load'] = df['perf_score'] / (df['workload'] + 1e-6)
        df['outcomes_per_load'] = df['student_outcomes'] / (df['workload'] + 1e-6)

        # Cleanup
        df = df.drop(columns=['education_level', 'subject', 'teacher_id', 'date_of_hire', 'date_of_last_eval', 'time_to_event'], errors='ignore')

        # Booleans -> 0/1
        df = df.map(lambda x: 1 if x is True else 0 if x is False else x)

        # Basic NA handling
        df = df.fillna(0)
        
        # Store feature columns for later use
        if 'performance_drop' in df.columns:
            self.feature_columns = [col for col in df.columns if col != 'performance_drop']
        else:
            self.feature_columns = list(df.columns)
            
        return df
    
    def save_feature_columns(self, filepath='feature_columns.json'):
        """Save feature columns to JSON file"""
        if self.feature_columns is None:
            raise ValueError("No feature columns to save. Run preprocess() first.")
        
        with open(filepath, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        print(f"✅ Feature columns saved to: {filepath}")
    
    @staticmethod
    def preprocess_for_prediction(data_dict, feature_columns):
        """
        Preprocess user input data for model prediction
        Args:
            data_dict: Dictionary with user input data
            feature_columns: List of required feature columns from training
        Returns:
            DataFrame ready for model prediction
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Binary encoding with error handling
        df['gender'] = df['gender'].apply(lambda x: 1 if str(x).upper() == 'M' else 0)
        df['employment_type'] = df['employment_type'].apply(lambda x: 1 if str(x) == 'Full-time' else 0)
        df['admin_support'] = df['admin_support'].apply(lambda x: 1 if str(x).upper() == 'YES' else 0)
        
        # Ordinal encoding for resource availability
        resource_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['resource_availability'] = df['resource_availability'].map(resource_map).fillna(0)
        
        # One-hot encoding for education and subject
        if 'education_level' in df.columns:
            education_dummies = pd.get_dummies(df['education_level'], prefix='Education_Level')
            df = pd.concat([df, education_dummies], axis=1)
        
        if 'subject' in df.columns:
            subject_dummies = pd.get_dummies(df['subject'], prefix='Subject')
            df = pd.concat([df, subject_dummies], axis=1)
        
        # Date processing
        if 'date_of_hire' in df.columns and 'date_of_last_eval' in df.columns:
            df['date_of_hire'] = pd.to_datetime(df['date_of_hire'], format='%m/%d/%Y', errors='coerce')
            df['date_of_last_eval'] = pd.to_datetime(df['date_of_last_eval'], format='%m/%d/%Y', errors='coerce')
            
            # Calculate tenure
            df['tenure_years'] = (df['date_of_last_eval'] - df['date_of_hire']).dt.days / 365
        
        # Feature engineering
        if 'perf_score' in df.columns and 'workload' in df.columns:
            df['performance_per_load'] = df['perf_score'] / (df['workload'] + 1e-6)
        
        if 'student_outcomes' in df.columns and 'workload' in df.columns:
            df['outcomes_per_load'] = df['student_outcomes'] / (df['workload'] + 1e-6)
        
        # Cleanup
        df = df.drop(columns=['education_level', 'subject', 'date_of_hire', 'date_of_last_eval'], errors='ignore')
        
        # Handle booleans
        df = df.map(lambda x: 1 if x is True else 0 if x is False else x)
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the required columns in the correct order
        df = df[feature_columns]
        
        # Fill any remaining NAs
        df = df.fillna(0)
        
        return df
    
    @staticmethod
    def load_feature_columns(filepath='feature_columns.json'):
        """Load feature columns from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)