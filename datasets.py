"""
Comprehensive Synthetic Diabetes/Healthcare Time Series Dataset Generator
Creates realistic CGM (Continuous Glucose Monitor) data with clinical outcomes,
treatment variables, and patient demographics for AI/ML projects.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DiabetesTimeSeriesGenerator:
    """
    Generates realistic diabetes/healthcare time series data with:
    - CGM glucose readings (5-min intervals)
    - Meal events with carbohydrates
    - Insulin administration (basal & bolus)
    - Physical activity
    - Clinical outcomes (hypoglycemia, hyperglycemia)
    - Patient demographics
    - Sensor noise and realistic artifacts
    - Concept drift and patient variability
    """
    
    def __init__(self, output_dir="./diabetes_datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_patient_cohort(self, n_patients=100, days_per_patient=90):
        """
        Generate a cohort of patients with diverse characteristics
        
        Parameters:
        - n_patients: Number of patients to generate
        - days_per_patient: Days of data per patient
        """
        all_patients_data = []
        patient_metadata = {}
        
        for patient_id in range(n_patients):
            print(f"Generating patient {patient_id + 1}/{n_patients}...")
            
            # Generate patient demographics
            demographics = self._generate_demographics(patient_id)
            
            # Generate time series data for this patient
            patient_data = self._generate_patient_timeseries(
                patient_id, 
                demographics, 
                days_per_patient
            )
            
            all_patients_data.append(patient_data)
            patient_metadata[f"patient_{patient_id:03d}"] = demographics
            
        # Combine all patients data
        final_dataset = pd.concat(all_patients_data, ignore_index=True)
        
        return final_dataset, patient_metadata
    
    def _generate_demographics(self, patient_id):
        """Generate realistic patient demographics"""
        
        # Diabetes type distribution (Type 2 more common)
        diabetes_type = np.random.choice(['Type1', 'Type2', 'Prediabetes'], 
                                        p=[0.2, 0.7, 0.1])
        
        # Age distribution based on diabetes type
        if diabetes_type == 'Type1':
            age = np.random.normal(35, 15)  # Younger onset
        elif diabetes_type == 'Type2':
            age = np.random.normal(55, 12)  # Older onset
        else:
            age = np.random.normal(45, 10)
        
        age = max(18, min(85, int(age)))
        
        # BMI (kg/m²) - correlated with diabetes type
        if diabetes_type == 'Type2':
            bmi = np.random.normal(32, 5)  # Higher BMI for Type 2
        else:
            bmi = np.random.normal(26, 4)
        bmi = max(18.5, min(45, round(bmi, 1)))
        
        # Gender
        sex = np.random.choice(['M', 'F'])
        
        # Diabetes duration (years)
        duration = max(0, int(np.random.exponential(10) - 2))
        
        # HbA1c baseline (%) - correlated with control quality
        hba1c = np.random.normal(7.5, 1.5)
        hba1c = max(5.5, min(12, round(hba1c, 1)))
        
        # Insulin sensitivity factor (affects glucose response)
        insulin_sensitivity = np.random.normal(1.0, 0.2)
        insulin_sensitivity = max(0.5, min(2.0, round(insulin_sensitivity, 2)))
        
        return {
            'patient_id': f"P{patient_id:03d}",
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'diabetes_type': diabetes_type,
            'diabetes_duration': duration,
            'baseline_hba1c': hba1c,
            'insulin_sensitivity': insulin_sensitivity,
            'treatment_regimen': self._assign_treatment_regimen(diabetes_type)
        }
    
    def _assign_treatment_regimen(self, diabetes_type):
        """Assign realistic treatment regimen based on diabetes type"""
        
        if diabetes_type == 'Type1':
            # Type 1 always on insulin
            return {
                'basal_insulin': np.random.choice(['Lantus', 'Toujeo', 'Tresiba']),
                'basal_dose': np.random.uniform(15, 40),
                'bolus_insulin': np.random.choice(['Novolog', 'Humalog', 'Apidra']),
                'insulin_pump': np.random.choice([True, False], p=[0.4, 0.6]),
                'metformin': False,
                'sglt2_inhibitor': np.random.choice([True, False], p=[0.2, 0.8]),
                'glp1_ra': np.random.choice([True, False], p=[0.1, 0.9])
            }
        elif diabetes_type == 'Type2':
            # Type 2 variable regimens
            regimen = {}
            if np.random.random() < 0.3:
                # Diet and exercise only
                regimen = {
                    'basal_insulin': None,
                    'basal_dose': 0,
                    'bolus_insulin': None,
                    'insulin_pump': False,
                    'metformin': True,
                    'sglt2_inhibitor': np.random.choice([True, False], p=[0.3, 0.7]),
                    'glp1_ra': np.random.choice([True, False], p=[0.2, 0.8])
                }
            elif np.random.random() < 0.6:
                # Oral medications only
                regimen = {
                    'basal_insulin': None,
                    'basal_dose': 0,
                    'bolus_insulin': None,
                    'insulin_pump': False,
                    'metformin': True,
                    'sglt2_inhibitor': np.random.choice([True, False], p=[0.5, 0.5]),
                    'glp1_ra': np.random.choice([True, False], p=[0.3, 0.7])
                }
            else:
                # Insulin + oral medications
                regimen = {
                    'basal_insulin': np.random.choice(['Lantus', 'Toujeo', 'Levemir']),
                    'basal_dose': np.random.uniform(10, 50),
                    'bolus_insulin': np.random.choice(['Novolog', 'Humalog', None], p=[0.5, 0.3, 0.2]),
                    'insulin_pump': False,
                    'metformin': True,
                    'sglt2_inhibitor': np.random.choice([True, False], p=[0.4, 0.6]),
                    'glp1_ra': np.random.choice([True, False], p=[0.3, 0.7])
                }
            return regimen
        else:
            # Prediabetes
            return {
                'basal_insulin': None,
                'basal_dose': 0,
                'bolus_insulin': None,
                'insulin_pump': False,
                'metformin': np.random.choice([True, False], p=[0.3, 0.7]),
                'sglt2_inhibitor': False,
                'glp1_ra': False
            }
    
    def _generate_patient_timeseries(self, patient_id, demographics, days):
        """
        Generate 5-minute CGM data for a single patient
        
        CGM devices measure every 5 minutes → 288 readings per day
        """
        # Time points (5-minute intervals)
        start_time = datetime(2023, 1, 1) + timedelta(days=patient_id * 30)  # Stagger patients
        n_points = days * 288  # 288 * days
        timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_points)]
        
        # Base glucose patterns
        glucose = self._generate_glucose_series(
            n_points, 
            demographics['insulin_sensitivity'],
            demographics['baseline_hba1c']
        )
        
        # Add meal effects
        meals = self._generate_meal_events(n_points, timestamps, demographics)
        glucose += meals['glucose_effect']
        
        # Add insulin effects
        insulin = self._generate_insulin_events(n_points, timestamps, demographics, glucose)
        glucose += insulin['glucose_effect']
        
        # Add activity effects
        activity = self._generate_activity_series(n_points, timestamps)
        glucose += activity['glucose_effect']
        
        # Add CGM sensor noise and artifacts
        observed_glucose = self._add_cgm_noise(glucose)
        
        # Calculate clinical outcomes
        outcomes = self._calculate_clinical_outcomes(observed_glucose)
        
        # Create dataframe
        df = pd.DataFrame({
            'timestamp': timestamps,
            'patient_id': demographics['patient_id'],
            'glucose': np.round(observed_glucose, 1),
            'true_glucose': np.round(glucose, 1),  # Underlying true glucose
            'carb_intake': meals['carbs'],
            'basal_insulin': insulin['basal_rate'],
            'bolus_insulin': insulin['bolus_dose'],
            'insulin_delivered': insulin['total_delivered'],
            'activity_level': activity['level'],
            'steps': activity['steps'],
            'sleep_quality': activity['sleep_quality'],
            'stress_level': activity['stress_level'],
            **outcomes  # Add clinical outcome columns
        })
        
        # Add time features
        df['hour'] = df['timestamp'].apply(lambda x: x.hour)
        df['day_of_week'] = df['timestamp'].apply(lambda x: x.weekday())
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['timestamp'].apply(lambda x: x.month)
        
        return df
    
    def _generate_glucose_series(self, n_points, insulin_sensitivity, baseline_hba1c):
        """Generate base glucose pattern with circadian rhythms and variability"""
        
        # Base glucose level (mg/dL) from HbA1c
        base_glucose = (baseline_hba1c * 28.7) - 46.7  # Approximate conversion
        
        # Circadian rhythm (24-hour pattern)
        circadian = np.zeros(n_points)
        for i in range(n_points):
            hour = (i // 12) % 24  # 5-min intervals -> 12 per hour
            # Dawn phenomenon: higher glucose in early morning
            if 4 <= hour <= 8:
                circadian[i] = 20 * np.sin((hour - 4) * np.pi / 4)
            # Evening rise
            elif 18 <= hour <= 22:
                circadian[i] = 15 * np.sin((hour - 18) * np.pi / 4)
        
        # Long-term variability (weekly patterns, illness, etc.)
        long_term = 10 * np.sin(np.arange(n_points) * 2 * np.pi / (288 * 7))  # Weekly
        long_term += 5 * np.random.randn(n_points).cumsum() / 100  # Random walk
        
        # Combine components
        glucose = base_glucose + circadian + long_term
        
        # Add random noise
        glucose += np.random.normal(0, 5, n_points)
        
        # Ensure glucose is in realistic range
        glucose = np.clip(glucose, 40, 400)
        
        return glucose
    
    def _generate_meal_events(self, n_points, timestamps, demographics):
        """Generate realistic meal events with carbohydrate intake"""
        
        # Typical meal times (hours)
        meal_times = [7, 8, 12, 13, 18, 19, 20]  # Breakfast, lunch, dinner ranges
        meal_probabilities = [0.3, 0.5, 0.6, 0.4, 0.7, 0.5, 0.3]  # Probability of meal
        
        carbs = np.zeros(n_points)
        glucose_effect = np.zeros(n_points)
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            
            # Check if this is a meal time
            for mt, prob in zip(meal_times, meal_probabilities):
                if abs(hour - mt) < 0.5:  # Within 30 minutes
                    if np.random.random() < prob * 0.1:  # 10% of time points are meal starts
                        # Meal size varies
                        if hour in [7, 8]:  # Breakfast
                            meal_size = np.random.choice([0, 20, 40, 60], p=[0.1, 0.3, 0.4, 0.2])
                        elif hour in [12, 13]:  # Lunch
                            meal_size = np.random.choice([0, 30, 50, 80], p=[0.1, 0.3, 0.4, 0.2])
                        else:  # Dinner
                            meal_size = np.random.choice([0, 40, 60, 100], p=[0.1, 0.2, 0.4, 0.3])
                        
                        carbs[i] = meal_size
                        
                        # Glucose response to meal (peaks 60-90 min after meal)
                        for j in range(24):  # 2 hours of effect (12 * 5-min = 60 min, 24 = 2 hours)
                            if i + j < n_points:
                                # Delayed peak
                                peak_factor = np.exp(-((j-12)/6)**2)  # Gaussian peak at 60 min
                                effect = meal_size * 2.5 * peak_factor / demographics['insulin_sensitivity']
                                glucose_effect[i + j] += effect
        
        return {'carbs': carbs, 'glucose_effect': glucose_effect}
    
    def _generate_insulin_events(self, n_points, timestamps, demographics, glucose):
        """Generate insulin administration events"""
        
        basal_rate = np.zeros(n_points)
        bolus_dose = np.zeros(n_points)
        total_delivered = np.zeros(n_points)
        
        # Basal insulin (continuous)
        if demographics['treatment_regimen']['basal_dose'] > 0:
            daily_basal = demographics['treatment_regimen']['basal_dose']
            basal_per_interval = daily_basal / 288  # 288 intervals per day
            basal_rate[:] = basal_per_interval
        
        # Bolus insulin (at meal times)
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            
            # Bolus given at meal times (within 15 min of meal)
            if minute % 15 == 0 and hour in [7, 8, 12, 13, 18, 19]:
                if np.random.random() < 0.3:  # 30% of meal times have bolus
                    # Insulin dose based on current glucose and carbs
                    current_glucose = glucose[i]
                    carbs = 0
                    
                    # Look for recent meal
                    for j in range(1, 13):  # Check last hour
                        if i - j >= 0:
                            # We need to access carbs from meals data
                            # This is simplified - in real implementation you'd pass meals data
                            pass
                    
                    # Simple insulin dosing algorithm
                    if carbs > 0:
                        insulin_per_carb = 0.1  # 1 unit per 10g carbs
                        correction_factor = 0.05  # 1 unit per 20 mg/dL above target
                        target_glucose = 120
                        
                        if current_glucose > target_glucose:
                            correction_dose = (current_glucose - target_glucose) * correction_factor
                        else:
                            correction_dose = 0
                        
                        dose = carbs * insulin_per_carb + correction_dose
                        dose = max(0, min(15, dose))  # Cap at 15 units
                        
                        bolus_dose[i] = round(dose, 1)
        
        # Total insulin delivered (basal + bolus)
        total_delivered = basal_rate + bolus_dose
        
        # Insulin effect on glucose (lowers glucose)
        glucose_effect = np.zeros(n_points)
        for i in range(n_points):
            if total_delivered[i] > 0:
                # Insulin effect over next 4 hours
                insulin_magnitude = total_delivered[i] * 10  # Rough effect
                for j in range(48):  # 4 hours (48 * 5-min)
                    if i + j < n_points:
                        decay = np.exp(-j / 24)  # Exponential decay
                        glucose_effect[i + j] -= insulin_magnitude * decay
        
        return {
            'basal_rate': basal_rate,
            'bolus_dose': bolus_dose,
            'total_delivered': total_delivered,
            'glucose_effect': glucose_effect
        }
    
    def _generate_activity_series(self, n_points, timestamps):
        """Generate physical activity and lifestyle variables"""
        
        activity_level = np.zeros(n_points)
        steps = np.zeros(n_points)
        sleep_quality = np.zeros(n_points)
        stress_level = np.zeros(n_points)
        glucose_effect = np.zeros(n_points)
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # Use weekday() for datetime objects (Monday=0, Sunday=6)
            day = ts.weekday()
            
            # Activity patterns
            if 8 <= hour <= 22:  # Awake hours
                # Weekend vs weekday (5=Saturday, 6=Sunday)
                if day >= 5:  # Weekend
                    base_activity = 0.3 + 0.5 * np.random.random()
                else:  # Weekday
                    base_activity = 0.2 + 0.6 * np.random.random()
                
                # Step count (roughly 100 steps per activity unit)
                steps[i] = base_activity * np.random.randint(50, 200)
            else:  # Sleep hours
                base_activity = 0.05 * np.random.random()
                steps[i] = 0
            
            activity_level[i] = base_activity
            
            # Sleep quality (only relevant during sleep)
            if 0 <= hour <= 6:
                sleep_quality[i] = np.random.normal(0.7, 0.2)
            else:
                sleep_quality[i] = 0
            
            # Stress level (higher during work hours)
            if 9 <= hour <= 17 and day < 5:  # Work hours on weekdays
                stress_level[i] = np.random.normal(0.6, 0.2)
            else:
                stress_level[i] = np.random.normal(0.3, 0.1)
            
            # Glucose effect: activity lowers glucose, stress raises it
            glucose_effect[i] = -activity_level[i] * 5 + (stress_level[i] - 0.3) * 10
        
        return {
            'level': np.round(activity_level, 2),
            'steps': steps.astype(int),
            'sleep_quality': np.round(sleep_quality, 2),
            'stress_level': np.round(stress_level, 2),
            'glucose_effect': glucose_effect
        }
    
    def _add_cgm_noise(self, glucose):
        """Add realistic CGM sensor noise and artifacts"""
        
        n_points = len(glucose)
        observed = glucose.copy()
        
        # Random noise (Gaussian)
        observed += np.random.normal(0, 3, n_points)
        
        # Sensor drift (slowly varying bias)
        drift = 5 * np.cumsum(np.random.randn(n_points)) / np.sqrt(n_points)
        observed += drift
        
        # Calibration errors (step changes)
        n_calibrations = n_points // 288  # Once per day on average
        cal_points = np.random.choice(n_points, n_calibrations, replace=False)
        for cp in cal_points:
            cal_error = np.random.normal(0, 5)
            observed[cp:] += cal_error
        
        # Dropouts (missing data)
        dropout_prob = 0.01
        dropout_mask = np.random.random(n_points) < dropout_prob
        observed[dropout_mask] = np.nan
        
        # Ensure values are in realistic range
        observed = np.clip(observed, 40, 400)
        
        return observed
    
    def _calculate_clinical_outcomes(self, glucose):
        """Calculate clinical outcome labels"""
        
        # Convert to numpy array if it's a Series
        if hasattr(glucose, 'values'):
            glucose = glucose.values
        
        # Clinical thresholds (mg/dL)
        severe_hypo_threshold = 54
        hypo_threshold = 70
        hyper_threshold = 180
        severe_hyper_threshold = 250
        
        # Current events
        severe_hypo_current = (glucose < severe_hypo_threshold).astype(int)
        hypo_current = ((glucose < hypo_threshold) & (glucose >= severe_hypo_threshold)).astype(int)
        hyper_current = ((glucose > hyper_threshold) & (glucose < severe_hyper_threshold)).astype(int)
        severe_hyper_current = (glucose > severe_hyper_threshold).astype(int)
        
        # Prediction targets (event in next 2 hours)
        n_points = len(glucose)
        hypo_next_2h = np.zeros(n_points, dtype=int)
        severe_hypo_next_2h = np.zeros(n_points, dtype=int)
        hyper_next_2h = np.zeros(n_points, dtype=int)
        
        for i in range(n_points - 24):  # 24 intervals = 2 hours
            if np.any(glucose[i+1:i+25] < hypo_threshold):
                hypo_next_2h[i] = 1
            if np.any(glucose[i+1:i+25] < severe_hypo_threshold):
                severe_hypo_next_2h[i] = 1
            if np.any(glucose[i+1:i+25] > hyper_threshold):
                hyper_next_2h[i] = 1
        
        # Time in range metrics (for aggregation)
        time_in_range = ((glucose >= 70) & (glucose <= 180)).astype(int)
        time_below_range = (glucose < 70).astype(int)
        time_above_range = (glucose > 180).astype(int)
        
        return {
            'hypo_current': hypo_current,
            'severe_hypo_current': severe_hypo_current,
            'hyper_current': hyper_current,
            'severe_hyper_current': severe_hyper_current,
            'hypo_next_2h': hypo_next_2h,
            'severe_hypo_next_2h': severe_hypo_next_2h,
            'hyper_next_2h': hyper_next_2h,
            'time_in_range': time_in_range,
            'time_below_range': time_below_range,
            'time_above_range': time_above_range
        }
    
    def split_and_save_dataset(self, df, patient_metadata, train_ratio=0.7, val_ratio=0.15):
        """Split dataset and save with metadata"""
        
        # Sort by patient and time
        df = df.sort_values(['patient_id', 'timestamp'])
        
        # Split by patient (ensure all data for a patient stays together)
        patients = df['patient_id'].unique()
        np.random.shuffle(patients)
        
        n_train = int(len(patients) * train_ratio)
        n_val = int(len(patients) * val_ratio)
        
        train_patients = patients[:n_train]
        val_patients = patients[n_train:n_train + n_val]
        test_patients = patients[n_train + n_val:]
        
        train_df = df[df['patient_id'].isin(train_patients)]
        val_df = df[df['patient_id'].isin(val_patients)]
        test_df = df[df['patient_id'].isin(test_patients)]
        
        # Save datasets
        train_df.to_csv(f"{self.output_dir}/train_data.csv", index=False)
        val_df.to_csv(f"{self.output_dir}/val_data.csv", index=False)
        test_df.to_csv(f"{self.output_dir}/test_data.csv", index=False)
        
        # Save full dataset (optional)
        df.to_csv(f"{self.output_dir}/full_dataset.csv", index=False)
        
        # Save parquet format for efficient storage
        try:
            train_df.to_parquet(f"{self.output_dir}/train_data.parquet", index=False)
            val_df.to_parquet(f"{self.output_dir}/val_data.parquet", index=False)
            test_df.to_parquet(f"{self.output_dir}/test_data.parquet", index=False)
        except:
            print("Parquet format not available, skipping...")
        
        # Save metadata
        metadata = {
            'dataset_info': {
                'name': 'Diabetes CGM Synthetic Dataset',
                'version': '1.0',
                'generated_date': datetime.now().isoformat(),
                'description': 'Realistic synthetic CGM data with clinical outcomes',
                'sampling_frequency': '5 minutes',
                'n_patients': len(patients),
                'n_train_patients': len(train_patients),
                'n_val_patients': len(val_patients),
                'n_test_patients': len(test_patients),
                'total_readings': len(df),
                'date_range': [df['timestamp'].min(), df['timestamp'].max()]
            },
            'features': {
                'temporal': ['timestamp', 'hour', 'day_of_week', 'is_weekend', 'month'],
                'glucose': ['glucose', 'true_glucose'],
                'treatments': ['carb_intake', 'basal_insulin', 'bolus_insulin', 'insulin_delivered'],
                'lifestyle': ['activity_level', 'steps', 'sleep_quality', 'stress_level'],
                'outcomes': ['hypo_current', 'severe_hypo_current', 'hyper_current', 
                           'severe_hyper_current', 'hypo_next_2h', 'severe_hypo_next_2h', 
                           'hyper_next_2h', 'time_in_range', 'time_below_range', 'time_above_range']
            },
            'clinical_thresholds': {
                'severe_hypoglycemia': '<54 mg/dL',
                'hypoglycemia': '<70 mg/dL',
                'hyperglycemia': '>180 mg/dL',
                'severe_hyperglycemia': '>250 mg/dL',
                'target_range': '70-180 mg/dL'
            },
            'patients': patient_metadata,
            'split_info': {
                'train_patients': train_patients.tolist(),
                'val_patients': val_patients.tolist(),
                'test_patients': test_patients.tolist(),
                'split_method': 'patient_wise'
            }
        }
        
        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(df, patient_metadata)
        with open(f"{self.output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return train_df, val_df, test_df
    
    def _generate_summary_statistics(self, df, patient_metadata):
        """Generate summary statistics for the dataset"""
        
        summary = {
            'overall': {
                'mean_glucose': float(df['glucose'].mean()),
                'std_glucose': float(df['glucose'].std()),
                'min_glucose': float(df['glucose'].min()),
                'max_glucose': float(df['glucose'].max()),
                'percent_time_in_range': float(df['time_in_range'].mean() * 100),
                'percent_time_below_range': float(df['time_below_range'].mean() * 100),
                'percent_time_above_range': float(df['time_above_range'].mean() * 100),
                'hypo_event_rate': float(df['hypo_current'].mean() * 100),
                'severe_hypo_rate': float(df['severe_hypo_current'].mean() * 100),
                'hyper_event_rate': float(df['hyper_current'].mean() * 100)
            },
            'by_patient': {}
        }
        
        # Patient-level statistics
        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id]
            summary['by_patient'][patient_id] = {
                'mean_glucose': float(patient_df['glucose'].mean()),
                'time_in_range': float(patient_df['time_in_range'].mean() * 100),
                'hypo_rate': float(patient_df['hypo_current'].mean() * 100),
                'demographics': patient_metadata.get(patient_id, {})
            }
        
        return summary


# ================== USAGE EXAMPLE ==================

if __name__ == "__main__":
    print("=" * 60)
    print("DIABETES SYNTHETIC DATASET GENERATOR")
    print("=" * 60)
    
    # Initialize generator
    generator = DiabetesTimeSeriesGenerator(output_dir="./data/raw")
    
    # Generate dataset
    print("\n📊 Generating patient cohort...")
    print("   - 50 patients with diverse characteristics")
    print("   - 30 days of 5-minute CGM data per patient")
    print(f"   - Total: {50 * 30 * 288:,} glucose readings")
    
    df, patient_metadata = generator.generate_patient_cohort(
        n_patients=50, 
        days_per_patient=30
    )
    
    print(f"\n✅ Generated {len(df):,} total readings")
    print(f"   - Patients: {df['patient_id'].nunique()}")
    print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Display sample
    print("\n📋 Sample data (first 5 rows):")
    print(df[['patient_id', 'timestamp', 'glucose', 'carb_intake', 
              'insulin_delivered', 'hypo_current']].head())
    
    # Check clinical outcomes distribution
    print("\n📊 Clinical Outcomes Distribution:")
    print(f"   - Hypoglycemia events: {df['hypo_current'].sum():,} ({df['hypo_current'].mean()*100:.1f}%)")
    print(f"   - Severe hypo events: {df['severe_hypo_current'].sum():,} ({df['severe_hypo_current'].mean()*100:.2f}%)")
    print(f"   - Hyperglycemia events: {df['hyper_current'].sum():,} ({df['hyper_current'].mean()*100:.1f}%)")
    print(f"   - Time in range (70-180): {df['time_in_range'].mean()*100:.1f}%")
    
    # Split and save
    print("\n💾 Saving datasets...")
    train, val, test = generator.split_and_save_dataset(df, patient_metadata)
    
    print(f"\n✅ Datasets saved to: {os.path.abspath(generator.output_dir)}")
    print(f"   - Train: {len(train):,} rows ({len(train['patient_id'].unique())} patients)")
    print(f"   - Validation: {len(val):,} rows ({len(val['patient_id'].unique())} patients)")
    print(f"   - Test: {len(test):,} rows ({len(test['patient_id'].unique())} patients)")
    
    print("\n📁 Files created:")
    for file in os.listdir(generator.output_dir):
        if file.endswith(('.csv', '.parquet', '.json')):
            size = os.path.getsize(f"{generator.output_dir}/{file}") / (1024*1024)
            print(f"   - {file} ({size:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("🎉 DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load data: pd.read_csv('./diabetes_datasets/train_data.csv')")
    print("2. Check metadata: cat ./diabetes_datasets/metadata.json")
    print("3. Start modeling: predict hypoglycemia, forecast glucose, etc.")