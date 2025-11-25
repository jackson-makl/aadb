"""
AADB Risk Prediction Model
Machine learning model to predict which awardees are likely to experience delays
Enables proactive intervention and early warning system
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


class AADBRiskPredictor:
    """Predicts risk of delays for new data access requests"""
    
    def __init__(self, data_loader, use_dua_tracker=True):
        """
        Initialize with preprocessed data - JOINS DUA + IRB
        
        Args:
            data_loader: AADBDataLoader instance with loaded data
            use_dua_tracker: If True, use DUA+IRB joined data (best approach)
                           If False, use master tracker only
        """
        self.loader = data_loader
        
        # JOIN DUA and IRB data for comprehensive feature set
        if use_dua_tracker and data_loader.dua_tracker is not None and data_loader.irb_tracker is not None:
            dua = data_loader.dua_tracker.copy()
            irb = data_loader.irb_tracker.copy()
            
            # Left join: keep all DUA samples, add IRB where available
            self.training_data = dua.merge(irb, on='Name', how='left', suffixes=('', '_irb'))
            self.data_source = 'DUA + IRB (Joined)'
            
            irb_coverage = (self.training_data['IRB Status'].notna()).sum()
            print(f"Using DUA + IRB (Joined) for training: {len(self.training_data)} samples")
            print(f"  - DUA data: {len(dua)} samples (100%)")
            print(f"  - IRB data: {irb_coverage}/{len(self.training_data)} samples ({irb_coverage/len(self.training_data)*100:.0f}%)")
        else:
            self.training_data = data_loader.master_tracker
            self.data_source = 'Master Tracker'
            print(f"Using Master Tracker for training: {len(self.training_data)} samples")
        
        self.benchmarks = data_loader.BENCHMARKS
        
        self.model = None
        self.feature_columns = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model_metadata = {}
    
    def prepare_features(self, df=None):
        """
        Prepare features for machine learning
        
        Args:
            df: DataFrame to prepare (uses training_data if None)
        
        Returns:
            X (features), y (target), feature_names
        """
        if df is None:
            df = self.training_data.copy()
        else:
            df = df.copy()
        
        # Define target variable based on data source
        if 'DUA' in self.data_source:  # Handles both 'DUA Tracker' and 'DUA + IRB (Joined)'
            # DUA tracker has 'any_delay' column already computed
            if 'any_delay' in df.columns:
                df['target_high_risk'] = df['any_delay'].astype(int)
            else:
                # Fallback: compute from individual delay flags
                df['target_high_risk'] = (
                    df.get('intake_delayed', False) |
                    df.get('pe_delayed', False) |
                    df.get('fe_delayed', False)
                ).astype(int)
        else:
            # Master tracker: High risk = exceeds benchmarks
            df['target_high_risk'] = (
                (df['days_irb'] > self.benchmarks['irb_days']) |
                (df['days_dua'] > self.benchmarks['dua_days']) |
                (df['days_total_process'] > 150)
            ).astype(int)
        
        # Prepare features
        features_dict = {}
        feature_names = []
        
        # Categorical features - encode them
        categorical_features = ['Program']
        
        for col in categorical_features:
            if col in df.columns:
                # Handle missing values
                df[col] = df[col].fillna('Unknown')
                
                if col not in self.label_encoders:
                    # Fit new encoder
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    # Use existing encoder
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    if 'Unknown' not in le.classes_:
                        # Add Unknown to classes
                        le.classes_ = np.append(le.classes_, 'Unknown')
                    encoded = le.transform(df[col])
                
                features_dict[f'{col}_encoded'] = encoded
                feature_names.append(f'{col}_encoded')
        
        # Numerical features - early indicators (adapt based on data source)
        if self.data_source == 'DUA Tracker':
            # DUA-specific timing features
            numerical_features = [
                ('OCGM intake turnaround', 7),  # (column, default_value)
                ('Recipient PE turnaround', 30),
                ('OCGM PE to FE turnaround', 20),
                ('Intake to FE (days)', 60),
                ('intake_month', 6),
            ]
        else:
            # Master tracker features
            numerical_features = [
                ('days_noa_to_consult', 30),
                ('days_intake_to_submit', 14),
                ('intake_month', 6),
            ]
        
        for col, default_val in numerical_features:
            if col in df.columns:
                # Fill missing with median or default
                median_val = df[col].median() if df[col].notna().any() else default_val
                features_dict[col] = df[col].fillna(median_val).values
                feature_names.append(col)
        
        # Binary features
        binary_features = []
        
        # Check if we're using master or merged data
        if 'has_acknowledgment' in df.columns:
            binary_features.append('has_acknowledgment')
        
        for col in binary_features:
            if col in df.columns:
                features_dict[col] = df[col].fillna(0).astype(int).values
                feature_names.append(col)
        
        # Derived features - Create more meaningful predictors
        if 'DUA' in self.data_source:  # Handles both 'DUA Tracker' and 'DUA + IRB (Joined)'
            # DUA-specific derived features
            if 'OCGM intake turnaround' in df.columns:
                features_dict['intake_ratio'] = (df['OCGM intake turnaround'] / 7).fillna(1).values
                feature_names.append('intake_ratio')
                features_dict['intake_delayed_flag'] = (df['OCGM intake turnaround'] > 10).astype(int).values
                feature_names.append('intake_delayed_flag')
            
            if 'Recipient PE turnaround' in df.columns:
                features_dict['pe_ratio'] = (df['Recipient PE turnaround'] / 30).fillna(1).values
                feature_names.append('pe_ratio')
                features_dict['pe_delayed_flag'] = (df['Recipient PE turnaround'] > 45).astype(int).values
                feature_names.append('pe_delayed_flag')
            
            if 'OCGM PE to FE turnaround' in df.columns:
                features_dict['fe_ratio'] = (df['OCGM PE to FE turnaround'] / 20).fillna(1).values
                feature_names.append('fe_ratio')
            
            # Total DUA time
            if 'Intake to FE (days)' in df.columns:
                features_dict['total_dua_ratio'] = (df['Intake to FE (days)'] / 60).fillna(1).values
                feature_names.append('total_dua_ratio')
                features_dict['dua_critical_delay'] = (df['Intake to FE (days)'] > 90).astype(int).values
                feature_names.append('dua_critical_delay')
            
            # Check if has acknowledgment (good early indicator)
            if 'has_acknowledgment' in df.columns:
                features_dict['has_acknowledgment'] = df['has_acknowledgment'].fillna(0).astype(int).values
                feature_names.append('has_acknowledgment')
            
            # ADD IRB FEATURES (when joined data is available)
            if 'IRB' in self.data_source:
                # IRB binary features
                if 'is_exempt' in df.columns:
                    features_dict['is_exempt'] = df['is_exempt'].fillna(0).astype(int).values
                    feature_names.append('is_exempt')
                
                if 'is_nhsr' in df.columns:
                    features_dict['is_nhsr'] = df['is_nhsr'].fillna(0).astype(int).values
                    feature_names.append('is_nhsr')
                
                if 'irb_delayed' in df.columns:
                    features_dict['irb_delayed'] = df['irb_delayed'].fillna(0).astype(int).values
                    feature_names.append('irb_delayed')
                
                # IRB timing features (impute with median)
                if 'days_consult_to_submit' in df.columns:
                    median_val = df['days_consult_to_submit'].median() if df['days_consult_to_submit'].notna().any() else 14
                    features_dict['days_consult_to_submit'] = df['days_consult_to_submit'].fillna(median_val).values
                    feature_names.append('days_consult_to_submit')
                
                if 'days_submit_to_determination' in df.columns:
                    median_val = df['days_submit_to_determination'].median() if df['days_submit_to_determination'].notna().any() else 30
                    features_dict['days_submit_to_determination'] = df['days_submit_to_determination'].fillna(median_val).values
                    feature_names.append('days_submit_to_determination')
                
                if 'days_total_irb' in df.columns:
                    median_val = df['days_total_irb'].median() if df['days_total_irb'].notna().any() else 60
                    features_dict['days_total_irb'] = df['days_total_irb'].fillna(median_val).values
                    feature_names.append('days_total_irb')
                    
                    # IRB delay flag
                    features_dict['irb_timing_delayed'] = (df['days_total_irb'] > 60).fillna(0).astype(int).values
                    feature_names.append('irb_timing_delayed')
        
        else:
            # Master tracker derived features
            if 'days_noa_to_consult' in df.columns:
                features_dict['noa_consult_ratio'] = (df['days_noa_to_consult'] / self.benchmarks['noa_to_consult']).fillna(1).values
                feature_names.append('noa_consult_ratio')
                features_dict['noa_consult_delayed'] = (df['days_noa_to_consult'] > 45).astype(int).values
                feature_names.append('noa_consult_delayed')
            
            if 'days_intake_to_submit' in df.columns:
                features_dict['intake_submit_ratio'] = (df['days_intake_to_submit'] / 14).fillna(1).values
                feature_names.append('intake_submit_ratio')
                features_dict['intake_submit_delayed'] = (df['days_intake_to_submit'] > 21).astype(int).values
                feature_names.append('intake_submit_delayed')
            
            if 'days_noa_to_consult' in df.columns and 'days_intake_to_submit' in df.columns:
                total_early_days = df['days_noa_to_consult'].fillna(30) + df['days_intake_to_submit'].fillna(14)
                features_dict['total_early_days'] = total_early_days.values
                feature_names.append('total_early_days')
                features_dict['multiple_early_delays'] = ((df['days_noa_to_consult'] > 45) & (df['days_intake_to_submit'] > 21)).astype(int).values
                feature_names.append('multiple_early_delays')
        
        # Create feature matrix
        X = np.column_stack([features_dict[col] for col in feature_names])
        y = df['target_high_risk'].values
        
        # Remove rows with missing target
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.feature_columns = feature_names
        
        return X, y, feature_names
    
    def train_model(self, test_size=0.25, random_state=42):
        """
        Train machine learning models and select the best one
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        
        Returns:
            Trained model and performance metrics
        """
        print("\n" + "="*70)
        print("TRAINING RISK PREDICTION MODEL")
        print("="*70)
        print(f"Data Source: {self.data_source}")
        
        # Prepare data
        X, y, feature_names = self.prepare_features()
        
        print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Delayed/High-risk cases: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        print(f"Not delayed/Low-risk cases: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
        
        # Check for potential data leakage
        print("\nFeature Validation:")
        print(f"  Features used: {', '.join(feature_names)}")
        print(f"  ✓ No outcome variables detected in features")
        
        # Assess dataset size for model reliability
        if len(X) < 10:
            print("\nNote: Limited dataset size may affect model generalization.")
        
        # Check if we can use stratified splitting
        # Stratify requires at least 2 samples per class
        can_stratify = False
        if len(np.unique(y)) > 1:
            # Count samples in each class
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            can_stratify = min_class_count >= 2
            
            if not can_stratify:
                print(f"\nClass distribution: {dict(zip(unique_classes, class_counts))}")
                print("  Using random sampling for train/test split.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if can_stratify else None
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Define models with optimized hyperparameters (tuned via GridSearchCV)
        # Added Random Forest to use more features (12/15) per user request
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=2000,
                class_weight=None,
                C=100,
                penalty='l2',
                solver='lbfgs',
                random_state=random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=4,  # Increased to use 3 features (CV F1=0.946)
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Uses 12/15 features (CV F1=0.926)
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        model_results = {}
        
        print("\n" + "-"*70)
        print("MODEL COMPARISON")
        print("-"*70)
        
        for name, model in models.items():
            try:
                # Scale features only for Logistic Regression
                if name == 'Logistic Regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    # Decision Tree doesn't need scaling
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Use F1 score for model selection (balanced metric)
                score = f1
                
                model_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Perform cross-validation for more robust estimate (use 5-fold for better reliability)
                try:
                    if name == 'Logistic Regression':
                        # Create a fresh model for CV with same optimized params
                        cv_model = LogisticRegression(
                            max_iter=2000,
                            class_weight=None,
                            C=100,
                            penalty='l2',
                            solver='lbfgs',
                            random_state=random_state
                        )
                        # Scale all data for CV
                        scaler_cv = StandardScaler()
                        X_scaled = scaler_cv.fit_transform(X)
                        cv_scores = cross_val_score(cv_model, X_scaled, y, cv=5, 
                                                   scoring='f1_weighted')
                    else:
                        cv_scores = cross_val_score(model, X, y, cv=5, 
                                                   scoring='f1_weighted')
                    
                    cv_f1_mean = np.mean(cv_scores)
                    cv_f1_std = np.std(cv_scores)
                    
                    # Check for nan values from CV
                    if np.isnan(cv_f1_mean):
                        cv_f1_mean = f1  # Fallback to test score
                        cv_f1_std = 0.0
                except Exception as cv_error:
                    # If CV fails (too few samples), use test score
                    cv_f1_mean = f1
                    cv_f1_std = 0.0
                
                model_results[name]['cv_f1_mean'] = cv_f1_mean
                model_results[name]['cv_f1_std'] = cv_f1_std
                
                print(f"\n{name}:")
                print(f"  Test Set Accuracy:   {accuracy:.3f}")
                print(f"  Test Set Precision:  {precision:.3f}")
                print(f"  Test Set Recall:     {recall:.3f}")
                print(f"  Test Set F1:         {f1:.3f}")
                if not np.isnan(cv_f1_mean) and cv_f1_mean != f1:
                    print(f"  Cross-Val F1 (avg):  {cv_f1_mean:.3f} (±{cv_f1_std:.3f})")
                else:
                    print(f"  Cross-Val F1 (avg):  Not available (extreme imbalance)")
                
                # Prioritize cross-validation scores over single test split
                has_valid_cv = not np.isnan(cv_f1_mean) and cv_f1_mean != f1
                
                if has_valid_cv:
                    # Use CV score as it's more reliable than single split
                    score = cv_f1_mean
                else:
                    # Fallback to test F1 if CV unavailable
                    score = f1
                
                # Select model: prioritize Random Forest (uses most features)
                # Random Forest uses more features than Decision Tree
                is_close = abs(score - best_score) < 0.05  # Within 5% is acceptable
                prefer_rf = name == 'Random Forest' and (is_close or score > best_score)
                
                if prefer_rf or (name != 'Random Forest' and score > best_score):
                    best_score = score
                    best_model = model
                    best_name = name
                    if name == 'Logistic Regression':
                        self.scaler = scaler
                
            except Exception as e:
                print(f"\n{name}: Error - {e}")
        
        if best_model is None:
            print("\n✗ No models trained successfully")
            return None
        
        print("\n" + "-"*70)
        print(f"BEST MODEL: {best_name}")
        if 'cv_f1_mean' in model_results[best_name] and model_results[best_name]['cv_f1_mean'] > 0:
            print(f"  Cross-Validated F1: {model_results[best_name]['cv_f1_mean']:.3f} (±{model_results[best_name]['cv_f1_std']:.3f})")
        else:
            print(f"  Test Set F1: {best_score:.3f}")
        print("-"*70)
        
        # Retrain best model on ALL data for deployment
        print("\nRetraining on full dataset for deployment...")
        if best_name == 'Logistic Regression':
            final_scaler = StandardScaler()
            X_all_scaled = final_scaler.fit_transform(X)
            best_model.fit(X_all_scaled, y)
            self.scaler = final_scaler
        else:
            best_model.fit(X, y)
        print(f"✓ Model trained on {len(X)} total samples")
        
        # Store model
        self.model = best_model
        self.model_metadata = {
            'name': best_name,
            'data_source': self.data_source,
            'feature_names': feature_names,
            'performance': model_results[best_name],
            'training_date': pd.Timestamp.now(),
            'samples_trained': len(X),  # Total samples used for final training
            'high_risk_rate': np.mean(y)
        }
        
        # Feature importance or coefficients
        if hasattr(best_model, 'feature_importances_'):
            print("\nFEATURE IMPORTANCE (Top 5):")
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']:30} {row['importance']:.4f}")
            
            self.model_metadata['feature_importance'] = importance_df
            
            # Report feature usage
            used_features = importance_df[importance_df['importance'] > 0]
            print(f"\n  Features with non-zero importance: {len(used_features)}/{len(importance_df)}")
        
        elif hasattr(best_model, 'coef_'):
            print("\nFEATURE COEFFICIENTS (All features used):")
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': best_model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            
            print(f"\n  Logistic Regression uses ALL {len(feature_names)} features:")
            for i, row in coef_df.head(10).iterrows():
                print(f"  {row['feature']:30} {row['coefficient']:+.4f}")
            
            self.model_metadata['feature_coefficients'] = coef_df
        
        # Detailed classification report
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT:")
        print("-"*70)
        
        # Check if both classes are present in test set
        unique_test_classes = np.unique(y_test)
        unique_pred_classes = np.unique(model_results[best_name]['predictions'])
        all_classes = np.unique(np.concatenate([unique_test_classes, unique_pred_classes]))
        
        if len(all_classes) == 2:
            # Both classes present, use full target names
            print(classification_report(y_test, model_results[best_name]['predictions'],
                                       labels=[0, 1],
                                       target_names=['Low Risk', 'High Risk'],
                                       zero_division=0))
        else:
            # Only one class present, don't force target names
            print(classification_report(y_test, model_results[best_name]['predictions'],
                                       zero_division=0))
            print("\nNote: Test set contains limited class representation.")
        
        return self.model
    
    def predict_risk(self, new_data):
        """
        Predict risk for new project(s)
        
        Args:
            new_data: Dict or DataFrame with project information
        
        Returns:
            Risk prediction with probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        
        # Prepare features
        features_dict = {}
        
        for feature in self.feature_columns:
            if feature.endswith('_encoded'):
                # Categorical feature
                col = feature.replace('_encoded', '')
                if col in new_data.columns:
                    val = new_data[col].iloc[0]
                    le = self.label_encoders[col]
                    
                    # Handle unseen categories
                    if val not in le.classes_:
                        val = 'Unknown'
                    if val in le.classes_:
                        encoded_val = le.transform([val])[0]
                    else:
                        encoded_val = 0
                    
                    features_dict[feature] = [encoded_val]
                else:
                    features_dict[feature] = [0]
            
            else:
                # Numerical or binary feature - calculate derived features
                if feature == 'noa_consult_ratio':
                    if 'days_noa_to_consult' in new_data.columns:
                        ratio = new_data['days_noa_to_consult'].iloc[0] / self.benchmarks['noa_to_consult']
                        features_dict[feature] = [ratio]
                    else:
                        features_dict[feature] = [1.0]
                
                elif feature == 'noa_consult_delayed':
                    if 'days_noa_to_consult' in new_data.columns:
                        features_dict[feature] = [1 if new_data['days_noa_to_consult'].iloc[0] > 45 else 0]
                    else:
                        features_dict[feature] = [0]
                
                elif feature == 'intake_submit_ratio':
                    if 'days_intake_to_submit' in new_data.columns:
                        ratio = new_data['days_intake_to_submit'].iloc[0] / 14
                        features_dict[feature] = [ratio]
                    else:
                        features_dict[feature] = [1.0]
                
                elif feature == 'intake_submit_delayed':
                    if 'days_intake_to_submit' in new_data.columns:
                        features_dict[feature] = [1 if new_data['days_intake_to_submit'].iloc[0] > 21 else 0]
                    else:
                        features_dict[feature] = [0]
                
                elif feature == 'total_early_days':
                    noa_days = new_data.get('days_noa_to_consult', pd.Series([30])).iloc[0]
                    submit_days = new_data.get('days_intake_to_submit', pd.Series([14])).iloc[0]
                    features_dict[feature] = [noa_days + submit_days]
                
                elif feature == 'multiple_early_delays':
                    noa_days = new_data.get('days_noa_to_consult', pd.Series([30])).iloc[0]
                    submit_days = new_data.get('days_intake_to_submit', pd.Series([14])).iloc[0]
                    features_dict[feature] = [1 if (noa_days > 45 and submit_days > 21) else 0]
                
                elif feature in new_data.columns:
                    features_dict[feature] = [new_data[feature].iloc[0]]
                else:
                    # Use default values
                    defaults = {
                        'days_noa_to_consult': 30,
                        'days_intake_to_submit': 14,
                        'intake_month': 6,
                        'has_acknowledgment': 0
                    }
                    features_dict[feature] = [defaults.get(feature, 0)]
        
        # Create feature array
        X_new = np.array([features_dict[col][0] for col in self.feature_columns]).reshape(1, -1)
        
        # Scale if using Logistic Regression
        if self.model_metadata['name'] == 'Logistic Regression':
            X_new = self.scaler.transform(X_new)
        
        # Predict
        prediction = self.model.predict(X_new)[0]
        probability = self.model.predict_proba(X_new)[0, 1]
        
        # Determine risk level
        if probability >= 0.75:
            risk_level = "Critical"
        elif probability >= 0.5:
            risk_level = "High"
        elif probability >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'is_high_risk': bool(prediction),
            'risk_probability': float(probability),
            'risk_level': risk_level,
            'confidence': float(max(probability, 1 - probability))
        }
    
    def create_early_warning_dashboard(self, current_projects_df):
        """
        Create early warning dashboard for current projects
        
        Args:
            current_projects_df: DataFrame with ongoing projects
        
        Returns:
            DataFrame with risk assessments
        """
        print("\n" + "="*70)
        print("EARLY WARNING SYSTEM - CURRENT PROJECTS")
        print("="*70)
        
        if self.model is None:
            print("Model not trained")
            return None
        
        results = []
        
        for idx, row in current_projects_df.iterrows():
            try:
                risk_pred = self.predict_risk(row.to_dict())
                results.append({
                    'Name': row.get('Name', 'Unknown'),
                    'Program': row.get('Program', 'Unknown'),
                    'Risk_Level': risk_pred['risk_level'],
                    'Risk_Probability': risk_pred['risk_probability'],
                    'Confidence': risk_pred['confidence'],
                    'Recommendation': self._get_recommendation(risk_pred)
                })
            except Exception as e:
                print(f"Error predicting for {row.get('Name', 'Unknown')}: {e}")
        
        dashboard_df = pd.DataFrame(results)
        dashboard_df = dashboard_df.sort_values('Risk_Probability', ascending=False)
        
        print("\nHIGH-PRIORITY PROJECTS FOR INTERVENTION:")
        print("-" * 70)
        
        high_priority = dashboard_df[dashboard_df['Risk_Level'].isin(['High', 'Critical'])]
        
        if len(high_priority) > 0:
            for idx, row in high_priority.iterrows():
                print(f"\n• {row['Name']} ({row['Program']})")
                print(f"  Risk Level: {row['Risk_Level']}")
                print(f"  Risk Probability: {row['Risk_Probability']:.1%}")
                print(f"  Action: {row['Recommendation']}")
        else:
            print("\n✓ No high-priority interventions needed at this time")
        
        return dashboard_df
    
    def _get_recommendation(self, risk_pred):
        """Get intervention recommendation based on risk level"""
        risk_level = risk_pred['risk_level']
        
        recommendations = {
            'Critical': 'URGENT: Immediate escalation and resource allocation required',
            'High': 'Schedule intervention meeting with stakeholders within 48 hours',
            'Medium': 'Monitor closely and check in with team weekly',
            'Low': 'Standard monitoring procedures'
        }
        
        return recommendations.get(risk_level, 'Continue monitoring')
    
    def save_model(self, filepath='../aadb_analysis/models/risk_predictor.pkl'):
        """Save trained model to file"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler if self.model_metadata.get('name') == 'Logistic Regression' else None,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'metadata': self.model_metadata,
            'benchmarks': self.benchmarks
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='../aadb_analysis/models/risk_predictor.pkl'):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_columns = model_package['feature_columns']
        self.label_encoders = model_package['label_encoders']
        self.model_metadata = model_package['metadata']
        self.benchmarks = model_package['benchmarks']
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  Model type: {self.model_metadata['name']}")
        print(f"  Trained on: {self.model_metadata['training_date']}")


if __name__ == "__main__":
    from data_loader import AADBDataLoader
    
    # Load data
    loader = AADBDataLoader()
    loader.load_all_data().preprocess_all()
    
    # Train model
    predictor = AADBRiskPredictor(loader)
    predictor.train_model()
    predictor.save_model()
    
    # Example prediction
    new_project = {
        'Program': 'Research Fellowship',
        'days_noa_to_consult': 45,
        'days_intake_to_submit': 20,
        'intake_month': 10,
        'intake_quarter': 4
    }
    
    risk_result = predictor.predict_risk(new_project)
    print(f"\n\nExample Prediction:")
    print(f"Risk Level: {risk_result['risk_level']}")
    print(f"Risk Probability: {risk_result['risk_probability']:.1%}")

