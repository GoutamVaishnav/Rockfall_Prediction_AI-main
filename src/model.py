import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, StratifiedKFold,
    RandomizedSearchCV, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import json
from scipy import stats
import joblib

warnings.filterwarnings('ignore')


class RockFallPredictor:
    """
    Advanced Rock Fall Prediction System for Open Pit Mines

    Features:
    - Multiple ML algorithms with hyperparameter optimization
    - Mining-specific feature engineering
    - Comprehensive evaluation and risk assessment
    - Model interpretability and feature importance analysis
    - Early warning system capabilities
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.ensemble_model = None
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = None
        self.engineered_features = []
        self.results = {}
        self.risk_thresholds = {}

    def load_data(self, data_path=None, X_train_path="../data/processed/X_train.csv",
                  X_test_path="../data/processed/X_test.csv",
                  y_train_path="../data/processed/y_train.csv",
                  y_test_path="../data/processed/y_test.csv"):
        """Load training and testing data for rock fall prediction"""
        print('Loading rock fall prediction data...')

        if data_path:
            data = pd.read_csv(data_path)

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state,
                stratify=y, shuffle=True
            )
        else:
            self.X_train = pd.read_csv(X_train_path)
            self.X_test = pd.read_csv(X_test_path)
            self.y_train = pd.read_csv(y_train_path).values.ravel()
            self.y_test = pd.read_csv(y_test_path).values.ravel()

        self._clean_data()

        self.feature_names = self.X_train.columns.tolist()
        print(f"Data loaded: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
        print(f"Features: {len(self.feature_names)}")
        print(f"Risk level distribution: {pd.Series(self.y_train).value_counts().to_dict()}")

        self._set_risk_thresholds()

    def _clean_data(self):
        """Clean and preprocess geological data"""
        print("Cleaning geological data...")

        for col in self.X_train.columns:
            if self.X_train[col].dtype in ['float64', 'int64']:
                median_val = self.X_train[col].median()
                self.X_train[col].fillna(median_val, inplace=True)
                self.X_test[col].fillna(median_val, inplace=True)

        critical_features = ['slope_angle', 'crack_length', 'displacement_mm', 'pore_pressure_kPa']
        for feature in critical_features:
            if feature in self.X_train.columns:
                Q1 = self.X_train[feature].quantile(0.25)
                Q3 = self.X_train[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                self.X_train[feature] = self.X_train[feature].clip(lower_bound, upper_bound)
                self.X_test[feature] = self.X_test[feature].clip(lower_bound, upper_bound)

    def _set_risk_thresholds(self):
        """Set risk probability thresholds for early warning system"""
        risk_counts = pd.Series(self.y_train).value_counts()
        total_samples = len(self.y_train)

        if 'High' in risk_counts.index:
            high_ratio = risk_counts['High'] / total_samples
            self.risk_thresholds['High'] = max(0.3, high_ratio * 2)

        if 'Medium' in risk_counts.index:
            medium_ratio = risk_counts['Medium'] / total_samples
            self.risk_thresholds['Medium'] = max(0.2, medium_ratio * 1.5)

        self.risk_thresholds['Low'] = 0.1
        print(f"Risk thresholds set: {self.risk_thresholds}")

    def mining_feature_engineering(self):
        """Create mining-specific engineered features"""
        print("Engineering mining-specific features...")

        if 'slope_angle' in self.X_train.columns and 'slope_height' in self.X_train.columns:
            self.X_train['stability_ratio'] = self.X_train['slope_height'] / np.tan(
                np.radians(self.X_train['slope_angle']))
            self.X_test['stability_ratio'] = self.X_test['slope_height'] / np.tan(
                np.radians(self.X_test['slope_angle']))
            self.engineered_features.append('stability_ratio')

        if 'pore_pressure_kPa' in self.X_train.columns and 'strain_micro' in self.X_train.columns:
            self.X_train['effective_stress'] = self.X_train['strain_micro'] - (self.X_train['pore_pressure_kPa'] / 1000)
            self.X_test['effective_stress'] = self.X_test['strain_micro'] - (self.X_test['pore_pressure_kPa'] / 1000)
            self.engineered_features.append('effective_stress')

        if 'rainfall_mm' in self.X_train.columns and 'temperature_Cvibration_mmps' in self.X_train.columns:
            self.X_train['weather_stress'] = (self.X_train['rainfall_mm'] * 0.7 +
                                              self.X_train['temperature_Cvibration_mmps'] * 0.3)
            self.X_test['weather_stress'] = (self.X_test['rainfall_mm'] * 0.7 +
                                             self.X_test['temperature_Cvibration_mmps'] * 0.3)
            self.engineered_features.append('weather_stress')

        if 'joint_density' in self.X_train.columns and 'crack_length' in self.X_train.columns:
            self.X_train['rock_quality'] = 100 / (1 + self.X_train['joint_density'] + self.X_train['crack_length'] / 10)
            self.X_test['rock_quality'] = 100 / (1 + self.X_test['joint_density'] + self.X_test['crack_length'] / 10)
            self.engineered_features.append('rock_quality')

        if 'displacement_mm' in self.X_train.columns:
            self.X_train['displacement_severity'] = pd.cut(self.X_train['displacement_mm'],
                                                           bins=[0, 5, 15, float('inf')],
                                                           labels=[1, 2, 3])
            self.X_test['displacement_severity'] = pd.cut(self.X_test['displacement_mm'],
                                                          bins=[0, 5, 15, float('inf')],
                                                          labels=[1, 2, 3])
            self.engineered_features.append('displacement_severity')

        print(f"Created {len(self.engineered_features)} engineered features: {self.engineered_features}")

    def prepare_features(self):
        """Prepare features for model training"""
        print("Preparing features for model training...")

        self.mining_feature_engineering()

        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object':
                le = LabelEncoder()
                self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
                self.X_test[col] = le.transform(self.X_test[col].astype(str))

        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("Feature preparation completed")

    def define_models(self):
        """Define multiple ML models optimized for rock fall prediction"""
        print("Defining ML models for rock fall prediction...")

        self.models['RandomForest'] = {
            'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'params': {
                'n_estimators': [200, 500, 800],
                'max_depth': [None, 15, 25, 35],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None]
            },
            'use_scaled': False
        }

        self.models['GradientBoosting'] = {
            'model': GradientBoostingClassifier(random_state=self.random_state),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'use_scaled': False
        }

        self.models['ExtraTrees'] = {
            'model': ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1),
            'params': {
                'n_estimators': [200, 400, 600],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', None]
            },
            'use_scaled': False
        }

        self.models['SVM'] = {
            'model': SVC(random_state=self.random_state, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly'],
                'class_weight': ['balanced', None]
            },
            'use_scaled': True
        }

        self.models['NeuralNetwork'] = {
            'model': MLPClassifier(random_state=self.random_state, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'use_scaled': True
        }

        print(f"Defined {len(self.models)} models for evaluation")

    def train_models(self, use_randomized_search=True):
        """Train all models with hyperparameter optimization"""
        print("Training models with hyperparameter optimization...")

        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for model_name, model_info in self.models.items():
            print(f"\nTraining {model_name}...")

            X_train = self.X_train_scaled if model_info['use_scaled'] else self.X_train
            X_test = self.X_test_scaled if model_info['use_scaled'] else self.X_test

            if use_randomized_search:
                search = RandomizedSearchCV(
                    model_info['model'],
                    model_info['params'],
                    n_iter=20,  # Reduced for faster training
                    cv=cv_folds,
                    scoring='f1_weighted',  # Better for imbalanced mining data
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1
                )

            search.fit(X_train, self.y_train_encoded)

            self.models[model_name]['best_model'] = search.best_estimator_
            self.models[model_name]['best_params'] = search.best_params_
            self.models[model_name]['cv_score'] = search.best_score_

            y_pred = search.best_estimator_.predict(X_test)
            y_pred_proba = search.best_estimator_.predict_proba(X_test)

            accuracy = accuracy_score(self.y_test_encoded, y_pred)
            f1 = f1_score(self.y_test_encoded, y_pred, average='weighted')
            precision = precision_score(self.y_test_encoded, y_pred, average='weighted')
            recall = recall_score(self.y_test_encoded, y_pred, average='weighted')

            self.models[model_name]['metrics'] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'cv_score': search.best_score_
            }

            print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV Score: {search.best_score_:.4f}")

        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['metrics']['f1_score'])
        self.best_model = self.models[best_model_name]['best_model']
        self.best_model_name = best_model_name

        print(f"\nBest model: {best_model_name} (F1 Score: {self.models[best_model_name]['metrics']['f1_score']:.4f})")

    def create_ensemble_model(self):
        """Create ensemble model from top performing models"""
        print("Creating ensemble model...")

        top_models = sorted(self.models.items(),
                            key=lambda x: x[1]['metrics']['f1_score'],
                            reverse=True)[:3]

        ensemble_estimators = []
        for model_name, model_info in top_models:
            ensemble_estimators.append((model_name, model_info['best_model']))

        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'
        )

        X_train = self.X_train
        self.ensemble_model.fit(X_train, self.y_train_encoded)

        y_pred = self.ensemble_model.predict(self.X_test)
        ensemble_f1 = f1_score(self.y_test_encoded, y_pred, average='weighted')

        print(f"Ensemble model F1 Score: {ensemble_f1:.4f}")
        print(f"Ensemble members: {[name for name, _ in top_models]}")

        if ensemble_f1 > self.models[self.best_model_name]['metrics']['f1_score']:
            self.best_model = self.ensemble_model
            self.best_model_name = 'Ensemble'
            print("Ensemble model selected as best performer")

    def analyze_feature_importance(self):
        """Analyze feature importance for rock fall prediction"""
        print("Analyzing feature importance...")

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns

        elif hasattr(self.best_model, 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in self.best_model.estimators_], axis=0)
            feature_names = self.X_train.columns

        else:
            X_test = self.X_test_scaled if self.best_model_name in ['SVM', 'NeuralNetwork'] else self.X_test
            perm_importance = permutation_importance(
                self.best_model, X_test, self.y_test_encoded,
                n_repeats=10, random_state=self.random_state
            )
            importances = perm_importance.importances_mean
            feature_names = self.X_train.columns

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

        self.feature_importance = importance_df
        return importance_df

    def evaluate_model(self):
        """Comprehensive model evaluation with mining-specific metrics"""
        print("Performing comprehensive model evaluation...")

        X_test = (self.X_test_scaled if self.best_model_name in ['SVM', 'NeuralNetwork']
                  else self.X_test)

        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)

        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        y_test_original = self.label_encoder.inverse_transform(self.y_test_encoded)

        accuracy = accuracy_score(y_test_original, y_pred_original)
        f1 = f1_score(self.y_test_encoded, y_pred, average='weighted')
        precision = precision_score(self.y_test_encoded, y_pred, average='weighted')
        recall = recall_score(self.y_test_encoded, y_pred, average='weighted')

        print(f"\n{self.best_model_name} Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_test_original, y_pred_original))

        cm = confusion_matrix(y_test_original, y_pred_original)
        print("\nConfusion Matrix:")
        print(cm)

        self.results = {
            'best_model': self.best_model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred_original.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }

        return self.results

    def predict_risk(self, new_data, return_probabilities=True):
        """Predict rock fall risk for new data points"""
        new_data_processed = new_data.copy()

        for col in new_data_processed.columns:
            if new_data_processed[col].dtype == 'object':
                new_data_processed[col] = self.label_encoder.transform(new_data_processed[col].astype(str))

        if self.best_model_name in ['SVM', 'NeuralNetwork']:
            new_data_processed = self.scaler.transform(new_data_processed)

        prediction = self.best_model.predict(new_data_processed)
        prediction_original = self.label_encoder.inverse_transform(prediction)

        if return_probabilities:
            probabilities = self.best_model.predict_proba(new_data_processed)
            prob_dict = {}
            for i, class_label in enumerate(self.label_encoder.classes_):
                prob_dict[class_label] = probabilities[:, i]

            return prediction_original, prob_dict

        return prediction_original

    def early_warning_system(self, current_data):
        """Early warning system for immediate risk assessment"""
        risk_prediction, probabilities = self.predict_risk(current_data, return_probabilities=True)

        warnings = []
        for i, (pred, high_prob) in enumerate(zip(risk_prediction, probabilities.get('High', [0]))):
            if high_prob > self.risk_thresholds['High']:
                warnings.append({
                    'location_id': i,
                    'risk_level': 'CRITICAL',
                    'probability': high_prob,
                    'recommendation': 'Immediate evacuation and safety measures required'
                })
            elif high_prob > self.risk_thresholds['Medium']:
                warnings.append({
                    'location_id': i,
                    'risk_level': 'WARNING',
                    'probability': high_prob,
                    'recommendation': 'Enhanced monitoring and safety precautions'
                })

        return warnings

    def save_model(self, model_path="../models/rock_fall_model.pkl",
                   metadata_path="../models/model_metadata.json"):
        """Save trained model and metadata"""
        print(f"Saving model to {model_path}...")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        joblib.dump({
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'engineered_features': self.engineered_features,
            'risk_thresholds': self.risk_thresholds
        }, model_path)

        metadata = {
            'model_name': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'engineered_features': self.engineered_features,
            'performance_metrics': self.results,
            'risk_thresholds': self.risk_thresholds,
            'model_parameters': str(self.best_model.get_params()) if hasattr(self.best_model, 'get_params') else 'N/A'
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model and metadata saved successfully!")
        return model_path


def main():
    """Main training pipeline for rock fall prediction"""
    print("=== Rock Fall Prediction System for Open Pit Mines ===\n")

    predictor = RockFallPredictor(random_state=42)

    try:
        predictor.load_data(
            X_train_path="../data/processed/X_train.csv",
            X_test_path="../data/processed/X_test.csv",
            y_train_path="../data/processed/y_train.csv",
            y_test_path="../data/processed/y_test.csv"
        )
    except FileNotFoundError:
        print("Training files not found. Please provide the correct data file path.")
        return

    predictor.prepare_features()

    predictor.define_models()
    predictor.train_models(use_randomized_search=True)

    predictor.create_ensemble_model()

    predictor.analyze_feature_importance()

    results = predictor.evaluate_model()

    model_path = predictor.save_model("../models/rock_fall_prediction_model.pkl")

    print(f"\n=== Training Complete ===")
    print(f"Best Model: {predictor.best_model_name}")
    print(f"Model saved to: {model_path}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")

    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
    print("\nRock fall prediction system ready for deployment!")