"""Machine learning model with walk-forward validation for macro trading."""
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import joblib
from src.utils.config import config
from src.utils.logger import logger
from src.utils.database import db
from src.feature_engineering.features import MacroFeatureEngineer


warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """Walk-forward validation for time series models."""
    
    def __init__(self, train_window: int = 60, test_window: int = 10, step_size: int = 5):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
    
    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for walk-forward validation."""
        indices = []
        start = 0
        
        while start + self.train_window + self.test_window <= n_samples:
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            train_idx = np.arange(start, train_end)
            test_idx = np.arange(train_end, test_end)
            
            indices.append((train_idx, test_idx))
            start += self.step_size
        
        return indices


class MacroModelTrainer:
    """Train and validate macro trading models."""
    
    def __init__(self):
        self.target_horizon = config.settings.modeling.target_horizon_days
        self.wfv = WalkForwardValidator(
            train_window=config.settings.modeling.train_window,
            test_window=config.settings.modeling.test_window,
            step_size=config.settings.modeling.step_size
        )
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.validation_results = {}
    
    def prepare_targets(self, features_df: pd.DataFrame, 
                       macro_df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for modeling."""
        df = features_df.copy()
        df['feature_date'] = pd.to_datetime(df['feature_date'])
        
        # Create forward returns/surprises as targets
        # Target 1: Next N-day cumulative surprise
        if not macro_df.empty:
            macro_df = macro_df.copy()
            macro_df['event_date'] = pd.to_datetime(macro_df['event_date'])
            
            # Daily surprise sum
            daily_surprise = macro_df.groupby('event_date')['surprise'].sum().reset_index()
            daily_surprise.columns = ['feature_date', 'daily_surprise']
            
            # Merge with features
            df = df.merge(daily_surprise, on='feature_date', how='left')
            df['daily_surprise'] = df['daily_surprise'].fillna(0)
            
            # Forward cumulative surprise over horizon
            df[f'target_surprise_{self.target_horizon}d'] = (
                df['daily_surprise']
                .rolling(self.target_horizon, min_periods=1)
                .sum()
                .shift(-self.target_horizon)
            )
            
            # Target 2: Directional accuracy (win rate proxy)
            df[f'target_direction_{self.target_horizon}d'] = (
                df[f'target_surprise_{self.target_horizon}d'] > 0
            ).astype(int)
            
            # Target 3: Magnitude (abs surprise)
            df[f'target_magnitude_{self.target_horizon}d'] = (
                df[f'target_surprise_{self.target_horizon}d'].abs()
            )
        
        # If no macro data, create synthetic targets based on features
        if f'target_surprise_{self.target_horizon}d' not in df.columns:
            # Use weighted surprise as proxy target
            surprise_cols = [c for c in df.columns if 'surprise' in c and 'lag' not in c]
            if surprise_cols:
                df[f'target_surprise_{self.target_horizon}d'] = df[surprise_cols[0]].shift(-self.target_horizon)
            else:
                # Random walk target
                df[f'target_surprise_{self.target_horizon}d'] = np.random.randn(len(df)) * 0.1
        
        return df
    
    def get_model(self, model_name: str):
        """Get model instance by name."""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        return models.get(model_name, models['random_forest'])
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, Any]:
        """Train a single model and evaluate."""
        model = self.get_model(model_name)
        
        # Scale features for linear models
        if model_name in ['linear_regression', 'ridge', 'lasso']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'directional_accuracy': ((y_test > 0) == (y_pred > 0)).mean()
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance[model_name] = importance
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_test.values
        }
    
    def walk_forward_validation(self, features_df: pd.DataFrame, 
                                target_col: str) -> Dict[str, Any]:
        """Run walk-forward validation for all models."""
        logger.info(f"Starting walk-forward validation for target: {target_col}")
        
        # Prepare features and target
        feature_cols = [c for c in features_df.columns if c not in [
            'feature_date', target_col, 
            f'target_surprise_{self.target_horizon}d',
            f'target_direction_{self.target_horizon}d',
            f'target_magnitude_{self.target_horizon}d'
        ]]
        
        # Remove non-numeric columns
        X = features_df[feature_cols].select_dtypes(include=[np.number])
        y = features_df[target_col]
        
        # Align and drop NaN
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        
        if len(X) < self.wfv.train_window + self.wfv.test_window:
            logger.warning(f"Insufficient data for walk-forward validation: {len(X)} samples")
            return {}
        
        results = {model_name: [] for model_name in config.settings.modeling.models}
        
        # Get splits
        splits = self.wfv.split(len(X))
        logger.info(f"Running {len(splits)} walk-forward folds")
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            fold_results = {}
            for model_name in config.settings.modeling.models:
                try:
                    result = self.train_single_model(model_name, X_train, y_train, X_test, y_test)
                    result['fold'] = fold
                    result['train_dates'] = features_df.iloc[train_idx]['feature_date'].tolist()
                    result['test_dates'] = features_df.iloc[test_idx]['feature_date'].tolist()
                    results[model_name].append(result)
                except Exception as e:
                    logger.warning(f"Model {model_name} failed on fold {fold}: {e}")
        
        # Aggregate results
        aggregated = {}
        for model_name, fold_results in results.items():
            if not fold_results:
                continue
            
            aggregated[model_name] = {
                'mean_mse': np.mean([r['metrics']['mse'] for r in fold_results]),
                'mean_rmse': np.mean([r['metrics']['rmse'] for r in fold_results]),
                'mean_mae': np.mean([r['metrics']['mae'] for r in fold_results]),
                'mean_r2': np.mean([r['metrics']['r2'] for r in fold_results]),
                'mean_directional_accuracy': np.mean([r['metrics']['directional_accuracy'] for r in fold_results]),
                'std_rmse': np.std([r['metrics']['rmse'] for r in fold_results]),
                'all_folds': fold_results
            }
            
            # Store best model (last fold)
            self.models[model_name] = fold_results[-1]['model']
        
        self.validation_results = aggregated
        logger.info(f"Walk-forward validation complete. Best model: {self._get_best_model()}")
        
        return aggregated
    
    def _get_best_model(self) -> str:
        """Get best model by directional accuracy."""
        if not self.validation_results:
            return "random_forest"
        return max(self.validation_results.keys(), 
                   key=lambda k: self.validation_results[k]['mean_directional_accuracy'])
    
    def predict_latest(self, features_df: pd.DataFrame, 
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """Make predictions on latest data."""
        if model_name is None:
            model_name = self._get_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        # Prepare features
        feature_cols = [c for c in features_df.columns if c not in [
            'feature_date', 
            f'target_surprise_{self.target_horizon}d',
            f'target_direction_{self.target_horizon}d',
            f'target_magnitude_{self.target_horizon}d'
        ]]
        
        X = features_df[feature_cols].select_dtypes(include=[np.number])
        X = X.fillna(0)
        
        # Get latest row
        latest_X = X.iloc[[-1]]
        latest_date = features_df.iloc[-1]['feature_date']
        
        # Predict
        if model_name in self.scalers:
            latest_X = self.scalers[model_name].transform(latest_X)
        
        prediction = model.predict(latest_X)[0]
        
        # Get feature importance
        importance = self.feature_importance.get(model_name, pd.DataFrame())
        top_features = importance.head(10).to_dict('records') if not importance.empty else []
        
        return {
            'model': model_name,
            'prediction_date': latest_date,
            'target_horizon_days': self.target_horizon,
            'predicted_surprise': float(prediction),
            'predicted_direction': 'positive' if prediction > 0 else 'negative',
            'confidence': 'high' if abs(prediction) > 0.2 else 'medium' if abs(prediction) > 0.1 else 'low',
            'top_features': top_features
        }
    
    def save_models(self, path: str = "outputs/models") -> None:
        """Save trained models."""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"{path}/{name}.pkl")
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{path}/{name}_scaler.pkl")
        
        # Save validation results
        with open(f"{path}/validation_results.json", 'w') as f:
            # Convert numpy types to native
            serializable = {}
            for k, v in self.validation_results.items():
                serializable[k] = {k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2 
                                   for k2, v2 in v.items() if k2 != 'all_folds'}
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str = "outputs/models") -> None:
        """Load trained models."""
        import os
        for model_file in os.listdir(path):
            if model_file.endswith('.pkl') and not model_file.endswith('_scaler.pkl'):
                name = model_file.replace('.pkl', '')
                self.models[name] = joblib.load(f"{path}/{model_file}")
            
            elif model_file.endswith('_scaler.pkl'):
                name = model_file.replace('_scaler.pkl', '')
                self.scalers[name] = joblib.load(f"{path}/{model_file}")
        
        logger.info(f"Loaded {len(self.models)} models from {path}")


def main():
    """Run model training and validation."""
    # Load features
    engineer = MacroFeatureEngineer()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    features = engineer.build_feature_matrix(start_date, end_date)
    macro_df = engineer.load_macro_data(start_date, end_date)
    macro_events = engineer.create_event_features(macro_df)
    
    # Prepare targets
    trainer = MacroModelTrainer()
    features_with_targets = trainer.prepare_targets(features, macro_events)
    
    # Run walk-forward validation
    target_col = f'target_surprise_{trainer.target_horizon}d'
    results = trainer.walk_forward_validation(features_with_targets, target_col)
    
    # Print results
    print("\n=== Walk-Forward Validation Results ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['mean_rmse']:.4f} ± {metrics['std_rmse']:.4f}")
        print(f"  MAE:  {metrics['mean_mae']:.4f}")
        print(f"  R²:   {metrics['mean_r2']:.4f}")
        print(f"  Dir. Acc: {metrics['mean_directional_accuracy']:.2%}")
    
    # Predict latest
    if trainer.models:
        prediction = trainer.predict_latest(features_with_targets)
        print(f"\n=== Latest Prediction ===")
        print(f"Model: {prediction['model']}")
        print(f"Date: {prediction['prediction_date']}")
        print(f"Predicted Surprise: {prediction['predicted_surprise']:.4f}")
        print(f"Direction: {prediction['predicted_direction']}")
        print(f"Confidence: {prediction['confidence']}")
    
    # Save models
    trainer.save_models()
    
    # Save features with targets
    features_with_targets.to_csv("data/processed/features_with_targets.csv", index=False)


if __name__ == "__main__":
    main()