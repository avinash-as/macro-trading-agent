"""Main pipeline orchestrator for macro trading ML system."""
import argparse
import sys
from datetime import datetime
from typing import Dict, Any
from src.utils.config import config
from src.utils.logger import logger, setup_logging
from src.data_ingestion.pipeline import DataIngestionPipeline
from src.feature_engineering.features import MacroFeatureEngineer
from src.feature_engineering.macro_summary import MacroSummaryGenerator
from src.modeling.trainer import MacroModelTrainer
from src.visualization.charts import MacroVisualizer, MatplotlibVisualizer


class MacroTradingPipeline:
    """Main pipeline orchestrating all components."""
    
    def __init__(self):
        self.ingestion = DataIngestionPipeline()
        self.engineer = MacroFeatureEngineer()
        self.summary_generator = MacroSummaryGenerator()
        self.trainer = MacroModelTrainer()
        self.visualizer = MacroVisualizer()
        self.mp_visualizer = MatplotlibVisualizer()
    
    def run_ingestion(self, lookback_days: int = 180, news_days: int = 30) -> Dict[str, Any]:
        """Run data ingestion step."""
        logger.info("=" * 50)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 50)
        
        results = self.ingestion.run_full_ingestion(lookback_days, news_days)
        
        logger.info(f"Ingestion complete: {results['macro_events']} events, {results['news_articles']} articles")
        return results
    
    def run_feature_engineering(self, lookback_days: int = 180) -> Dict[str, Any]:
        """Run feature engineering step."""
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        features = self.engineer.build_feature_matrix(start_date, end_date)
        self.engineer.save_features(features)
        
        macro_df = self.engineer.load_macro_data(start_date, end_date)
        macro_events = self.engineer.create_event_features(macro_df)
        
        logger.info(f"Feature matrix: {features.shape[0]} rows, {features.shape[1]} features")
        return {
            "features_shape": features.shape,
            "feature_columns": list(features.columns),
            "date_range": [start_date, end_date]
        }
    
    def run_macro_summary(self, lookback_days: int = 60) -> Dict[str, Any]:
        """Generate macro economic outlook summary."""
        logger.info("=" * 50)
        logger.info("STEP 3: MACRO ECONOMIC OUTLOOK")
        logger.info("=" * 50)
        
        outlook = self.summary_generator.generate_outlook(lookback_days)
        
        # Save reports
        md_path = self.summary_generator.save_report(outlook, "markdown")
        json_path = self.summary_generator.save_report(outlook, "json")
        
        logger.info(f"Macro outlook saved to {md_path} and {json_path}")
        return outlook
    
    def run_modeling(self, lookback_days: int = 180) -> Dict[str, Any]:
        """Run model training and validation."""
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL TRAINING & VALIDATION")
        logger.info("=" * 50)
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # Load features
        features = self.engineer.build_feature_matrix(start_date, end_date)
        macro_df = self.engineer.load_macro_data(start_date, end_date)
        macro_events = self.engineer.create_event_features(macro_df)
        
        # Prepare targets and train
        features_with_targets = self.trainer.prepare_targets(features, macro_events)
        
        target_col = f'target_surprise_{self.trainer.target_horizon}d'
        results = self.trainer.walk_forward_validation(features_with_targets, target_col)
        
        # Save models
        self.trainer.save_models()
        
        # Latest prediction
        prediction = self.trainer.predict_latest(features_with_targets)
        
        logger.info(f"Modeling complete. Best model: {self.trainer._get_best_model()}")
        return {
            "validation_results": results,
            "best_model": self.trainer._get_best_model(),
            "latest_prediction": prediction
        }
    
    def run_visualization(self, lookback_days: int = 60) -> Dict[str, Any]:
        """Generate all visualizations."""
        logger.info("=" * 50)
        logger.info("STEP 5: VISUALIZATION")
        logger.info("=" * 50)
        
        # Load data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        macro_df = self.engineer.load_macro_data(start_date, end_date)
        news_df = self.engineer.load_news_data(start_date, end_date)
        
        generated = []
        
        if not macro_df.empty:
            self.visualizer.plot_surprise_timeline(macro_df)
            generated.append("surprise_timeline")
        
        if not news_df.empty:
            self.visualizer.plot_sentiment_heatmap(news_df)
            generated.append("sentiment_heatmap")
        
        # Dashboard
        self.visualizer.plot_macro_sentiment_dashboard(lookback_days)
        generated.append("macro_dashboard")
        
        # Model charts
        try:
            self.trainer.load_models()
            features = self.engineer.build_feature_matrix(start_date, end_date)
            macro_events = self.engineer.create_event_features(macro_df)
            features_with_targets = self.trainer.prepare_targets(features, macro_events)
            
            target_col = f'target_surprise_{self.trainer.target_horizon}d'
            
            self.visualizer.plot_feature_importance(self.trainer)
            generated.append("feature_importance")
            
            self.visualizer.plot_walk_forward_results(self.trainer)
            generated.append("walk_forward")
            
            self.visualizer.plot_prediction_vs_actual(self.trainer, features_with_targets, target_col)
            generated.append("pred_vs_actual")
            
            self.visualizer.plot_model_predictions_timeline(self.trainer, features_with_targets)
            generated.append("predictions_timeline")
        except Exception as e:
            logger.warning(f"Model charts skipped: {e}")
        
        # Summary report charts
        outlook = self.summary_generator.generate_outlook(lookback_days)
        self.mp_visualizer.create_summary_report_charts(outlook)
        generated.append("summary_report")
        
        logger.info(f"Visualizations generated: {generated}")
        return {"generated_charts": generated}
    
    def run_full_pipeline(self, lookback_days: int = 180, news_days: int = 30) -> Dict[str, Any]:
        """Run complete pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING MACRO TRADING ML PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        results = {}
        
        try:
            # Step 1: Ingestion
            results["ingestion"] = self.run_ingestion(lookback_days, news_days)
            
            # Step 2: Feature Engineering
            results["features"] = self.run_feature_engineering(lookback_days)
            
            # Step 3: Macro Summary
            results["outlook"] = self.run_macro_summary(60)
            
            # Step 4: Modeling
            results["modeling"] = self.run_modeling(lookback_days)
            
            # Step 5: Visualization
            results["visualization"] = self.run_visualization(60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results["error"] = str(e)
            raise
        
        end_time = datetime.now()
        results["total_duration_seconds"] = (end_time - start_time).total_seconds()
        results["completed_at"] = end_time.isoformat()
        
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETED IN {results['total_duration_seconds']:.1f} SECONDS")
        logger.info("=" * 60)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Macro Trading ML Pipeline")
    parser.add_argument("--step", choices=["all", "ingestion", "features", "summary", "model", "viz"],
                       default="all", help="Pipeline step to run")
    parser.add_argument("--lookback", type=int, default=180, help="Lookback days for data")
    parser.add_argument("--news-days", type=int, default=30, help="News lookback days")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load config
    config.load_config(args.config)
    
    pipeline = MacroTradingPipeline()
    
    if args.step == "all":
        results = pipeline.run_full_pipeline(args.lookback, args.news_days)
    elif args.step == "ingestion":
        results = pipeline.run_ingestion(args.lookback, args.news_days)
    elif args.step == "features":
        results = pipeline.run_feature_engineering(args.lookback)
    elif args.step == "summary":
        results = pipeline.run_macro_summary(60)
    elif args.step == "model":
        results = pipeline.run_modeling(args.lookback)
    elif args.step == "viz":
        results = pipeline.run_visualization(60)
    
    print("\n=== PIPELINE RESULTS ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    import json
    from datetime import timedelta
    main()