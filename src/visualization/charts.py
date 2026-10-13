"""Visualization module for macro sentiment and model outputs."""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from src.utils.config import config
from src.utils.logger import logger
from src.utils.database import db
from src.feature_engineering.features import MacroFeatureEngineer
from src.modeling.trainer import MacroModelTrainer


# Set style
plt.style.use(config.settings.visualization.chart_style)
sns.set_palette(config.settings.visualization.color_palette)
pio.templates.default = "plotly_white"


class MacroVisualizer:
    """Create visualizations for macro sentiment and model outputs."""
    
    def __init__(self):
        self.fig_size = config.settings.visualization.figure_size
        self.dpi = config.settings.visualization.dpi
        self.output_formats = config.settings.visualization.output_formats
        self.engineer = MacroFeatureEngineer()
    
    def plot_surprise_timeline(self, macro_df: pd.DataFrame, 
                               save_path: str = "outputs/charts/surprise_timeline") -> None:
        """Plot macro surprise timeline with events."""
        if macro_df.empty:
            logger.warning("No macro data for surprise timeline")
            return
        
        df = macro_df.copy()
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
        # Create event features if not present
        if 'surprise' not in df.columns:
            df = self.engineer.create_event_features(df)
        
        # Plotly interactive version
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Event Surprises', 'Cumulative Surprise by Type'),
            row_heights=[0.6, 0.4]
        )
        
        # Color by surprise direction
        colors = ['green' if s > 0 else 'red' for s in df['surprise']]
        
        # Scatter plot of surprises
        fig.add_trace(
            go.Scatter(
                x=df['event_date'],
                y=df['surprise'],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=df['event_name'],
                textposition='top center',
                textfont=dict(size=9),
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Surprise: %{y:.2%}<br>Actual: %{customdata[0]}<br>Forecast: %{customdata[1]}<extra></extra>',
                customdata=df[['actual_value', 'forecast_value']].values,
                name='Surprises'
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Cumulative surprise by event type
        for event_type in df['event_name'].unique()[:8]:  # Limit to 8 types
            event_data = df[df['event_name'] == event_type].sort_values('event_date')
            cum_surprise = event_data['surprise'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=event_data['event_date'],
                    y=cum_surprise,
                    mode='lines+markers',
                    name=event_type,
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Macroeconomic Surprise Timeline',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Surprise (%)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Surprise", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        self._save_figure(fig, save_path)
    
    def plot_sentiment_heatmap(self, news_df: pd.DataFrame,
                               save_path: str = "outputs/charts/sentiment_heatmap") -> None:
        """Plot news sentiment heatmap over time."""
        if news_df.empty:
            logger.warning("No news data for sentiment heatmap")
            return
        
        df = news_df.copy()
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['date'] = df['published_at'].dt.date
        df['hour'] = df['published_at'].dt.hour
        
        # Daily sentiment pivot
        daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        # Create weekly heatmap data
        daily_sentiment['week'] = daily_sentiment['date'].dt.isocalendar().week
        daily_sentiment['year'] = daily_sentiment['date'].dt.year
        daily_sentiment['day_of_week'] = daily_sentiment['date'].dt.dayofweek
        
        # Pivot for heatmap
        heatmap_data = daily_sentiment.pivot_table(
            index='day_of_week', 
            columns='week', 
            values='sentiment_score',
            aggfunc='mean'
        )
        
        # Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="Sentiment"),
            hovertemplate='Week: %{x}<br>Day: %{y}<br>Sentiment: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='News Sentiment Heatmap (Daily, by Week)',
            xaxis_title='Week of Year',
            yaxis_title='Day of Week',
            height=400
        )
        
        self._save_figure(fig, save_path)
    
    def plot_feature_importance(self, trainer: MacroModelTrainer,
                                save_path: str = "outputs/charts/feature_importance") -> None:
        """Plot feature importance for trained models."""
        if not trainer.feature_importance:
            logger.warning("No feature importance data available")
            return
        
        # Get best model
        best_model = trainer._get_best_model()
        importance_df = trainer.feature_importance.get(best_model, pd.DataFrame())
        
        if importance_df.empty:
            return
        
        top_features = importance_df.head(20)
        
        fig = go.Figure(data=go.Bar(
            x=top_features['importance'][::-1],
            y=top_features['feature'][::-1],
            orientation='h',
            marker=dict(
                color=top_features['importance'][::-1],
                colorscale='Viridis',
                showscale=True
            ),
            hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Top 20 Feature Importance - {best_model}',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            margin=dict(l=200)
        )
        
        self._save_figure(fig, save_path)
    
    def plot_walk_forward_results(self, trainer: MacroModelTrainer,
                                   save_path: str = "outputs/charts/walk_forward") -> None:
        """Plot walk-forward validation results."""
        if not trainer.validation_results:
            logger.warning("No validation results to plot")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE by Model', 'Directional Accuracy by Model', 
                          'R² by Model', 'RMSE Stability (Std Dev)'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        models = list(trainer.validation_results.keys())
        
        # RMSE
        fig.add_trace(
            go.Bar(x=models, y=[trainer.validation_results[m]['mean_rmse'] for m in models],
                   name='RMSE', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Directional Accuracy
        fig.add_trace(
            go.Bar(x=models, y=[trainer.validation_results[m]['mean_directional_accuracy'] for m in models],
                   name='Dir. Acc.', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # R²
        fig.add_trace(
            go.Bar(x=models, y=[trainer.validation_results[m]['mean_r2'] for m in models],
                   name='R²', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # RMSE Std Dev
        fig.add_trace(
            go.Bar(x=models, y=[trainer.validation_results[m]['std_rmse'] for m in models],
                   name='RMSE Std', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Walk-Forward Validation Results',
            height=700,
            showlegend=False
        )
        
        # Add percentage formatting for directional accuracy
        fig.update_yaxes(tickformat='.1%', row=1, col=2)
        
        self._save_figure(fig, save_path)
    
    def plot_prediction_vs_actual(self, trainer: MacroModelTrainer,
                                   features_df: pd.DataFrame,
                                   target_col: str,
                                   save_path: str = "outputs/charts/pred_vs_actual") -> None:
        """Plot predictions vs actuals for last fold."""
        if not trainer.validation_results:
            return
        
        best_model = trainer._get_best_model()
        fold_results = trainer.validation_results[best_model]['all_folds']
        
        if not fold_results:
            return
        
        # Use last fold
        last_fold = fold_results[-1]
        dates = pd.to_datetime(last_fold['test_dates'])
        actuals = last_fold['actuals']
        predictions = last_fold['predictions']
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'Predictions vs Actuals - {best_model}', 'Prediction Error'),
            row_heights=[0.7, 0.3]
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(x=dates, y=actuals, mode='lines+markers', name='Actual',
                       line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=predictions, mode='lines+markers', name='Predicted',
                       line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Error
        errors = actuals - predictions
        colors = ['green' if e > 0 else 'red' for e in errors]
        fig.add_trace(
            go.Bar(x=dates, y=errors, name='Error', marker_color=colors),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Walk-Forward Predictions vs Actuals (Last Fold)',
            height=600
        )
        fig.update_yaxes(title_text="Target Value", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        self._save_figure(fig, save_path)
    
    def plot_macro_sentiment_dashboard(self, lookback_days: int = 60,
                                        save_path: str = "outputs/charts/macro_dashboard") -> None:
        """Create comprehensive macro sentiment dashboard."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # Load data
        macro_df = self.engineer.load_macro_data(start_date, end_date)
        news_df = self.engineer.load_news_data(start_date, end_date)
        
        if macro_df.empty and news_df.empty:
            logger.warning("No data for dashboard")
            return
        
        # Create features
        if not macro_df.empty:
            macro_events = self.engineer.create_event_features(macro_df)
        else:
            macro_events = pd.DataFrame()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Daily Surprise Sum', 'Surprise Distribution',
                'News Sentiment Trend', 'Sentiment vs Surprise',
                'Event Frequency by Type', 'Importance-weighted Surprise',
                'Rolling Surprise Mean (20d)', 'Surprise Volatility (20d)'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        if not macro_events.empty:
            macro_events['event_date'] = pd.to_datetime(macro_events['event_date'])
            daily_surprise = macro_events.groupby('event_date').agg({
                'surprise': 'sum',
                'weighted_surprise': 'sum',
                'importance_score': 'max',
                'event_name': 'count'
            }).reset_index()
            
            # 1. Daily Surprise Sum
            fig.add_trace(
                go.Bar(x=daily_surprise['event_date'], y=daily_surprise['surprise'],
                       name='Daily Surprise', marker_color=[
                           'green' if v > 0 else 'red' for v in daily_surprise['surprise']
                       ]),
                row=1, col=1
            )
            
            # 2. Surprise Distribution
            fig.add_trace(
                go.Histogram(x=macro_events['surprise'], nbinsx=30, name='Distribution',
                             marker_color='lightblue'),
                row=1, col=2
            )
            
            # 5. Event Frequency
            event_counts = macro_events['event_name'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=event_counts.values[::-1], y=event_counts.index[::-1],
                       orientation='h', name='Event Count', marker_color='lightgreen'),
                row=3, col=1
            )
            
            # 6. Importance-weighted Surprise
            fig.add_trace(
                go.Scatter(x=daily_surprise['event_date'], y=daily_surprise['weighted_surprise'],
                           mode='lines+markers', name='Weighted Surprise', line=dict(color='purple')),
                row=3, col=2
            )
            
            # 7. Rolling Mean
            daily_surprise['rolling_mean_20'] = daily_surprise['surprise'].rolling(20, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(x=daily_surprise['event_date'], y=daily_surprise['rolling_mean_20'],
                           mode='lines', name='20D Rolling Mean', line=dict(color='orange', width=2)),
                row=4, col=1
            )
            
            # 8. Rolling Volatility
            daily_surprise['rolling_std_20'] = daily_surprise['surprise'].rolling(20, min_periods=1).std()
            fig.add_trace(
                go.Scatter(x=daily_surprise['event_date'], y=daily_surprise['rolling_std_20'],
                           mode='lines', name='20D Rolling Vol', line=dict(color='brown', width=2)),
                row=4, col=2
            )
        
        if not news_df.empty:
            news_df['published_at'] = pd.to_datetime(news_df['published_at'])
            daily_sentiment = news_df.groupby(news_df['published_at'].dt.date).agg({
                'sentiment_score': 'mean',
                'relevance_score': 'mean'
            }).reset_index()
            daily_sentiment['published_at'] = pd.to_datetime(daily_sentiment['published_at'])
            
            # 3. News Sentiment Trend
            fig.add_trace(
                go.Scatter(x=daily_sentiment['published_at'], y=daily_sentiment['sentiment_score'],
                           mode='lines+markers', name='News Sentiment', line=dict(color='teal')),
                row=2, col=1
            )
            
            # 4. Sentiment vs Surprise (if both exist)
            if not macro_events.empty:
                merged = daily_sentiment.merge(
                    daily_surprise[['event_date', 'surprise']], 
                    left_on='published_at', right_on='event_date', how='inner'
                )
                if not merged.empty:
                    fig.add_trace(
                        go.Scatter(x=merged['sentiment_score'], y=merged['surprise'],
                                   mode='markers', name='Sentiment vs Surprise',
                                   marker=dict(size=8, color=merged['surprise'], colorscale='RdYlGn', showscale=True)),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title=f'Macro Sentiment Dashboard (Last {lookback_days} Days)',
            height=1400,
            showlegend=False
        )
        
        self._save_figure(fig, save_path)
    
    def plot_model_predictions_timeline(self, trainer: MacroModelTrainer,
                                         features_df: pd.DataFrame,
                                         save_path: str = "outputs/charts/predictions_timeline") -> None:
        """Plot model predictions over time."""
        if not trainer.validation_results:
            return
        
        best_model = trainer._get_best_model()
        fold_results = trainer.validation_results[best_model]['all_folds']
        
        if not fold_results:
            return
        
        # Collect all predictions
        all_dates = []
        all_actuals = []
        all_preds = []
        all_folds = []
        
        for i, fold in enumerate(fold_results):
            all_dates.extend(pd.to_datetime(fold['test_dates']))
            all_actuals.extend(fold['actuals'])
            all_preds.extend(fold['predictions'])
            all_folds.extend([i] * len(fold['test_dates']))
        
        pred_df = pd.DataFrame({
            'date': all_dates,
            'actual': all_actuals,
            'predicted': all_preds,
            'fold': all_folds
        }).sort_values('date')
        
        fig = go.Figure()
        
        # Actual
        fig.add_trace(go.Scatter(
            x=pred_df['date'], y=pred_df['actual'],
            mode='lines', name='Actual', line=dict(color='blue', width=2)
        ))
        
        # Predicted
        fig.add_trace(go.Scatter(
            x=pred_df['date'], y=pred_df['predicted'],
            mode='lines', name='Predicted', line=dict(color='red', width=2, dash='dash')
        ))
        
        # Fold boundaries
        for fold_num in pred_df['fold'].unique():
            fold_dates = pred_df[pred_df['fold'] == fold_num]['date']
            if len(fold_dates) > 0:
                fig.add_vline(x=fold_dates.iloc[0], line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f'Model Predictions Over Time - {best_model}',
            xaxis_title='Date',
            yaxis_title='Target Value',
            height=500,
            hovermode='x unified'
        )
        
        self._save_figure(fig, save_path)
    
    def _save_figure(self, fig, base_path: str) -> None:
        """Save figure in multiple formats."""
        for fmt in self.output_formats:
            if fmt == 'html':
                fig.write_html(f"{base_path}.html")
            elif fmt == 'png':
                fig.write_image(f"{base_path}.png", width=self.fig_size[0]*100, 
                               height=self.fig_size[1]*100, scale=2)
            elif fmt == 'pdf':
                fig.write_image(f"{base_path}.pdf")
        logger.info(f"Saved charts to {base_path}.*")


class MatplotlibVisualizer:
    """Matplotlib-based static visualizations for reports."""
    
    def __init__(self):
        self.fig_size = config.settings.visualization.figure_size
        self.dpi = config.settings.visualization.dpi
    
    def create_summary_report_charts(self, outlook: Dict[str, Any],
                                      save_path: str = "outputs/charts/summary_report") -> None:
        """Create charts for the macro summary report."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Macro Economic Outlook - Summary Charts', fontsize=16)
        
        # 1. Surprise Trend
        ax = axes[0, 0]
        ax.text(0.5, 0.5, 'Surprise Trend\n(Data from outlook)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Surprise Trend (60 Days)')
        
        # 2. Sentiment Trend
        ax = axes[0, 1]
        ax.text(0.5, 0.5, 'News Sentiment\n(Data from outlook)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('News Sentiment (14 Days)')
        
        # 3. Event Count by Type
        ax = axes[0, 2]
        event_data = outlook.get('event_analysis', {}).get('by_type', {})
        if event_data:
            events = list(event_data.keys())[:8]
            counts = list(event_data.values())[:8]
            ax.barh(events[::-1], counts[::-1], color='steelblue')
        ax.set_title('Top Event Types')
        
        # 4. Risk Factors
        ax = axes[1, 0]
        risks = outlook.get('risk_factors', [])[:5]
        y_pos = np.arange(len(risks))
        ax.barh(y_pos, [1]*len(risks), color='lightcoral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r[:40]+'...' if len(r)>40 else r for r in risks], fontsize=8)
        ax.set_title('Key Risk Factors')
        
        # 5. Trading Implications - Rates/FX
        ax = axes[1, 1]
        trading = outlook.get('trading_implications', {})
        if trading:
            assets = ['Rates', 'FX', 'Equities', 'Credit', 'Commodities']
            biases = [trading.get(a.lower(), {}).get('bias', 'neutral') for a in assets]
            colors = {'neutral to slight easing bias': 'yellow', 'neutral': 'gray', 
                      'cautiously constructive': 'lightgreen', 'constructive': 'green'}
            bar_colors = [colors.get(b, 'gray') for b in biases]
            ax.bar(assets, [1]*5, color=bar_colors)
            ax.set_yticks([])
        ax.set_title('Asset Class Bias')
        
        # 6. Scenario Probabilities (placeholder)
        ax = axes[1, 2]
        scenarios = ['Bear', 'Base', 'Bull']
        probs = [0.25, 0.55, 0.20]
        ax.bar(scenarios, probs, color=['red', 'blue', 'green'], alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Scenario Probabilities')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}.png", dpi=self.dpi, bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary charts to {save_path}.*")


def main():
    """Generate all visualizations."""
    visualizer = MacroVisualizer()
    mp_visualizer = MatplotlibVisualizer()
    
    # Load data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    macro_df = visualizer.engineer.load_macro_data(start_date, end_date)
    news_df = visualizer.engineer.load_news_data(start_date, end_date)
    
    # Generate charts
    if not macro_df.empty:
        visualizer.plot_surprise_timeline(macro_df)
    
    if not news_df.empty:
        visualizer.plot_sentiment_heatmap(news_df)
    
    # Dashboard
    visualizer.plot_macro_sentiment_dashboard(60)
    
    # Model charts (if models exist)
    try:
        trainer = MacroModelTrainer()
        trainer.load_models()
        
        # Need features for prediction charts
        features = visualizer.engineer.build_feature_matrix(start_date, end_date)
        macro_events = visualizer.engineer.create_event_features(macro_df)
        features_with_targets = trainer.prepare_targets(features, macro_events)
        
        target_col = f'target_surprise_{trainer.target_horizon}d'
        visualizer.plot_feature_importance(trainer)
        visualizer.plot_walk_forward_results(trainer)
        visualizer.plot_prediction_vs_actual(trainer, features_with_targets, target_col)
        visualizer.plot_model_predictions_timeline(trainer, features_with_targets)
    except Exception as e:
        logger.warning(f"Could not generate model charts: {e}")
    
    # Summary report charts
    from src.feature_engineering.macro_summary import MacroSummaryGenerator
    generator = MacroSummaryGenerator()
    outlook = generator.generate_outlook(60)
    mp_visualizer.create_summary_report_charts(outlook)
    
    print("All visualizations generated!")


if __name__ == "__main__":
    main()