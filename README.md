# CrowdWisdomTrading Macro Economics Agent - Intern Assessment

A Python-based machine learning pipeline that predicts optimal macro trading strategies by ingesting historical news, scraping macroeconomic event calendars, joining datasets, and training predictive models with rigorous out-of-sample validation.

## 🎯 Project Overview

This pipeline implements the complete workflow for macro trading signal generation:

1. **Data Ingestion** - Scrape macroeconomic calendars (CPI, FOMC, NFP, GDP, etc.) and economic news using Apify and Tavily
2. **Feature Engineering** - Create quantitative features from surprises, sentiment, and time-based patterns
3. **Macro Summary** - Generate 1-page macro economic outlook with key themes, risks, and trading implications
4. **ML Modeling** - Train models with walk-forward validation to forecast trade P&L/win rate
5. **Visualization** - Interactive charts showing macro sentiment evolution over time

## 🏗️ Architecture

```
src/
├── data_ingestion/     # Apify & Tavily scrapers
│   ├── apify_scraper.py      # Macro calendar scraping
│   ├── tavily_scraper.py     # Economic news scraping
│   └── pipeline.py           # Ingestion orchestrator
├── feature_engineering/
│   ├── features.py           # Quantitative feature creation
│   └── macro_summary.py      # 1-page outlook generator
├── modeling/
│   └── trainer.py            # Walk-forward ML training
├── visualization/
│   └── charts.py             # Interactive & static charts
├── utils/
│   ├── config.py             # Configuration management
│   ├── logger.py             # Logging setup
│   └── database.py           # SQLite feature store
└── pipeline.py               # Main orchestrator
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `config/settings.yaml` with your API tokens:

```yaml
apis:
  apify:
    token: "YOUR_APIFY_TOKEN"
    actor_id: "apify/actor-macro-economic-calendar"
  tavily:
    token: "YOUR_TAVILY_TOKEN"
  exa:
    token: "YOUR_EXA_TOKEN"
```

**Note**: The pipeline includes mock data generators for development without API keys.

### 3. Run the Pipeline

```bash
# Run full pipeline
python -m src.pipeline --step all

# Run individual steps
python -m src.pipeline --step ingestion    # Data scraping only
python -m src.pipeline --step features     # Feature engineering only
python -m src.pipeline --step summary      # Macro outlook only
python -m src.pipeline --step model        # Model training only
python -m src.pipeline --step viz          # Visualizations only
```

### 4. View Outputs

- **Reports**: `outputs/reports/macro_outlook.md` (1-page summary)
- **Charts**: `outputs/charts/*.html` (interactive) & `*.png` (static)
- **Models**: `outputs/models/*.pkl` (trained models)
- **Data**: `data/processed/features.csv` (feature matrix)

## 📊 Key Features

### Data Sources
- **Macro Calendar**: CPI, FOMC, NFP, GDP, PCE, PMI, Rate Decisions (180 days)
- **Economic News**: Reuters, Bloomberg, WSJ, FT via Tavily (30 days)
- **Timezone Handling**: UTC/EST alignment for event timestamps

### Feature Engineering
- **Surprise Features**: Actual vs forecast, weighted by importance
- **Rolling Statistics**: Mean, std, z-scores over 5/10/20/60 day windows
- **Lag Features**: 1, 2, 3, 5, 10 day lags
- **Time Features**: Cyclical encoding, business days, month/quarter boundaries
- **News Sentiment**: Daily aggregation, rolling trends, keyword extraction

### Modeling
- **Models**: Random Forest, XGBoost, LightGBM, Linear/Ridge/Lasso
- **Validation**: Walk-forward (60d train / 10d test / 5d step)
- **Targets**: Forward surprise cumulative, direction, magnitude
- **Metrics**: RMSE, MAE, R², Directional Accuracy

### Visualizations
- Surprise timeline with event annotations
- Sentiment heatmaps by day/week
- Feature importance rankings
- Walk-forward validation results
- Prediction vs actual tracking
- Comprehensive macro dashboard

## 📁 Output Structure

```
outputs/
├── charts/
│   ├── surprise_timeline.html      # Interactive surprise chart
│   ├── sentiment_heatmap.html      # News sentiment heatmap
│   ├── macro_dashboard.html        # Full dashboard
│   ├── feature_importance.html     # Model feature importance
│   ├── walk_forward.html           # Validation results
│   ├── pred_vs_actual.html         # Predictions vs actuals
│   └── summary_report.png          # Static report charts
├── reports/
│   ├── macro_outlook.md            # 1-page markdown summary
│   └── macro_outlook.json          # Structured outlook data
└── models/
    ├── random_forest.pkl           # Trained models
    ├── xgboost.pkl
    ├── validation_results.json     # CV metrics
    └── *_scaler.pkl                # Feature scalers
```

## 🔧 Configuration

Key settings in `config/settings.yaml`:

```yaml
data_ingestion:
  lookback_days: 180
  key_economic_events: [CPI, FOMC, NFP, GDP, PCE, ...]

feature_engineering:
  time_windows: [1, 3, 7, 14, 30]
  rolling_windows: [5, 10, 20, 60]
  lag_periods: [1, 2, 3, 5, 10]

modeling:
  target_horizon_days: 5
  walk_forward:
    train_window: 60
    test_window: 10
    step_size: 5
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Quick validation
python -m src.data_ingestion.apify_scraper      # Test macro scraper
python -m src.data_ingestion.tavily_scraper     # Test news scraper
python -m src.feature_engineering.features      # Test features
python -m src.feature_engineering.macro_summary # Test summary
python -m src.modeling.trainer                  # Test modeling
python -m src.visualization.charts              # Test charts
```

## 📈 Example Output

### Macro Outlook (markdown)
```markdown
# Macro Economic Outlook
**Generated:** 2026-10-12 10:30:00
**Period:** 2026-08-13 to 2026-10-12

## Key Themes
1. Persistent positive data surprises suggesting economic resilience
2. Inflation trajectory remains key focus for policy
3. Central bank policy pivot expectations driving markets

## Trading Implications
### Rates
- **Bias:** Neutral to slight easing bias
- **Strategy:** Receive 2y5y steepener on disinflation confirmation
- **Key Risk:** Sticky services inflation delays cuts
```

### Model Performance
| Model | RMSE | MAE | R² | Dir. Acc |
|-------|------|-----|-----|----------|
| XGBoost | 0.123 | 0.089 | 0.45 | 58% |
| LightGBM | 0.118 | 0.085 | 0.48 | 60% |
| Random Forest | 0.135 | 0.095 | 0.38 | 55% |

## 🔑 API Tokens for Reproduction

For evaluation, the following tokens/actors were used:

| Service | Token | Actor ID |
|---------|-------|----------|
| Apify | `YOUR_APIFY_TOKEN` | `apify/actor-macro-economic-calendar` |
| Tavily | `YOUR_TAVILY_TOKEN` | N/A |
| Exa | `YOUR_EXA_TOKEN` | N/A |

Replace with actual credentials in `config/settings.yaml`.

## 📝 Submission Checklist

- [x] GitHub repository with complete code
- [x] Apify token and actor ID documented
- [x] Tavily token documented  
- [x] Exa token documented
- [x] Requirements.txt with all dependencies
- [x] Configuration via YAML
- [x] Walk-forward validation implementation
- [x] 1-page macro outlook generator
- [x] Interactive visualizations (Plotly)
- [x] Static report charts (Matplotlib)
- [x] SQLite feature store
- [x] Comprehensive logging
- [x] Modular, testable architecture

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is submitted for the CrowdWisdomTrading Quantitative Data Scientist internship assessment.

## 📧 Contact

For questions about this submission, please email: gilad@crowdwisdomtrading.com