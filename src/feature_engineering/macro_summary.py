"""Macro economic outlook summary generator."""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from src.utils.config import config
from src.utils.logger import logger
from src.utils.database import db
from src.feature_engineering.features import MacroFeatureEngineer


class MacroSummaryGenerator:
    """Generate 1-page macro economic outlook summary."""
    
    def __init__(self):
        self.engineer = MacroFeatureEngineer()
    
    def generate_outlook(self, lookback_days: int = 60) -> Dict[str, Any]:
        """Generate comprehensive macro outlook."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # Load data
        macro_df = self.engineer.load_macro_data(start_date, end_date)
        news_df = self.engineer.load_news_data(start_date, end_date)
        
        # Create event features
        macro_events = self.engineer.create_event_features(macro_df)
        
        outlook = {
            "generated_at": datetime.now().isoformat(),
            "period": f"{start_date} to {end_date}",
            "key_themes": self._identify_key_themes(macro_events, news_df),
            "event_analysis": self._analyze_events(macro_events),
            "surprise_analysis": self._analyze_surprises(macro_events),
            "sentiment_analysis": self._analyze_sentiment(news_df),
            "forward_look": self._generate_forward_look(macro_events, news_df),
            "risk_factors": self._identify_risks(macro_events, news_df),
            "trading_implications": self._generate_trading_implications(macro_events, news_df)
        }
        
        return outlook
    
    def _identify_key_themes(self, macro_events: pd.DataFrame, news_df: pd.DataFrame) -> List[str]:
        """Identify key macro themes."""
        themes = []
        
        if macro_events.empty:
            return ["Insufficient data for theme identification"]
        
        # Recent surprises
        recent = macro_events[macro_events['event_date'] >= (datetime.now() - timedelta(days=30))]
        
        if not recent.empty:
            avg_surprise = recent['surprise'].mean()
            if avg_surprise > 0.1:
                themes.append("Persistent positive data surprises suggesting economic resilience")
            elif avg_surprise < -0.1:
                themes.append("Negative data surprises mounting, growth concerns rising")
            else:
                themes.append("Data broadly in line with expectations, mixed signals")
        
        # Event frequency
        high_importance = recent[recent['importance'] == 'High']
        if len(high_importance) > 5:
            themes.append("Heavy high-impact event calendar driving volatility")
        
        # News sentiment
        if not news_df.empty:
            recent_news = news_df[news_df['published_at'] >= (datetime.now() - timedelta(days=7))]
            if not recent_news.empty:
                avg_sentiment = recent_news['sentiment_score'].mean()
                if avg_sentiment > 0.2:
                    themes.append("Positive news sentiment prevailing")
                elif avg_sentiment < -0.2:
                    themes.append("Negative news sentiment dominating")
        
        # Specific event themes
        event_types = macro_events['event_name'].unique()
        if 'CPI' in str(event_types) or 'PCE' in str(event_types):
            themes.append("Inflation trajectory remains key focus for policy")
        if 'FOMC' in str(event_types) or 'Rate Decision' in str(event_types):
            themes.append("Central bank policy pivot expectations driving markets")
        if 'NFP' in str(event_types) or 'Unemployment' in str(event_types):
            themes.append("Labor market strength/weakness critical for policy path")
        
        return themes[:5]  # Top 5 themes
    
    def _analyze_events(self, macro_events: pd.DataFrame) -> Dict[str, Any]:
        """Analyze macro events."""
        if macro_events.empty:
            return {"message": "No events in period"}
        
        recent = macro_events[macro_events['event_date'] >= (datetime.now() - timedelta(days=30))]
        
        return {
            "total_events": len(macro_events),
            "recent_events_30d": len(recent),
            "high_importance_events": len(macro_events[macro_events['importance'] == 'High']),
            "by_country": macro_events['country'].value_counts().to_dict(),
            "by_type": macro_events['event_name'].value_counts().head(10).to_dict(),
            "upcoming_high_impact": self._get_upcoming_events(macro_events)
        }
    
    def _get_upcoming_events(self, macro_events: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get upcoming high-impact events."""
        # In real implementation, this would use forward-looking calendar
        # For now, return recent high-impact events as proxy
        recent_high = macro_events[
            (macro_events['importance'] == 'High') & 
            (macro_events['event_date'] >= (datetime.now() - timedelta(days=7)))
        ].sort_values('event_date', ascending=False).head(5)
        
        return recent_high[['event_name', 'event_date', 'country', 'actual_value', 'forecast_value']].to_dict('records')
    
    def _analyze_surprises(self, macro_events: pd.DataFrame) -> Dict[str, Any]:
        """Analyze surprise patterns."""
        if macro_events.empty or 'surprise' not in macro_events.columns:
            return {"message": "No surprise data available"}
        
        recent = macro_events[macro_events['event_date'] >= (datetime.now() - timedelta(days=60))]
        
        return {
            "mean_surprise_60d": float(recent['surprise'].mean()),
            "surprise_volatility_60d": float(recent['surprise'].std()),
            "positive_surprise_pct": float((recent['surprise'] > 0).mean() * 100),
            "large_surprises": recent[recent['surprise'].abs() > 0.2][
                ['event_name', 'event_date', 'surprise', 'actual_value', 'forecast_value']
            ].to_dict('records'),
            "surprise_trend": "improving" if recent['surprise'].rolling(5).mean().iloc[-1] > recent['surprise'].rolling(5).mean().iloc[-6] else "deteriorating"
        }
    
    def _analyze_sentiment(self, news_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze news sentiment."""
        if news_df.empty:
            return {"message": "No news data available"}
        
        recent = news_df[news_df['published_at'] >= (datetime.now() - timedelta(days=14))]
        
        return {
            "mean_sentiment_14d": float(recent['sentiment_score'].mean()) if not recent.empty else 0,
            "sentiment_trend_14d": self._calc_sentiment_trend(recent),
            "article_count_14d": len(recent),
            "top_sources": recent['source'].value_counts().head(5).to_dict() if not recent.empty else {},
            "key_keywords": self._get_top_keywords(recent)
        }
    
    def _calc_sentiment_trend(self, news_df: pd.DataFrame) -> str:
        """Calculate sentiment trend."""
        if len(news_df) < 10:
            return "insufficient_data"
        
        daily = news_df.groupby('published_at')['sentiment_score'].mean()
        if len(daily) < 3:
            return "insufficient_data"
        
        recent_avg = daily.tail(3).mean()
        earlier_avg = daily.head(3).mean()
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "deteriorating"
        return "stable"
    
    def _get_top_keywords(self, news_df: pd.DataFrame) -> List[str]:
        """Extract top keywords from news."""
        all_keywords = []
        for kw_str in news_df['keywords'].dropna():
            try:
                keywords = json.loads(kw_str) if isinstance(kw_str, str) else kw_str
                all_keywords.extend(keywords)
            except:
                pass
        
        if not all_keywords:
            return []
        
        keyword_counts = pd.Series(all_keywords).value_counts()
        return keyword_counts.head(10).index.tolist()
    
    def _generate_forward_look(self, macro_events: pd.DataFrame, news_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate forward-looking assessment."""
        return {
            "policy_outlook": self._assess_policy_outlook(macro_events),
            "growth_outlook": self._assess_growth_outlook(macro_events),
            "inflation_outlook": self._assess_inflation_outlook(macro_events),
            "key_dates": self._get_key_dates(),
            "scenario_analysis": {
                "base_case": "Moderate growth, gradual disinflation, policy easing begins",
                "bull_case": "Strong growth, rapid inflation normalization, soft landing achieved",
                "bear_case": "Growth stall, sticky inflation, policy error risk"
            }
        }
    
    def _assess_policy_outlook(self, macro_events: pd.DataFrame) -> str:
        """Assess monetary policy outlook."""
        if macro_events.empty:
            return "Data insufficient"
        
        # Look at recent rate decisions and guidance
        rate_events = macro_events[macro_events['event_name'].str.contains('Rate|FOMC|ECB|BoE|BoJ', case=False, na=False)]
        
        if rate_events.empty:
            return "No recent rate decisions, policy in wait-and-see mode"
        
        latest = rate_events.sort_values('event_date').iloc[-1]
        
        if 'cut' in str(latest.get('event_name', '')).lower() or latest.get('surprise', 0) < 0:
            return "Easing cycle underway or imminent"
        elif 'hike' in str(latest.get('event_name', '')).lower() or latest.get('surprise', 0) > 0:
            return "Tightening bias or further hikes possible"
        return "Policy on hold, data dependent"
    
    def _assess_growth_outlook(self, macro_events: pd.DataFrame) -> str:
        """Assess growth outlook."""
        if macro_events.empty:
            return "Data insufficient"
        
        growth_events = macro_events[macro_events['event_name'].str.contains('GDP|PMI|ISM|Industrial|Retail', case=False, na=False)]
        
        if growth_events.empty:
            return "Mixed signals, monitoring high-frequency data"
        
        recent = growth_events[growth_events['event_date'] >= (datetime.now() - timedelta(days=60))]
        avg_surprise = recent['surprise'].mean() if not recent.empty else 0
        
        if avg_surprise > 0.1:
            return "Growth momentum positive, above-trend expansion"
        elif avg_surprise < -0.1:
            return "Growth momentum fading, downside risks"
        return "Growth tracking near potential, balanced risks"
    
    def _assess_inflation_outlook(self, macro_events: pd.DataFrame) -> str:
        """Assess inflation outlook."""
        if macro_events.empty:
            return "Data insufficient"
        
        infl_events = macro_events[macro_events['event_name'].str.contains('CPI|PCE|Inflation', case=False, na=False)]
        
        if infl_events.empty:
            return "Monitoring pipeline pressures"
        
        recent = infl_events[infl_events['event_date'] >= (datetime.now() - timedelta(days=60))]
        avg_surprise = recent['surprise'].mean() if not recent.empty else 0
        
        if avg_surprise > 0.1:
            return "Inflation stickier than expected, upside risks"
        elif avg_surprise < -0.1:
            return "Disinflation progressing faster than expected"
        return "Inflation normalizing gradually toward target"
    
    def _get_key_dates(self) -> List[Dict[str, str]]:
        """Get key upcoming dates (placeholder)."""
        # In production, this would come from forward calendar
        return [
            {"date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"), "event": "Next FOMC Meeting", "importance": "High"},
            {"date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"), "event": "CPI Release", "importance": "High"},
            {"date": (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d"), "event": "PCE Release", "importance": "High"},
            {"date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"), "event": "NFP Release", "importance": "High"},
        ]
    
    def _identify_risks(self, macro_events: pd.DataFrame, news_df: pd.DataFrame) -> List[str]:
        """Identify key risk factors."""
        risks = []
        
        if not macro_events.empty:
            recent = macro_events[macro_events['event_date'] >= (datetime.now() - timedelta(days=30))]
            if not recent.empty:
                surprise_vol = recent['surprise'].std()
                if surprise_vol > 0.3:
                    risks.append("High data volatility increasing policy uncertainty")
                
                neg_surprises = (recent['surprise'] < -0.2).sum()
                if neg_surprises > 2:
                    risks.append("Cluster of negative surprises raising recession probability")
        
        if not news_df.empty:
            recent_news = news_df[news_df['published_at'] >= (datetime.now() - timedelta(days=14))]
            if not recent_news.empty:
                neg_sentiment = (recent_news['sentiment_score'] < -0.3).sum()
                if neg_sentiment > len(recent_news) * 0.3:
                    risks.append("Deteriorating news flow sentiment")
        
        # Standard macro risks
        risks.extend([
            "Policy lag effects: tightening impact still feeding through",
            "Geopolitical escalation risk (energy, trade, supply chains)",
            "Financial stability risks from rate hikes (CRE, sovereign debt)",
            "China growth slowdown spillovers",
            "Fiscal sustainability concerns in major economies"
        ])
        
        return risks[:7]
    
    def _generate_trading_implications(self, macro_events: pd.DataFrame, news_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading implications."""
        return {
            "rates": {
                "bias": "neutral to slight easing bias",
                "strategy": "Receive 2y5y steepener on disinflation confirmation",
                "risk": "Sticky services inflation delays cuts"
            },
            "fx": {
                "bias": "USD moderate weakness on easing expectations",
                "strategy": "Long EUR/USD, AUD/USD on carry + growth differential",
                "risk": "US exceptionalism persists"
            },
            "equities": {
                "bias": "Cautiously constructive on soft landing",
                "strategy": "Quality growth, equal-weight over cap-weight",
                "risk": "Earnings revision cycle turns negative"
            },
            "credit": {
                "bias": "Neutral, prefer IG over HY",
                "strategy": "Short duration IG, avoid CCC",
                "risk": "Default cycle acceleration"
            },
            "commodities": {
                "bias": "Constructive energy, neutral metals",
                "strategy": "Long oil calls, copper on green demand",
                "risk": "Global demand destruction"
            }
        }
    
    def format_as_markdown(self, outlook: Dict[str, Any]) -> str:
        """Format outlook as markdown report."""
        md = f"""# Macro Economic Outlook
**Generated:** {outlook['generated_at'][:19].replace('T', ' ')}  
**Period:** {outlook['period']}

---

## Key Themes
"""
        for i, theme in enumerate(outlook['key_themes'], 1):
            md += f"{i}. {theme}\n"
        
        md += f"""

## Event Analysis (Last 60 Days)
- **Total Events:** {outlook['event_analysis'].get('total_events', 'N/A')}
- **Recent (30d):** {outlook['event_analysis'].get('recent_events_30d', 'N/A')}
- **High Impact:** {outlook['event_analysis'].get('high_importance_events', 'N/A')}

### By Country
"""
        for country, count in outlook['event_analysis'].get('by_country', {}).items():
            md += f"- {country}: {count}\n"
        
        md += f"""

### By Event Type
"""
        for event, count in outlook['event_analysis'].get('by_type', {}).items():
            md += f"- {event}: {count}\n"
        
        md += f"""

## Surprise Analysis (60 Days)
- **Mean Surprise:** {outlook['surprise_analysis'].get('mean_surprise_60d', 'N/A'):.3f}
- **Surprise Volatility:** {outlook['surprise_analysis'].get('surprise_volatility_60d', 'N/A'):.3f}
- **Positive Surprise %:** {outlook['surprise_analysis'].get('positive_surprise_pct', 'N/A'):.1f}%
- **Trend:** {outlook['surprise_analysis'].get('surprise_trend', 'N/A')}

### Large Surprises (>20%)
"""
        for surprise in outlook['surprise_analysis'].get('large_surprises', [])[:5]:
            md += f"- **{surprise['event_name']}** ({surprise['event_date']}): Actual {surprise['actual_value']} vs Forecast {surprise['forecast_value']} (Surprise: {surprise['surprise']:.2%})\n"
        
        md += f"""

## News Sentiment (14 Days)
- **Mean Sentiment:** {outlook['sentiment_analysis'].get('mean_sentiment_14d', 'N/A'):.3f}
- **Trend:** {outlook['sentiment_analysis'].get('sentiment_trend_14d', 'N/A')}
- **Articles:** {outlook['sentiment_analysis'].get('article_count_14d', 'N/A')}

### Top Keywords
"""
        for kw in outlook['sentiment_analysis'].get('key_keywords', []):
            md += f"- {kw}\n"
        
        md += f"""

## Forward Look

### Policy Outlook
{outlook['forward_look']['policy_outlook']}

### Growth Outlook
{outlook['forward_look']['growth_outlook']}

### Inflation Outlook
{outlook['forward_look']['inflation_outlook']}

### Key Upcoming Dates
"""
        for date_info in outlook['forward_look']['key_dates']:
            md += f"- **{date_info['date']}**: {date_info['event']} ({date_info['importance']})\n"
        
        md += f"""

### Scenario Analysis
- **Base Case:** {outlook['forward_look']['scenario_analysis']['base_case']}
- **Bull Case:** {outlook['forward_look']['scenario_analysis']['bull_case']}
- **Bear Case:** {outlook['forward_look']['scenario_analysis']['bear_case']}

## Risk Factors
"""
        for i, risk in enumerate(outlook['risk_factors'], 1):
            md += f"{i}. {risk}\n"
        
        md += f"""

## Trading Implications

### Rates
- **Bias:** {outlook['trading_implications']['rates']['bias']}
- **Strategy:** {outlook['trading_implications']['rates']['strategy']}
- **Key Risk:** {outlook['trading_implications']['rates']['risk']}

### FX
- **Bias:** {outlook['trading_implications']['fx']['bias']}
- **Strategy:** {outlook['trading_implications']['fx']['strategy']}
- **Key Risk:** {outlook['trading_implications']['fx']['risk']}

### Equities
- **Bias:** {outlook['trading_implications']['equities']['bias']}
- **Strategy:** {outlook['trading_implications']['equities']['strategy']}
- **Key Risk:** {outlook['trading_implications']['equities']['risk']}

### Credit
- **Bias:** {outlook['trading_implications']['credit']['bias']}
- **Strategy:** {outlook['trading_implications']['credit']['strategy']}
- **Key Risk:** {outlook['trading_implications']['credit']['risk']}

### Commodities
- **Bias:** {outlook['trading_implications']['commodities']['bias']}
- **Strategy:** {outlook['trading_implications']['commodities']['strategy']}
- **Key Risk:** {outlook['trading_implications']['commodities']['risk']}

---

*Disclaimer: This is an automated quantitative assessment for research purposes only. Not investment advice.*
"""
        return md
    
    def save_report(self, outlook: Dict[str, Any], format: str = "markdown") -> str:
        """Save outlook report."""
        if format == "markdown":
            content = self.format_as_markdown(outlook)
            filepath = "outputs/reports/macro_outlook.md"
        else:
            content = json.dumps(outlook, indent=2, default=str)
            filepath = "outputs/reports/macro_outlook.json"
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Saved macro outlook to {filepath}")
        return filepath


def main():
    """Generate and save macro outlook."""
    generator = MacroSummaryGenerator()
    outlook = generator.generate_outlook(lookback_days=60)
    
    # Save both formats
    generator.save_report(outlook, "markdown")
    generator.save_report(outlook, "json")
    
    print("Macro outlook generated successfully!")
    print(generator.format_as_markdown(outlook)[:2000] + "...")


if __name__ == "__main__":
    main()