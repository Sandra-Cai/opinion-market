"""
Time Series Forecasting Engine
Advanced time series analysis and forecasting system
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesType(Enum):
    """Time series types"""
    PRICE_SERIES = "price_series"
    VOLUME_SERIES = "volume_series"
    VOLATILITY_SERIES = "volatility_series"
    RETURN_SERIES = "return_series"
    SENTIMENT_SERIES = "sentiment_series"
    CORRELATION_SERIES = "correlation_series"

class SeasonalityType(Enum):
    """Seasonality types"""
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    MULTIPLE = "multiple"

class TrendType(Enum):
    """Trend types"""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    LOGARITHMIC = "logarithmic"

@dataclass
class TimeSeriesData:
    """Time series data structure"""
    series_id: str
    asset: str
    series_type: TimeSeriesType
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TimeSeriesAnalysis:
    """Time series analysis results"""
    analysis_id: str
    series_id: str
    stationarity_test: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    autocorrelation: Dict[str, Any]
    decomposition: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ForecastResult:
    """Forecast result"""
    forecast_id: str
    series_id: str
    model_type: str
    forecast_periods: int
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    accuracy_metrics: Dict[str, float]
    model_parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class TimeSeriesForecastingEngine:
    """Advanced Time Series Forecasting Engine"""
    
    def __init__(self):
        self.engine_id = f"timeseries_forecast_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Time series data
        self.time_series_data: Dict[str, TimeSeriesData] = {}
        self.time_series_analyses: Dict[str, TimeSeriesAnalysis] = {}
        self.forecast_results: Dict[str, ForecastResult] = {}
        
        # Model configurations
        self.model_configs = {
            "arima": {"enabled": True, "max_order": (5, 5, 5)},
            "exponential_smoothing": {"enabled": True, "trend": True, "seasonal": True},
            "linear_trend": {"enabled": True, "degree": 1},
            "polynomial_trend": {"enabled": True, "degree": 2},
            "seasonal_naive": {"enabled": True, "seasonal_period": 7}
        }
        
        # Performance tracking
        self.forecast_accuracy: Dict[str, List[float]] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Processing tasks
        self.analysis_task: Optional[asyncio.Task] = None
        self.forecasting_task: Optional[asyncio.Task] = None
        self.model_optimization_task: Optional[asyncio.Task] = None
        
        logger.info(f"Time Series Forecasting Engine {self.engine_id} initialized")

    async def start_time_series_forecasting_engine(self):
        """Start the time series forecasting engine"""
        if self.is_running:
            return
        
        logger.info("Starting Time Series Forecasting Engine...")
        
        # Initialize time series data
        await self._initialize_time_series_data()
        
        # Start processing tasks
        self.is_running = True
        
        self.analysis_task = asyncio.create_task(self._analysis_loop())
        self.forecasting_task = asyncio.create_task(self._forecasting_loop())
        self.model_optimization_task = asyncio.create_task(self._model_optimization_loop())
        
        logger.info("Time Series Forecasting Engine started")

    async def stop_time_series_forecasting_engine(self):
        """Stop the time series forecasting engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Time Series Forecasting Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.analysis_task,
            self.forecasting_task,
            self.model_optimization_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Time Series Forecasting Engine stopped")

    async def _initialize_time_series_data(self):
        """Initialize time series data"""
        try:
            # Generate mock time series data for major assets
            assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
            
            for asset in assets:
                # Generate 2 years of daily data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Generate different types of time series
                for series_type in TimeSeriesType:
                    series_id = f"{asset}_{series_type.value}"
                    
                    if series_type == TimeSeriesType.PRICE_SERIES:
                        values = self._generate_price_series(len(dates))
                    elif series_type == TimeSeriesType.VOLUME_SERIES:
                        values = self._generate_volume_series(len(dates))
                    elif series_type == TimeSeriesType.VOLATILITY_SERIES:
                        values = self._generate_volatility_series(len(dates))
                    elif series_type == TimeSeriesType.RETURN_SERIES:
                        values = self._generate_return_series(len(dates))
                    elif series_type == TimeSeriesType.SENTIMENT_SERIES:
                        values = self._generate_sentiment_series(len(dates))
                    else:
                        values = self._generate_correlation_series(len(dates))
                    
                    time_series = TimeSeriesData(
                        series_id=series_id,
                        asset=asset,
                        series_type=series_type,
                        timestamps=dates.tolist(),
                        values=values,
                        metadata={
                            "frequency": "daily",
                            "length": len(values),
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat()
                        }
                    )
                    
                    self.time_series_data[series_id] = time_series
            
            logger.info(f"Initialized {len(self.time_series_data)} time series")
            
        except Exception as e:
            logger.error(f"Error initializing time series data: {e}")

    def _generate_price_series(self, length: int) -> List[float]:
        """Generate realistic price series"""
        try:
            # Start with base price
            base_price = np.random.uniform(50, 500)
            prices = [base_price]
            
            # Generate price movements with trend and volatility
            trend = np.random.uniform(-0.001, 0.001)  # Daily trend
            volatility = np.random.uniform(0.01, 0.05)  # Daily volatility
            
            for i in range(1, length):
                # Add trend and random walk
                price_change = np.random.normal(trend, volatility)
                new_price = prices[-1] * (1 + price_change)
                prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            return prices
            
        except Exception as e:
            logger.error(f"Error generating price series: {e}")
            return [100.0] * length

    def _generate_volume_series(self, length: int) -> List[float]:
        """Generate realistic volume series"""
        try:
            # Generate log-normal volume with some seasonality
            base_volume = np.random.uniform(1000000, 10000000)
            volumes = []
            
            for i in range(length):
                # Add weekly seasonality
                day_of_week = i % 7
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)
                
                # Generate volume with some randomness
                volume = base_volume * seasonal_factor * np.random.lognormal(0, 0.5)
                volumes.append(volume)
            
            return volumes
            
        except Exception as e:
            logger.error(f"Error generating volume series: {e}")
            return [1000000.0] * length

    def _generate_volatility_series(self, length: int) -> List[float]:
        """Generate realistic volatility series"""
        try:
            # Generate volatility with clustering (GARCH-like behavior)
            base_volatility = np.random.uniform(0.01, 0.05)
            volatilities = [base_volatility]
            
            for i in range(1, length):
                # Volatility clustering
                prev_vol = volatilities[-1]
                vol_change = np.random.normal(0, 0.1)
                new_vol = prev_vol * (1 + vol_change)
                
                # Keep volatility within reasonable bounds
                new_vol = max(0.001, min(0.2, new_vol))
                volatilities.append(new_vol)
            
            return volatilities
            
        except Exception as e:
            logger.error(f"Error generating volatility series: {e}")
            return [0.02] * length

    def _generate_return_series(self, length: int) -> List[float]:
        """Generate realistic return series"""
        try:
            # Generate returns with fat tails and some autocorrelation
            returns = []
            
            for i in range(length):
                # Generate return with some momentum
                if i > 0:
                    momentum = 0.1 * returns[-1]  # Small momentum effect
                else:
                    momentum = 0
                
                # Add random component
                random_component = np.random.normal(0, 0.02)
                
                # Combine momentum and random component
                return_value = momentum + random_component
                returns.append(return_value)
            
            return returns
            
        except Exception as e:
            logger.error(f"Error generating return series: {e}")
            return [0.0] * length

    def _generate_sentiment_series(self, length: int) -> List[float]:
        """Generate realistic sentiment series"""
        try:
            # Generate sentiment with mean reversion
            sentiment = []
            current_sentiment = np.random.uniform(-0.5, 0.5)
            
            for i in range(length):
                # Mean reversion
                mean_reversion = -0.1 * current_sentiment
                
                # Random component
                random_component = np.random.normal(0, 0.1)
                
                # Update sentiment
                current_sentiment += mean_reversion + random_component
                
                # Keep within bounds
                current_sentiment = max(-1.0, min(1.0, current_sentiment))
                sentiment.append(current_sentiment)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error generating sentiment series: {e}")
            return [0.0] * length

    def _generate_correlation_series(self, length: int) -> List[float]:
        """Generate realistic correlation series"""
        try:
            # Generate correlation with some persistence
            correlations = []
            current_correlation = np.random.uniform(-0.5, 0.5)
            
            for i in range(length):
                # Persistence
                persistence = 0.8 * current_correlation
                
                # Random component
                random_component = np.random.normal(0, 0.1)
                
                # Update correlation
                current_correlation = persistence + random_component
                
                # Keep within bounds
                current_correlation = max(-1.0, min(1.0, current_correlation))
                correlations.append(current_correlation)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error generating correlation series: {e}")
            return [0.0] * length

    async def _analysis_loop(self):
        """Time series analysis loop"""
        while self.is_running:
            try:
                # Analyze time series that haven't been analyzed yet
                for series_id, time_series in self.time_series_data.items():
                    if series_id not in self.time_series_analyses:
                        await self._analyze_time_series(series_id)
                
                await asyncio.sleep(3600)  # Analyze every hour
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(3600)

    async def _forecasting_loop(self):
        """Forecasting loop"""
        while self.is_running:
            try:
                # Generate forecasts for all time series
                for series_id, time_series in self.time_series_data.items():
                    await self._generate_forecast(series_id)
                
                await asyncio.sleep(1800)  # Forecast every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in forecasting loop: {e}")
                await asyncio.sleep(1800)

    async def _model_optimization_loop(self):
        """Model optimization loop"""
        while self.is_running:
            try:
                # Optimize model parameters
                for series_id in self.time_series_data:
                    await self._optimize_model_parameters(series_id)
                
                await asyncio.sleep(7200)  # Optimize every 2 hours
                
            except Exception as e:
                logger.error(f"Error in model optimization loop: {e}")
                await asyncio.sleep(7200)

    async def _analyze_time_series(self, series_id: str):
        """Analyze a time series"""
        try:
            if series_id not in self.time_series_data:
                return
            
            time_series = self.time_series_data[series_id]
            
            # Convert to pandas Series for analysis
            df = pd.DataFrame({
                'timestamp': time_series.timestamps,
                'value': time_series.values
            })
            df.set_index('timestamp', inplace=True)
            series = df['value']
            
            # Stationarity tests
            stationarity_test = await self._test_stationarity(series)
            
            # Seasonality analysis
            seasonality_analysis = await self._analyze_seasonality(series)
            
            # Trend analysis
            trend_analysis = await self._analyze_trend(series)
            
            # Autocorrelation analysis
            autocorrelation = await self._analyze_autocorrelation(series)
            
            # Decomposition
            decomposition = await self._decompose_time_series(series)
            
            # Create analysis result
            analysis = TimeSeriesAnalysis(
                analysis_id=f"analysis_{secrets.token_hex(8)}",
                series_id=series_id,
                stationarity_test=stationarity_test,
                seasonality_analysis=seasonality_analysis,
                trend_analysis=trend_analysis,
                autocorrelation=autocorrelation,
                decomposition=decomposition,
                metadata={
                    "analysis_date": datetime.now().isoformat(),
                    "series_length": len(series),
                    "frequency": "daily"
                }
            )
            
            self.time_series_analyses[series_id] = analysis
            
            logger.info(f"Analyzed time series {series_id}")
            
        except Exception as e:
            logger.error(f"Error analyzing time series {series_id}: {e}")

    async def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test stationarity of time series"""
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            
            # KPSS test
            kpss_result = kpss(series.dropna(), regression='c')
            
            return {
                "adf_test": {
                    "statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "critical_values": adf_result[4],
                    "is_stationary": adf_result[1] < 0.05
                },
                "kpss_test": {
                    "statistic": kpss_result[0],
                    "p_value": kpss_result[1],
                    "critical_values": kpss_result[3],
                    "is_stationary": kpss_result[1] > 0.05
                },
                "overall_stationary": adf_result[1] < 0.05 and kpss_result[1] > 0.05
            }
            
        except Exception as e:
            logger.error(f"Error testing stationarity: {e}")
            return {"error": str(e)}

    async def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonality in time series"""
        try:
            # Check for different seasonal periods
            seasonal_periods = [7, 30, 90, 365]  # Daily, weekly, monthly, yearly
            seasonality_detected = []
            
            for period in seasonal_periods:
                if len(series) > period * 2:
                    # Calculate autocorrelation at seasonal lag
                    autocorr = series.autocorr(lag=period)
                    
                    if abs(autocorr) > 0.3:  # Threshold for seasonality
                        seasonality_detected.append({
                            "period": period,
                            "autocorrelation": autocorr,
                            "strength": abs(autocorr)
                        })
            
            return {
                "seasonality_detected": seasonality_detected,
                "primary_seasonality": seasonality_detected[0] if seasonality_detected else None,
                "has_seasonality": len(seasonality_detected) > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {e}")
            return {"error": str(e)}

    async def _analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series"""
        try:
            # Linear trend
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # Trend strength
            trend_strength = abs(r_value)
            
            # Trend direction
            if slope > 0:
                trend_direction = "increasing"
            elif slope < 0:
                trend_direction = "decreasing"
            else:
                trend_direction = "flat"
            
            return {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "is_significant": p_value < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {"error": str(e)}

    async def _analyze_autocorrelation(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze autocorrelation in time series"""
        try:
            # Calculate autocorrelation for different lags
            max_lags = min(50, len(series) // 4)
            autocorrelations = []
            
            for lag in range(1, max_lags + 1):
                autocorr = series.autocorr(lag=lag)
                if not np.isnan(autocorr):
                    autocorrelations.append({
                        "lag": lag,
                        "autocorrelation": autocorr
                    })
            
            # Find significant autocorrelations
            significant_lags = [ac for ac in autocorrelations if abs(ac["autocorrelation"]) > 0.2]
            
            return {
                "autocorrelations": autocorrelations,
                "significant_lags": significant_lags,
                "max_autocorrelation": max([abs(ac["autocorrelation"]) for ac in autocorrelations]) if autocorrelations else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing autocorrelation: {e}")
            return {"error": str(e)}

    async def _decompose_time_series(self, series: pd.Series) -> Dict[str, Any]:
        """Decompose time series into components"""
        try:
            if len(series) < 24:  # Need at least 2 periods for decomposition
                return {"error": "Insufficient data for decomposition"}
            
            # Seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=7)
            
            return {
                "trend": decomposition.trend.dropna().tolist(),
                "seasonal": decomposition.seasonal.dropna().tolist(),
                "residual": decomposition.resid.dropna().tolist(),
                "observed": decomposition.observed.dropna().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error decomposing time series: {e}")
            return {"error": str(e)}

    async def _generate_forecast(self, series_id: str):
        """Generate forecast for a time series"""
        try:
            if series_id not in self.time_series_data:
                return
            
            time_series = self.time_series_data[series_id]
            
            # Convert to pandas Series
            df = pd.DataFrame({
                'timestamp': time_series.timestamps,
                'value': time_series.values
            })
            df.set_index('timestamp', inplace=True)
            series = df['value']
            
            # Generate forecasts using different models
            forecast_periods = 30  # 30 days ahead
            
            # ARIMA forecast
            if self.model_configs["arima"]["enabled"]:
                await self._generate_arima_forecast(series_id, series, forecast_periods)
            
            # Exponential smoothing forecast
            if self.model_configs["exponential_smoothing"]["enabled"]:
                await self._generate_exponential_smoothing_forecast(series_id, series, forecast_periods)
            
            # Linear trend forecast
            if self.model_configs["linear_trend"]["enabled"]:
                await self._generate_linear_trend_forecast(series_id, series, forecast_periods)
            
        except Exception as e:
            logger.error(f"Error generating forecast for {series_id}: {e}")

    async def _generate_arima_forecast(self, series_id: str, series: pd.Series, periods: int):
        """Generate ARIMA forecast"""
        try:
            # Auto-select ARIMA order (simplified)
            order = (1, 1, 1)  # Default order
            
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=periods)
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            
            # Create forecast result
            forecast_result = ForecastResult(
                forecast_id=f"arima_{secrets.token_hex(8)}",
                series_id=series_id,
                model_type="arima",
                forecast_periods=periods,
                forecast_values=forecast.tolist(),
                confidence_intervals=[(row[0], row[1]) for row in conf_int.values],
                accuracy_metrics={
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "log_likelihood": fitted_model.llf
                },
                model_parameters={
                    "order": order,
                    "params": fitted_model.params.tolist()
                }
            )
            
            self.forecast_results[f"{series_id}_arima"] = forecast_result
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast for {series_id}: {e}")

    async def _generate_exponential_smoothing_forecast(self, series_id: str, series: pd.Series, periods: int):
        """Generate exponential smoothing forecast"""
        try:
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                series, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=7
            )
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=periods)
            
            # Calculate confidence intervals (simplified)
            std_error = np.std(series) * 0.1
            conf_int = [
                (forecast.iloc[i] - 1.96 * std_error, forecast.iloc[i] + 1.96 * std_error)
                for i in range(periods)
            ]
            
            # Create forecast result
            forecast_result = ForecastResult(
                forecast_id=f"exp_smooth_{secrets.token_hex(8)}",
                series_id=series_id,
                model_type="exponential_smoothing",
                forecast_periods=periods,
                forecast_values=forecast.tolist(),
                confidence_intervals=conf_int,
                accuracy_metrics={
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "sse": fitted_model.sse
                },
                model_parameters={
                    "alpha": fitted_model.params.get("smoothing_level", 0.5),
                    "beta": fitted_model.params.get("smoothing_trend", 0.5),
                    "gamma": fitted_model.params.get("smoothing_seasonal", 0.5)
                }
            )
            
            self.forecast_results[f"{series_id}_exp_smooth"] = forecast_result
            
        except Exception as e:
            logger.error(f"Error generating exponential smoothing forecast for {series_id}: {e}")

    async def _generate_linear_trend_forecast(self, series_id: str, series: pd.Series, periods: int):
        """Generate linear trend forecast"""
        try:
            # Fit linear trend
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # Generate forecast
            future_x = np.arange(len(series), len(series) + periods)
            forecast = slope * future_x + intercept
            
            # Calculate confidence intervals
            conf_int = []
            for i, x_val in enumerate(future_x):
                std_error = std_err * np.sqrt(1 + 1/len(series) + (x_val - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                lower = forecast[i] - 1.96 * std_error
                upper = forecast[i] + 1.96 * std_error
                conf_int.append((lower, upper))
            
            # Create forecast result
            forecast_result = ForecastResult(
                forecast_id=f"linear_trend_{secrets.token_hex(8)}",
                series_id=series_id,
                model_type="linear_trend",
                forecast_periods=periods,
                forecast_values=forecast.tolist(),
                confidence_intervals=conf_int,
                accuracy_metrics={
                    "r_squared": r_value ** 2,
                    "p_value": p_value,
                    "std_error": std_err
                },
                model_parameters={
                    "slope": slope,
                    "intercept": intercept,
                    "r_value": r_value
                }
            )
            
            self.forecast_results[f"{series_id}_linear_trend"] = forecast_result
            
        except Exception as e:
            logger.error(f"Error generating linear trend forecast for {series_id}: {e}")

    async def _optimize_model_parameters(self, series_id: str):
        """Optimize model parameters"""
        try:
            # This would implement parameter optimization
            # For now, we'll just log that optimization is happening
            logger.debug(f"Optimizing parameters for {series_id}")
            
        except Exception as e:
            logger.error(f"Error optimizing parameters for {series_id}: {e}")

    # Public API methods
    async def get_time_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Get time series data"""
        try:
            if series_id not in self.time_series_data:
                return None
            
            time_series = self.time_series_data[series_id]
            
            return {
                "series_id": time_series.series_id,
                "asset": time_series.asset,
                "series_type": time_series.series_type.value,
                "timestamps": [ts.isoformat() for ts in time_series.timestamps],
                "values": time_series.values,
                "metadata": time_series.metadata,
                "created_at": time_series.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting time series {series_id}: {e}")
            return None

    async def get_time_series_analysis(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Get time series analysis"""
        try:
            if series_id not in self.time_series_analyses:
                return None
            
            analysis = self.time_series_analyses[series_id]
            
            return {
                "analysis_id": analysis.analysis_id,
                "series_id": analysis.series_id,
                "stationarity_test": analysis.stationarity_test,
                "seasonality_analysis": analysis.seasonality_analysis,
                "trend_analysis": analysis.trend_analysis,
                "autocorrelation": analysis.autocorrelation,
                "decomposition": analysis.decomposition,
                "metadata": analysis.metadata,
                "created_at": analysis.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting time series analysis {series_id}: {e}")
            return None

    async def get_forecast_results(self, series_id: str) -> List[Dict[str, Any]]:
        """Get forecast results for a time series"""
        try:
            results = []
            
            for forecast_id, forecast in self.forecast_results.items():
                if forecast.series_id == series_id:
                    results.append({
                        "forecast_id": forecast.forecast_id,
                        "series_id": forecast.series_id,
                        "model_type": forecast.model_type,
                        "forecast_periods": forecast.forecast_periods,
                        "forecast_values": forecast.forecast_values,
                        "confidence_intervals": forecast.confidence_intervals,
                        "accuracy_metrics": forecast.accuracy_metrics,
                        "model_parameters": forecast.model_parameters,
                        "created_at": forecast.created_at.isoformat()
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting forecast results for {series_id}: {e}")
            return []

    async def get_available_time_series(self) -> List[Dict[str, Any]]:
        """Get available time series"""
        try:
            time_series_list = []
            
            for series_id, time_series in self.time_series_data.items():
                time_series_list.append({
                    "series_id": series_id,
                    "asset": time_series.asset,
                    "series_type": time_series.series_type.value,
                    "length": len(time_series.values),
                    "start_date": time_series.timestamps[0].isoformat() if time_series.timestamps else None,
                    "end_date": time_series.timestamps[-1].isoformat() if time_series.timestamps else None,
                    "has_analysis": series_id in self.time_series_analyses,
                    "has_forecasts": any(f.series_id == series_id for f in self.forecast_results.values())
                })
            
            return time_series_list
            
        except Exception as e:
            logger.error(f"Error getting available time series: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_time_series": len(self.time_series_data),
                "total_analyses": len(self.time_series_analyses),
                "total_forecasts": len(self.forecast_results),
                "model_configs": self.model_configs,
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

# Global instance
time_series_forecasting_engine = TimeSeriesForecastingEngine()
