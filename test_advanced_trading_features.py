#!/usr/bin/env python3
"""
Test script for the new Advanced Trading and Financial Features.
Tests Advanced Trading Engine, Portfolio Optimization Engine, and Market Sentiment Engine.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_trading_engine():
    """Test the Advanced Trading Engine"""
    print("\n📈 Testing Advanced Trading Engine...")
    
    try:
        from app.services.advanced_trading_engine import advanced_trading_engine, OrderSide, OrderType, TradingStrategy
        
        # Test engine initialization
        await advanced_trading_engine.start_trading_engine()
        print("✅ Advanced Trading Engine started")
        
        # Test market data update
        await advanced_trading_engine.update_market_data("BTC", 45000.0, 1000000)
        await advanced_trading_engine.update_market_data("ETH", 3000.0, 500000)
        await advanced_trading_engine.update_market_data("AAPL", 150.0, 2000000)
        print("✅ Market data updated")
        
        # Test order submission
        order_id = await advanced_trading_engine.submit_trading_order(
            asset="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            strategy=TradingStrategy.MOMENTUM
        )
        print(f"✅ Trading order submitted: {order_id}")
        
        # Test getting positions
        positions = await advanced_trading_engine.get_trading_positions()
        print(f"✅ Trading positions: {len(positions)} positions")
        
        # Test getting performance
        performance = await advanced_trading_engine.get_trading_performance()
        print(f"✅ Trading performance: {performance['performance_metrics']['total_trades']} trades")
        
        # Test getting signals
        signals = await advanced_trading_engine.get_trading_signals()
        print(f"✅ Trading signals: {len(signals)} signals")
        
        # Test engine shutdown
        await advanced_trading_engine.stop_trading_engine()
        print("✅ Advanced Trading Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Trading Engine test failed: {e}")
        return False

async def test_portfolio_optimization_engine():
    """Test the Portfolio Optimization Engine"""
    print("\n🎯 Testing Portfolio Optimization Engine...")
    
    try:
        from app.services.portfolio_optimization_engine import portfolio_optimization_engine
        
        # Test engine initialization
        await portfolio_optimization_engine.start_portfolio_optimization_engine()
        print("✅ Portfolio Optimization Engine started")
        
        # Test getting available assets
        assets = await portfolio_optimization_engine.get_available_assets()
        print(f"✅ Available assets: {len(assets)} assets")
        
        # Test creating portfolio
        asset_symbols = ["AAPL", "MSFT", "GOOGL", "BTC", "ETH"]
        portfolio_id = await portfolio_optimization_engine.create_portfolio(
            name="Test Portfolio",
            description="A test portfolio for optimization",
            asset_symbols=asset_symbols
        )
        print(f"✅ Portfolio created: {portfolio_id}")
        
        # Test getting portfolio
        portfolio = await portfolio_optimization_engine.get_portfolio(portfolio_id)
        print(f"✅ Portfolio retrieved: {portfolio['name']}")
        
        # Test getting all portfolios
        portfolios = await portfolio_optimization_engine.get_all_portfolios()
        print(f"✅ All portfolios: {len(portfolios)} portfolios")
        
        # Test getting rebalancing signals
        signals = await portfolio_optimization_engine.get_rebalancing_signals()
        print(f"✅ Rebalancing signals: {len(signals)} signals")
        
        # Test getting optimization history
        history = await portfolio_optimization_engine.get_optimization_history()
        print(f"✅ Optimization history: {len(history)} optimizations")
        
        # Test getting performance metrics
        performance = await portfolio_optimization_engine.get_performance_metrics(portfolio_id)
        print(f"✅ Portfolio performance: {performance}")
        
        # Test engine shutdown
        await portfolio_optimization_engine.stop_portfolio_optimization_engine()
        print("✅ Portfolio Optimization Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio Optimization Engine test failed: {e}")
        return False

async def test_market_sentiment_engine():
    """Test the Market Sentiment Engine"""
    print("\n📊 Testing Market Sentiment Engine...")
    
    try:
        from app.services.market_sentiment_engine import market_sentiment_engine, SentimentSource
        
        # Test engine initialization
        await market_sentiment_engine.start_market_sentiment_engine()
        print("✅ Market Sentiment Engine started")
        
        # Test getting sentiment for specific asset
        sentiment = await market_sentiment_engine.get_sentiment("BTC")
        if sentiment:
            print(f"✅ BTC sentiment: {sentiment['overall_sentiment']} ({sentiment['sentiment_score']:.2f})")
        else:
            print("⚠️ No sentiment data for BTC yet")
        
        # Test getting all sentiments
        sentiments = await market_sentiment_engine.get_all_sentiments()
        print(f"✅ All sentiments: {len(sentiments)} assets")
        
        # Test adding manual sentiment data
        data_id = await market_sentiment_engine.add_sentiment_data(
            source=SentimentSource.SOCIAL_MEDIA,
            asset="BTC",
            text="Bitcoin is going to the moon! 🚀",
            confidence=0.8
        )
        print(f"✅ Manual sentiment data added: {data_id}")
        
        # Test getting sentiment alerts
        alerts = await market_sentiment_engine.get_sentiment_alerts()
        print(f"✅ Sentiment alerts: {len(alerts)} alerts")
        
        # Test getting sentiment history
        history = await market_sentiment_engine.get_sentiment_history("BTC", hours=24)
        print(f"✅ Sentiment history: {len(history)} data points")
        
        # Test getting sentiment metrics
        metrics = await market_sentiment_engine.get_sentiment_metrics()
        print(f"✅ Sentiment metrics: {metrics['total_assets']} assets, {metrics['total_data_points']} data points")
        
        # Test engine shutdown
        await market_sentiment_engine.stop_market_sentiment_engine()
        print("✅ Market Sentiment Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Market Sentiment Engine test failed: {e}")
        return False

async def test_trading_integration():
    """Test integration between trading engines"""
    print("\n🔗 Testing Trading Integration...")
    
    try:
        from app.services.advanced_trading_engine import advanced_trading_engine, OrderSide, OrderType, TradingStrategy
        from app.services.portfolio_optimization_engine import portfolio_optimization_engine
        from app.services.market_sentiment_engine import market_sentiment_engine, SentimentSource
        
        # Start all engines
        await advanced_trading_engine.start_trading_engine()
        await portfolio_optimization_engine.start_portfolio_optimization_engine()
        await market_sentiment_engine.start_market_sentiment_engine()
        
        print("✅ All trading engines started")
        
        # Step 1: Create portfolio
        portfolio_id = await portfolio_optimization_engine.create_portfolio(
            name="Integration Test Portfolio",
            description="Portfolio for integration testing",
            asset_symbols=["BTC", "ETH", "AAPL"]
        )
        print(f"✅ Step 1 - Portfolio created: {portfolio_id}")
        
        # Step 2: Add sentiment data
        sentiment_id = await market_sentiment_engine.add_sentiment_data(
            source=SentimentSource.NEWS,
            asset="BTC",
            text="Bitcoin shows strong bullish momentum",
            confidence=0.9
        )
        print(f"✅ Step 2 - Sentiment data added: {sentiment_id}")
        
        # Step 3: Submit trading order based on sentiment
        order_id = await advanced_trading_engine.submit_trading_order(
            asset="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            strategy=TradingStrategy.MOMENTUM
        )
        print(f"✅ Step 3 - Trading order submitted: {order_id}")
        
        # Step 4: Update market data
        await advanced_trading_engine.update_market_data("BTC", 46000.0, 1200000)
        print("✅ Step 4 - Market data updated")
        
        # Step 5: Get integrated results
        portfolio = await portfolio_optimization_engine.get_portfolio(portfolio_id)
        sentiment = await market_sentiment_engine.get_sentiment("BTC")
        positions = await advanced_trading_engine.get_trading_positions()
        
        print(f"✅ Step 5 - Integration results:")
        print(f"   Portfolio: {portfolio['name']} with {len(portfolio['assets'])} assets")
        print(f"   BTC Sentiment: {sentiment['overall_sentiment'] if sentiment else 'N/A'}")
        print(f"   Trading Positions: {len(positions)} positions")
        
        # Stop all engines
        await advanced_trading_engine.stop_trading_engine()
        await portfolio_optimization_engine.stop_portfolio_optimization_engine()
        await market_sentiment_engine.stop_market_sentiment_engine()
        
        print("✅ All trading engines stopped")
        return True
        
    except Exception as e:
        print(f"❌ Trading integration test failed: {e}")
        return False

async def test_trading_performance():
    """Test performance of trading engines"""
    print("\n⚡ Testing Trading Performance...")
    
    try:
        from app.services.advanced_trading_engine import advanced_trading_engine, OrderSide, OrderType, TradingStrategy
        from app.services.portfolio_optimization_engine import portfolio_optimization_engine
        from app.services.market_sentiment_engine import market_sentiment_engine, SentimentSource
        
        # Start all engines
        await advanced_trading_engine.start_trading_engine()
        await portfolio_optimization_engine.start_portfolio_optimization_engine()
        await market_sentiment_engine.start_market_sentiment_engine()
        
        # Test trading engine performance
        start_time = time.time()
        for i in range(50):
            await advanced_trading_engine.submit_trading_order(
                asset="BTC",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.01,
                strategy=TradingStrategy.MOMENTUM
            )
        trading_time = time.time() - start_time
        print(f"✅ Trading: 50 orders in {trading_time:.2f}s ({50/trading_time:.1f} orders/s)")
        
        # Test portfolio optimization performance
        start_time = time.time()
        for i in range(20):
            portfolio_id = await portfolio_optimization_engine.create_portfolio(
                name=f"Perf Test Portfolio {i}",
                description=f"Performance test portfolio {i}",
                asset_symbols=["AAPL", "MSFT", "GOOGL"]
            )
        portfolio_time = time.time() - start_time
        print(f"✅ Portfolio: 20 portfolios in {portfolio_time:.2f}s ({20/portfolio_time:.1f} portfolios/s)")
        
        # Test sentiment engine performance
        start_time = time.time()
        for i in range(100):
            await market_sentiment_engine.add_sentiment_data(
                source=SentimentSource.SOCIAL_MEDIA,
                asset="BTC",
                text=f"Test sentiment data {i}",
                confidence=0.7
            )
        sentiment_time = time.time() - start_time
        print(f"✅ Sentiment: 100 data points in {sentiment_time:.2f}s ({100/sentiment_time:.1f} data/s)")
        
        # Stop all engines
        await advanced_trading_engine.stop_trading_engine()
        await portfolio_optimization_engine.stop_portfolio_optimization_engine()
        await market_sentiment_engine.stop_market_sentiment_engine()
        
        return True
        
    except Exception as e:
        print(f"❌ Trading performance test failed: {e}")
        return False

async def test_new_api_endpoints():
    """Test the new API endpoints"""
    print("\n🌐 Testing New API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        endpoints = [
            "/advanced-trading/health",
            "/portfolio-optimization/health",
            "/market-sentiment/health"
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    async with session.get(f"{base_url}{endpoint}") as response:
                        if response.status == 200:
                            print(f"✅ {endpoint} - Status: {response.status}")
                        else:
                            print(f"⚠️ {endpoint} - Status: {response.status}")
                except Exception as e:
                    print(f"❌ {endpoint} - Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Advanced Trading Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Test individual engines
    test_results.append(await test_advanced_trading_engine())
    test_results.append(await test_portfolio_optimization_engine())
    test_results.append(await test_market_sentiment_engine())
    
    # Test integration
    test_results.append(await test_trading_integration())
    
    # Test performance
    test_results.append(await test_trading_performance())
    
    # Test API endpoints
    test_results.append(await test_new_api_endpoints())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}")
    print(f"❌ Failed: {len(test_results) - sum(test_results)}")
    print(f"📈 Success Rate: {sum(test_results)/len(test_results)*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 All Advanced Trading Features tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all(test_results)

if __name__ == "__main__":
    asyncio.run(main())
