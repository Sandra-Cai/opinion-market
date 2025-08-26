# Opinion Market

A sophisticated prediction market platform where people can trade and vote on their opinions, built with advanced features similar to Polymarket.

## 🎯 Purpose

Opinion Market is a decentralized prediction market platform that allows users to:
- **Trade on predictions** about real-world events
- **Vote on market outcomes** with confidence levels
- **Earn rewards** for accurate predictions
- **Participate in governance** through dispute resolution
- **Access real-time analytics** and market insights

## ✨ Features

### Core Features
- **Prediction Markets**: Create and trade on binary and multiple-choice markets
- **Advanced Trading**: Market orders, limit orders, and stop orders
- **Portfolio Management**: Track positions, P&L, and performance metrics
- **Real-time Updates**: WebSocket-based live market data and trade feeds
- **AMM Pricing**: Automated Market Maker for dynamic price discovery

### Social Features
- **User Profiles**: Comprehensive trading statistics and reputation scores
- **Leaderboards**: Rank users by profit, volume, win rate, and reputation
- **Market Discovery**: Trending markets, categories, and search functionality
- **Community Voting**: Vote on market outcomes with confidence levels

### Advanced Features
- **Dispute Resolution**: Community-driven dispute system for contested market resolutions
- **Market Verification**: Moderator/admin system for market quality control
- **Notification System**: Email and push notifications for market events
- **Analytics Dashboard**: Comprehensive market and user analytics
- **Order Book**: Advanced order matching and depth visualization
- **Price Predictions**: AI-powered market price forecasting
- **Sentiment Analysis**: Trading sentiment and market psychology insights
- **Mobile API**: Mobile-optimized endpoints and push notifications
- **Advanced Orders**: Stop-loss, take-profit, trailing stops, and conditional orders
- **Rewards System**: Gamification with achievements and token rewards
- **Real-time Analytics**: Live market data and performance tracking

### Polymarket-like Features
- **Liquidity Pools**: Separate pools for each market outcome
- **Fee Structure**: Trading fees and platform revenue sharing
- **Market Quality Scoring**: Algorithmic market quality assessment
- **Trending Algorithm**: Real-time market trending detection
- **Advanced Order Types**: Limit, market, and stop orders
- **Real-time Price Feeds**: Live market data and trade broadcasts

## 🎯 Advanced Features

### AI-Powered Analytics
- **Market Predictions**: ML-based price forecasting and trend analysis
- **User Insights**: Personalized trading recommendations and performance analysis
- **Sentiment Analysis**: Real-time market sentiment tracking
- **Risk Assessment**: AI-powered risk analysis for trades and portfolios
- **Performance Analytics**: Advanced portfolio performance metrics
- **Price Forecasting**: Short, medium, and long-term price predictions
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
- **Market Correlation Analysis**: Cross-market relationship insights

### Rewards & Gamification
- **Daily Login Rewards**: Earn tokens and XP for daily engagement
- **Trade Streaks**: Bonus rewards for consecutive trading days
- **Achievement System**: Unlock achievements for various milestones
- **Volume Milestones**: Rewards for reaching trading volume targets
- **Profit Milestones**: Bonuses for profitable trading performance
- **Market Creation Rewards**: Incentives for creating quality markets
- **Governance Participation**: Rewards for active governance involvement
- **Winning Streaks**: Special rewards for consecutive profitable trades
- **Leaderboards**: Competitive rankings across multiple categories

### Mobile API Support
- **Mobile-Optimized Endpoints**: Streamlined API responses for mobile apps
- **Push Notifications**: Real-time alerts for trades, market updates, and rewards
- **Device Registration**: Secure mobile device management
- **Mobile Dashboard**: Optimized data for mobile interfaces
- **Portfolio Tracking**: Mobile-friendly portfolio management
- **Performance Charts**: Mobile-optimized chart data
- **Search & Discovery**: Fast market search for mobile users

### Advanced Order Types
- **Stop-Loss Orders**: Automatic selling when price falls below threshold
- **Take-Profit Orders**: Secure gains when price reaches target
- **Trailing Stops**: Dynamic stop-loss that follows price movements
- **Conditional Orders**: Orders that trigger based on other market conditions
- **Bracket Orders**: Entry orders with automatic stop-loss and take-profit
- **Risk Management**: Built-in risk assessment and warnings
- **Order Statistics**: Performance tracking for advanced orders

### Real-Time Market Data Feeds
- **Live Price Updates**: Real-time market price and volume data
- **Market Alerts**: Automated alerts for price spikes, volume surges, and liquidity drops
- **Market Statistics**: Detailed analytics for any time period
- **Trending Markets**: Algorithm-based market trending detection
- **Volatility Tracking**: Real-time volatility calculations and alerts
- **WebSocket Feeds**: Live data streams for real-time applications

### Comprehensive Monitoring & Alerting
- **System Performance Monitoring**: CPU, memory, disk, and network monitoring
- **Application Metrics**: User activity, trading volume, and performance tracking
- **Automated Alerting**: Intelligent alerts for system and application issues
- **Health Checks**: Database, Redis, and external service monitoring
- **Performance Analysis**: Trend analysis and predictive insights
- **Production-Ready Infrastructure**: Docker, monitoring stack, and backup systems

### Advanced Machine Learning & AI
- **Market Price Predictions**: ML-powered price forecasting with confidence scores
- **User Behavior Analysis**: Comprehensive trading profile and risk assessment
- **Personalized Recommendations**: AI-driven trading suggestions based on user patterns
- **Model Performance Tracking**: Continuous model evaluation and retraining
- **Bulk Market Analysis**: Multi-market prediction and trend analysis
- **Intelligent Insights**: Automated market and user insights generation

### Blockchain Integration & DeFi Features
- **Smart Contract Integration**: Market creation and trading on blockchain
- **Multi-Chain Support**: Ethereum, Polygon, and Arbitrum networks
- **Governance on Blockchain**: Decentralized proposal creation and voting
- **Token Balance Tracking**: Real-time cryptocurrency and token balances
- **Transaction Monitoring**: Automated blockchain transaction tracking
- **DeFi Rewards**: Blockchain-based reward distribution system

### Enterprise Security & Compliance
- **Advanced Threat Detection**: Real-time threat detection and prevention
- **Encryption**: End-to-end encryption for sensitive data
- **Audit Logging**: Comprehensive security audit trails
- **Risk Assessment**: Dynamic user risk profiling and scoring
- **Rate Limiting**: Intelligent rate limiting and DDoS protection
- **Compliance**: GDPR, SOC2, and financial compliance features
- **Security Monitoring**: 24/7 security event monitoring and alerting

### Performance Optimization
- **Intelligent Caching**: Multi-layer caching with Redis
- **Database Optimization**: Connection pooling and query optimization
- **Memory Management**: Automatic memory optimization and garbage collection
- **Performance Monitoring**: Real-time performance metrics and profiling
- **Load Balancing**: Intelligent request distribution and load management
- **Resource Optimization**: CPU, memory, and network optimization

### Social Features & Community
- **User Profiles**: Rich user profiles with trading statistics
- **Social Posts**: Share trading insights and market analysis
- **Communities**: Create and join trading communities
- **Follow System**: Follow other traders and get personalized feeds
- **Social Analytics**: Influence scores, engagement metrics, and reach analysis
- **Trending Topics**: Real-time trending topics and discussions
- **Content Discovery**: Advanced content search and recommendation

### Comprehensive API Documentation
- **OpenAPI/Swagger**: Interactive API documentation
- **Code Examples**: Multiple programming language examples
- **SDK Libraries**: Official SDKs for Python, JavaScript, and Java
- **Webhook Support**: Real-time webhook notifications
- **Rate Limiting**: Transparent rate limiting documentation
- **Error Handling**: Comprehensive error codes and troubleshooting
- **Integration Guides**: Step-by-step integration tutorials

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 13+
- Redis 6+
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd opinion-market
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Start the application**
   ```bash
   python run.py
   ```

### Environment Setup

1. **Copy environment file**
   ```bash
   cp env.example .env
   ```

2. **Configure environment variables**
   ```bash
   # Database
   DATABASE_URL=postgresql://user:password@localhost/opinion_market
   
   # Security
   SECRET_KEY=your-secret-key-here
   
   # Redis
   REDIS_URL=redis://localhost:6379
   
   # Email (optional)
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ```

### Running with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## 📁 Project Structure

```
opinion-market/
├── app/
│   ├── api/v1/endpoints/     # API endpoints
│   │   ├── auth.py          # Authentication
│   │   ├── users.py         # User management
│   │   ├── markets.py       # Market operations
│   │   ├── trades.py        # Trading operations
│   │   ├── positions.py     # Portfolio management
│   │   ├── orders.py        # Order book system
│   │   ├── disputes.py      # Dispute resolution
│   │   ├── notifications.py # Notification system
│   │   ├── analytics.py     # Analytics endpoints
│   │   ├── verification.py  # Market verification
│   │   ├── leaderboard.py   # Leaderboards
│   │   └── websocket.py     # Real-time updates
│   ├── core/                # Core functionality
│   │   ├── config.py        # Configuration
│   │   ├── database.py      # Database setup
│   │   └── auth.py          # Authentication utilities
│   ├── models/              # Database models
│   │   ├── user.py          # User model
│   │   ├── market.py        # Market model
│   │   ├── trade.py         # Trade model
│   │   ├── position.py      # Position model
│   │   ├── order.py         # Order book models
│   │   ├── dispute.py       # Dispute models
│   │   └── notification.py  # Notification models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   │   ├── price_feed.py    # Real-time price updates
│   │   ├── notification_service.py # Notification handling
│   │   └── analytics_service.py # Analytics processing
│   └── main.py              # FastAPI application
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker services
├── Dockerfile              # Application container
├── setup.py                # Setup script
└── run.py                  # Application runner
```

## 🛠 Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM
- **PostgreSQL**: Primary database
- **Redis**: Caching and message brokering
- **Alembic**: Database migrations
- **Pydantic**: Data validation and serialization

### Authentication & Security
- **JWT**: JSON Web Tokens for authentication
- **Bcrypt**: Password hashing
- **OAuth2**: Token-based authentication

### Real-time Features
- **WebSockets**: Real-time communication
- **Redis Pub/Sub**: Message broadcasting
- **Background Tasks**: Async task processing

### Development & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **Uvicorn**: ASGI server
- **Pytest**: Testing framework

## 📊 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/users/me` - Get current user

### Markets
- `GET /api/v1/markets/` - List markets
- `POST /api/v1/markets/` - Create market
- `GET /api/v1/markets/{id}` - Get market details
- `GET /api/v1/markets/trending` - Get trending markets
- `GET /api/v1/markets/stats` - Get market statistics

### Trading
- `POST /api/v1/trades/` - Create trade
- `GET /api/v1/trades/` - Get user trades
- `POST /api/v1/orders/` - Create order
- `GET /api/v1/orders/` - Get user orders
- `GET /api/v1/orders/market/{id}/orderbook` - Get order book

### Portfolio
- `GET /api/v1/positions/` - Get user positions
- `GET /api/v1/positions/portfolio` - Get portfolio summary
- `POST /api/v1/positions/{id}/close` - Close position

### Analytics
- `GET /api/v1/analytics/market/{id}` - Market analytics
- `GET /api/v1/analytics/user/me` - User analytics
- `GET /api/v1/analytics/platform` - Platform analytics
- `GET /api/v1/analytics/market/{id}/predictions` - Market predictions
- `GET /api/v1/analytics/sentiment-analysis` - Sentiment analysis

### Disputes
- `POST /api/v1/disputes/` - Create dispute
- `GET /api/v1/disputes/` - List disputes
- `POST /api/v1/disputes/{id}/vote` - Vote on dispute

### Notifications
- `GET /api/v1/notifications/` - Get notifications
- `PUT /api/v1/notifications/preferences` - Update preferences
- `POST /api/v1/notifications/{id}/read` - Mark as read

### Verification
- `GET /api/v1/verification/pending` - Pending verifications
- `POST /api/v1/verification/{id}/verify` - Verify market
- `POST /api/v1/verification/{id}/reject` - Reject market

### Leaderboards
- `GET /api/v1/leaderboard/traders` - Top traders
- `GET /api/v1/leaderboard/volume` - Top volume traders
- `GET /api/v1/leaderboard/creators` - Top market creators

### WebSocket
- `ws://localhost:8000/ws/market/{id}` - Market updates
- `ws://localhost:8000/ws/user` - User updates
- `ws://localhost:8000/ws/global` - Global updates

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_api.py

# Run specific test modules
pytest tests/test_auth.py
pytest tests/test_markets.py
pytest tests/test_trades.py
```

## 📈 Advanced Features

### Market Verification System
- **Quality Control**: Markets require verification before trading
- **Moderator Tools**: Admin interface for market management
- **Dispute Resolution**: Community voting on contested resolutions
- **Quality Scoring**: Algorithmic assessment of market quality

### Notification System
- **Real-time Alerts**: Market resolutions, price changes, trade executions
- **Customizable Preferences**: Email and push notification settings
- **Price Alerts**: Configurable thresholds for price movements
- **System Announcements**: Platform-wide notifications

### Analytics & Insights
- **Market Analytics**: Volume, sentiment, and price movement analysis
- **User Analytics**: Trading performance and portfolio metrics
- **Platform Analytics**: Overall platform statistics and trends
- **Predictions**: AI-powered price forecasting with confidence scores
- **Sentiment Analysis**: Trading sentiment and market psychology

### Order Book System
- **Limit Orders**: Set specific prices for trades
- **Market Orders**: Immediate execution at current prices
- **Stop Orders**: Trigger-based order execution
- **Order Matching**: Advanced order matching engine
- **Depth Visualization**: Real-time order book depth

### Real-time Features
- **Live Price Feeds**: Real-time market price updates
- **Trade Broadcasting**: Live trade execution notifications
- **WebSocket API**: Low-latency real-time communication
- **Market Updates**: Live market statistics and changes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the API documentation at `/docs`
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions for questions and ideas

## 🗺 Roadmap

### Completed ✅
- Core prediction market functionality
- User authentication and profiles
- Market creation and trading
- Portfolio management
- Real-time WebSocket updates
- Leaderboards and rankings
- Advanced analytics system
- Dispute resolution system
- Notification system
- Market verification system
- Order book and advanced trading
- Price prediction algorithms

### In Progress 🚧
- Mobile app development
- Advanced charting and visualization
- Social features and following
- API rate limiting and optimization
- Advanced security features

### Planned 📋
- DeFi integration (liquidity mining, yield farming)
- Cross-chain compatibility
- Advanced market types (futures, options)
- Institutional trading features
- Advanced AI/ML predictions
- Governance token system
- DAO governance structure

## 🙏 Acknowledgments

- Inspired by Polymarket and other prediction market platforms
- Built with modern Python web technologies
- Community-driven development approach
- Open source contributors and supporters
