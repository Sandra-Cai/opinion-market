# Opinion Market - Advanced Prediction Market Platform

A comprehensive prediction market platform built with FastAPI, React, and modern web technologies. This platform enables users to create, trade, and manage prediction markets with advanced features including real-time trading, AI analytics, and social features.

## 🚀 Features

### Core Features
- **Prediction Markets**: Create and trade on future events
- **Real-time Trading**: Live price updates and instant trade execution
- **Advanced Orders**: Market, limit, stop, and stop-limit orders
- **Portfolio Management**: Track performance and manage positions
- **Social Features**: Follow traders, share insights, and build reputation

### Advanced Features
- **AI Analytics**: Machine learning-powered market analysis
- **Risk Management**: Advanced risk assessment and portfolio optimization
- **Blockchain Integration**: Smart contract integration for decentralized features
- **Enterprise Security**: Multi-layer security with threat detection
- **Performance Optimization**: Caching, compression, and monitoring
- **Real-time Notifications**: WebSocket-based live updates

### Admin Features
- **User Management**: Comprehensive user administration
- **Market Moderation**: Content moderation and market management
- **System Monitoring**: Real-time system health and performance metrics
- **Analytics Dashboard**: Business intelligence and reporting
- **Security Audit**: Comprehensive security monitoring and logging

## 🏗️ Architecture

### Backend (FastAPI)
- **API Layer**: RESTful API with OpenAPI documentation
- **Service Layer**: Business logic separation
- **Database Layer**: PostgreSQL with SQLAlchemy ORM
- **Cache Layer**: Redis for high-performance caching
- **Security Layer**: JWT authentication, rate limiting, input validation
- **WebSocket Layer**: Real-time communication
- **ML Layer**: Machine learning models for predictions

### Frontend (React)
- **Component Library**: Reusable UI components with Tailwind CSS
- **State Management**: Zustand for global state
- **Routing**: React Router for navigation
- **Real-time Updates**: WebSocket integration
- **Charts & Analytics**: Recharts for data visualization
- **Responsive Design**: Mobile-first approach

### Infrastructure
- **Containerization**: Docker and Docker Compose
- **Reverse Proxy**: Nginx with SSL termination
- **Monitoring**: Prometheus and Grafana
- **Database**: PostgreSQL with connection pooling
- **Cache**: Redis for session and data caching
- **File Storage**: Local and cloud storage options

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: SQL toolkit and ORM
- **PostgreSQL**: Primary database
- **Redis**: Caching and message brokering
- **Pydantic**: Data validation and serialization
- **JWT**: Authentication and authorization
- **WebSockets**: Real-time communication
- **Alembic**: Database migrations
- **Pytest**: Testing framework

### Frontend
- **React 18**: Modern React with hooks
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **Recharts**: Chart library for React
- **React Router**: Client-side routing
- **Zustand**: State management
- **Framer Motion**: Animation library

### DevOps & Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **GitHub Actions**: CI/CD pipeline
- **AWS/GCP**: Cloud deployment options

## 📦 Installation

### Prerequisites
- Python 3.11+
- Node.js 16+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/opinion-market/opinion-market.git
   cd opinion-market
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

3. **Run database migrations**
   ```bash
   docker-compose run --rm migration
   ```

4. **Access the application**
   - API: http://localhost:8000
   - Frontend: http://localhost:3000
   - Admin Panel: http://localhost:3000/admin
   - API Docs: http://localhost:8000/docs

### Manual Installation

1. **Backend Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your configuration
   
   # Run database migrations
   alembic upgrade head
   
   # Start the server
   uvicorn app.main:app --reload
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/opinion_market

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# CORS
ALLOWED_HOSTS=["http://localhost:3000", "http://localhost:8000"]

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Caching
ENABLE_CACHING=true
CACHE_TTL=300

# WebSocket
WS_ENABLED=true
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=1000

# ML
ML_ENABLED=true
ML_MODEL_PATH=./models

# Blockchain
BLOCKCHAIN_ENABLED=false

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 📚 API Documentation

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get current user
- `POST /api/v1/auth/change-password` - Change password
- `POST /api/v1/auth/logout` - User logout

### Markets
- `GET /api/v1/markets/` - List markets
- `POST /api/v1/markets/` - Create market
- `GET /api/v1/markets/{id}` - Get market details
- `GET /api/v1/markets/trending` - Get trending markets
- `GET /api/v1/markets/stats` - Get market statistics

### Trading
- `POST /api/v1/trades/` - Create trade
- `GET /api/v1/trades/` - List user trades
- `GET /api/v1/trades/{id}` - Get trade details
- `POST /api/v1/orders/` - Create order
- `GET /api/v1/orders/` - List user orders
- `DELETE /api/v1/orders/{id}` - Cancel order

### WebSocket
- `ws://localhost:8000/api/v1/ws` - WebSocket connection
- Subscribe to market updates, trade notifications, and price feeds

### Admin
- `GET /api/v1/admin/stats` - Admin statistics
- `GET /api/v1/admin/users` - User management
- `GET /api/v1/admin/markets` - Market management
- `POST /api/v1/admin/users/{id}/moderate` - Moderate user
- `POST /api/v1/admin/markets/{id}/moderate` - Moderate market

## 🧪 Testing

### Backend Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api/test_auth.py

# Run with verbose output
pytest -v
```

### Frontend Tests
```bash
cd frontend

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

## 🚀 Deployment

### Production Deployment

1. **Using Docker Compose**
   ```bash
   # Build and start production services
   docker-compose -f docker-compose.prod.yml up -d
   
   # Run migrations
   docker-compose -f docker-compose.prod.yml run --rm migration
   ```

2. **Using Kubernetes**
   ```bash
   # Apply Kubernetes manifests
   kubectl apply -f k8s/
   
   # Check deployment status
   kubectl get pods
   ```

3. **Using Cloud Providers**
   - AWS: Use ECS, EKS, or Elastic Beanstalk
   - GCP: Use Cloud Run, GKE, or App Engine
   - Azure: Use Container Instances, AKS, or App Service

### Environment-Specific Configurations

- **Development**: `docker-compose.yml`
- **Production**: `docker-compose.prod.yml`
- **Testing**: `docker-compose.test.yml`

## 📊 Monitoring

### Health Checks
- `GET /health` - Application health
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **AlertManager**: Alert management
- **Jaeger**: Distributed tracing

### Key Metrics
- Request rate and latency
- Database connection pool
- Redis cache hit rate
- WebSocket connections
- Error rates and exceptions

## 🔒 Security

### Security Features
- JWT-based authentication
- Rate limiting and IP blocking
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection
- Security headers
- Audit logging

### Security Best Practices
- Regular security updates
- Dependency scanning
- Code analysis with Bandit
- Penetration testing
- Security monitoring
- Incident response plan

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint and Prettier for JavaScript/TypeScript
- Write tests for new features
- Update documentation
- Follow conventional commits

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.opinionmarket.com](https://docs.opinionmarket.com)
- **Issues**: [GitHub Issues](https://github.com/opinion-market/opinion-market/issues)
- **Discussions**: [GitHub Discussions](https://github.com/opinion-market/opinion-market/discussions)
- **Email**: support@opinionmarket.com

## 🗺️ Roadmap

### Phase 1 (Current)
- ✅ Core prediction market functionality
- ✅ Real-time trading
- ✅ User management
- ✅ Admin panel
- ✅ Basic analytics

### Phase 2 (Q2 2024)
- 🔄 Advanced order types
- 🔄 Mobile application
- 🔄 Social features
- 🔄 API marketplace
- 🔄 Advanced analytics

### Phase 3 (Q3 2024)
- 📋 Blockchain integration
- 📋 DeFi features
- 📋 Cross-chain support
- 📋 NFT integration
- 📋 Governance tokens

### Phase 4 (Q4 2024)
- 📋 AI-powered insights
- 📋 Automated trading
- 📋 Institutional features
- 📋 Global expansion
- 📋 Enterprise solutions

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- React team for the powerful frontend library
- PostgreSQL team for the robust database
- Redis team for the high-performance cache
- All contributors and community members

---

**Opinion Market** - Empowering prediction markets with cutting-edge technology.