# Opinion Market

A decentralized platform where people can trade and vote on their opinions, creating a dynamic marketplace of ideas and predictions.

## 🎯 Purpose

Opinion Market is a social trading platform that allows users to:
- **Trade opinions** - Buy and sell shares in various opinions and predictions
- **Vote on outcomes** - Participate in community voting on real-world events
- **Earn rewards** - Profit from accurate predictions and successful trades
- **Build reputation** - Establish credibility through consistent performance

## ✨ Features

### Core Functionality
- **Opinion Trading**: Users can create markets for any opinion or prediction
- **Voting System**: Community-driven voting on market outcomes
- **Portfolio Management**: Track your investments and performance
- **Market Discovery**: Browse trending opinions and popular markets
- **Real-time Updates**: Live price feeds and market activity

### Social Features
- **User Profiles**: Showcase your trading history and success rate
- **Following System**: Follow successful traders and opinion leaders
- **Discussion Forums**: Debate opinions and share insights
- **Leaderboards**: Compete with other traders for top rankings

### Advanced Features
- **Market Categories**: Organized by topics (politics, sports, entertainment, etc.)
- **Time-based Markets**: Short-term and long-term prediction markets
- **Liquidity Pools**: Ensure fair pricing and easy trading
- **Dispute Resolution**: Community governance for contested outcomes

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- PostgreSQL
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/opinion-market.git
   cd opinion-market
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Or install manually**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Start the development server**
   ```bash
   python run.py
   # or
   uvicorn app.main:app --reload
   ```

5. **Access the API**
   - API: `http://localhost:8000`
   - Documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

## 📁 Project Structure

```
opinion-market/
├── app/
│   ├── api/          # API routes and endpoints
│   ├── core/         # Core configuration and utilities
│   ├── models/       # Database models
│   └── schemas/      # Pydantic schemas
├── tests/            # Test files
├── alembic/          # Database migrations
├── requirements.txt  # Python dependencies
├── run.py           # Application entry point
└── setup.py         # Setup script
```

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI
- **Database**: PostgreSQL, SQLAlchemy
- **Authentication**: JWT, bcrypt
- **API Documentation**: OpenAPI/Swagger
- **Testing**: pytest
- **Deployment**: Docker, uvicorn

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.opinionmarket.com](https://docs.opinionmarket.com)
- **Discord**: [Join our community](https://discord.gg/opinionmarket)
- **Email**: support@opinionmarket.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/opinion-market/issues)

## 🗺️ Roadmap

### Phase 1: Core Platform ✅
- [x] Basic opinion trading functionality
- [x] User authentication and profiles
- [x] Market creation and voting
- [x] RESTful API with FastAPI
- [x] Database models and relationships
- [ ] Portfolio management
- [ ] Real-time price updates

### Phase 2: Social Features
- [ ] Following system
- [ ] Discussion forums
- [ ] Leaderboards
- [ ] Reputation system

### Phase 3: Advanced Features
- [ ] WebSocket real-time updates
- [ ] Mobile app
- [ ] API for third-party integrations
- [ ] Advanced analytics
- [ ] Governance tokens

## 🙏 Acknowledgments

- Inspired by prediction markets like PredictIt and Polymarket
- Built with modern web technologies
- Community-driven development

---

**Opinion Market** - Where ideas have value, and predictions become profits.
