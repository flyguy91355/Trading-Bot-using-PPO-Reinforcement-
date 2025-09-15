# Trading Bot with PPO Reinforcement Learning

## 📈 Overview

This project implements a **reinforcement learning-based trading bot** using **Proximal Policy Optimization (PPO)** from Stable Baselines3. It fetches historical stock or cryptocurrency data via yfinance (no API key required), simulates trading in a custom Gymnasium environment, and provides a user-friendly Tkinter GUI for training, testing, and visualization.

### 🚀 Key Enhancements

- **Technical Indicators**: RSI integration via the `ta` library
- **Risk Management**: Configurable stop-loss thresholds and trade costs (commissions)
- **Model Persistence**: Continue training from saved models
- **Hyperparameter Tuning**: Support via Optuna (optional)
- **Live Trading**: Stub for paper trading using Alpaca (optional, requires API keys)
- **Visualization**: Matplotlib charts with buy/sell signals and equity curves
- **Benchmarking**: Performance comparison against buy-and-hold strategies

### 🧠 Trading Model

The bot models trading as a **Markov Decision Process**:

- **State**: Normalized closing prices over a lookback window + optional RSI + portfolio metrics (balance, shares, net worth)
- **Actions**: Discrete (Hold, Buy, Sell)
- **Rewards**: Portfolio value change, penalized for stop-loss triggers and trade costs

> ⚠️ **Important**: This is for educational and backtesting purposes only. Trading involves significant risk; do not use for real investments without professional validation and compliance with regulations.

## ✨ Features

### 📊 **Data & Market Support**
- **Multi-Asset**: Supports stocks (e.g., AAPL) and crypto (e.g., BTC-USD) from Yahoo Finance
- **Auto-Symbol Fixing**: Automatically corrects symbols (e.g., BTC_USD → BTC-USD)

### 🤖 **AI Trading Environment**
- **Gymnasium-Compatible**: Custom environment with RSI integration
- **Realistic Trading**: Stop-loss protection and fractional shares support
- **Risk Management**: Configurable trade costs and portfolio limits

### 🎯 **Training & Testing**
- **PPO Agent**: Advanced reinforcement learning with TensorBoard logging
- **Model Persistence**: Resume training from saved models
- **Comprehensive Testing**: Full dataset evaluation with detailed metrics

### 🖥️ **User Interface**
- **Tkinter GUI**: Easy parameter tuning and control
- **Real-time Logging**: Live training progress and results
- **Non-blocking Operations**: Multi-threaded for smooth experience
- **Interactive Plotting**: Matplotlib charts with trading signals

### 🔧 **Optional Modules**
- **RSI Indicator**: Requires `ta` library
- **Hyperparameter Optimization**: Requires `optuna`
- **Live Trading**: Requires `alpaca-py` and API keys

### 📈 **Performance Analytics**
- Return percentage and trade statistics
- Action distribution analysis
- Outperformance vs. buy-and-hold comparison
- Portfolio equity curve visualization

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/flyguy91355/Trading-Bot-using-PPO-Reinforcement-.git
cd Trading-Bot-using-PPO-Reinforcement-
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `yfinance`, `pandas`, `numpy` - Data handling
- `gymnasium`, `stable-baselines3[extra]`, `torch` - Reinforcement learning
- `matplotlib` - Visualization

**Optional Dependencies:**
- `ta` - Technical analysis (RSI)
- `optuna` - Hyperparameter tuning
- `alpaca-py`, `websockets` - Live trading

> **Note**: Requires Python 3.9+. Tkinter is included in Python standard library (Linux users: `sudo apt-get install python3-tk`)

### 4. Live Trading Setup (Optional)
1. Sign up for [Alpaca](https://alpaca.markets) (paper trading recommended)
2. Set environment variables:
```bash
export APCA_API_KEY_ID=your_key
export APCA_API_SECRET_KEY=your_secret
```

## 📖 Usage

### Starting the Application
```bash
python TradingBotPPO.py
```

### GUI Workflow

#### 1. **Parameter Configuration**
- **Symbol**: Enter stock (AAPL) or crypto (BTC-USD)
- **Date Range**: Set training period (e.g., 2022-01-01 to 2023-12-31)
- **Initial Balance**: Starting portfolio value ($10,000 default)
- **Timesteps**: Training duration (100,000 default)
- **RSI Toggle**: Enable technical analysis
- **Trade Cost**: Commission percentage (0.001 default)
- **Continue from Saved**: Resume previous training

#### 2. **Trading Workflow**
1. **Fetch Data**: Downloads OHLCV data and creates environment
2. **Train Model**: Trains/resumes PPO agent (progress shown in log)
3. **Test Model**: Runs simulation with detailed metrics
4. **Plot Chart**: Displays interactive Matplotlib charts
5. **Save/Load Model**: Persist models as ZIP files

#### 3. **Performance Analysis**
```
Final Portfolio Value: $11,234.56
Return: 12.35%
Total Trades: 45
Buy & Hold Return: 8.21%
Strategy Outperformance: 4.14%
Actions: {'hold': 200, 'buy': 25, 'sell': 20}
```

### TensorBoard Monitoring
```bash
tensorboard --logdir ./trading_tensorboard/
```
View training metrics at http://localhost:6006

## 📁 Project Structure
```
Trading-Bot-using-PPO-Reinforcement-/
├── TradingBotPPO.py       # Main application with GUI
├── requirements.txt       # Python dependencies
├── requirement.txt        # Alternative requirements file
├── README.md             # Documentation
├── .gitignore            # Git ignore rules
├── test_trading_bot.py   # Automated testing
├── test_gui.py           # GUI testing
├── trading_model_*.zip   # Saved models (auto-generated)
└── trading_tensorboard/  # TensorBoard logs
```

## 🔧 Customization

### Environment Modifications
Edit `TradingBotPPO.py` to add features:
- Additional technical indicators in `TradingEnv`
- Custom reward functions
- Enhanced risk management

### Hyperparameter Tuning
Integrate Optuna in `train_ppo_agent()` for automated optimization

### Live Trading
Uncomment Alpaca stubs for paper/live trading (use paper mode first!)

## ⚠️ Limitations & Risks

### Backtesting Considerations
- **Historical Bias**: Past performance ≠ future results
- **Simplified Model**: No slippage, taxes, or market impact
- **Data Limitations**: yfinance provides delayed data

### Technical Limitations
- **Compute Requirements**: GPU recommended for large timesteps
- **Real-time Data**: Use Alpaca API for live trading
- **Trading Assumptions**: Fractional shares, no shorting

### Legal & Financial
- **Not Financial Advice**: Educational purposes only
- **Regulatory Compliance**: Follow SEC rules and Investment Advisers Act of 1940
- **Risk Warning**: Trading involves substantial risk of loss

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/enhance-rsi`
3. **Commit** changes: `git commit -m "Add RSI window config"`
4. **Push** to branch: `git push origin feature/enhance-rsi`
5. **Open** Pull Request

Issues and PRs welcome! Built with open-source principles.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **[Stable Baselines3](https://stable-baselines3.readthedocs.io/)** - PPO implementation
- **[Gymnasium](https://gymnasium.farama.org/)** - RL environment framework
- **[yfinance](https://pypi.org/project/yfinance/)** - Market data access
- **[Alpaca](https://alpaca.markets/)** - Brokerage API
- **[TA-Lib](https://pypi.org/project/ta/)** - Technical analysis indicators

---

**🇺🇸 Algorithmic trading for the people—built with American ingenuity!**

**Questions?** File an issue on GitHub!
