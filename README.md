Trading Bot using PPO Reinforcement Learning
Project Description
This project is an advanced algorithmic trading bot that utilizes a Proximal Policy Optimization (PPO) reinforcement learning agent to make trading decisions. The bot operates within a custom trading environment built with Gym, and it is capable of fetching historical data for stocks and cryptocurrencies. It includes an intuitive GUI for user interaction, allowing for easy data fetching, model training, and performance testing.

Key Features
Data Fetching: Automatically downloads historical price data from Yahoo Finance for any stock or crypto symbol.

Reinforcement Learning: Trains a PPO agent using Stable Baselines3 to learn an optimal trading strategy.

Custom Trading Environment: A Gym-compatible environment that simulates real-world trading, including an initial balance, share holdings, and portfolio value.

Technical Indicators: Optionally uses the Relative Strength Index (RSI) as a key feature for the agent's observation space.

Graphical User Interface (GUI): A user-friendly interface built with Tkinter for seamless control over the bot's functions.

Performance Evaluation: Provides a comprehensive report on the model's performance, including final portfolio value, return percentage, and a comparison against a simple buy-and-hold strategy.

Installation
To set up and run this project, you need to have Python installed. It is highly recommended to use a virtual environment to manage project dependencies.

Clone the repository:


cd your-repo

Create a virtual environment (optional but recommended):

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install the required libraries:
The project uses a requirements.txt file to list all dependencies. Use the following command to install them:

pip install -r requirements.txt

Usage
To run the trading bot and launch the GUI, execute the main Python script from your terminal:

python trading_bot.py

GUI Controls
Symbol, Dates, and Balance: Enter the stock or crypto symbol (e.g., AAPL, BTC-USD), the start and end dates, and your initial trading balance.

Fetch Data: Downloads the historical data and sets up the trading environment.

Train Model: Initiates the PPO training process. This may take a while depending on the number of timesteps.

Test Model: Evaluates the performance of the trained model on the historical data.

Save/Load Model: Allows you to save a trained model to a .zip file or load a previously saved model to avoid re-training.

Dependencies
This project relies on the following Python libraries:

pandas==2.2.2

numpy==1.26.4

yfinance==0.2.38

gymnasium==0.29.1

stable-baselines3==2.3.0

torch==2.3.0

ta==0.11.0 (Optional, for technical indicators)

optuna==3.6.1 (Optional, for hyperparameter optimization)

alpaca-py==0.11.0 (Optional, for live trading functionality)

websockets==12.0 (Required for alpaca-py)

License
