# Trading Bot using PPO Reinforcement Learning with GUI
# This script fetches historical stock/crypto data using yfinance (best free method, no API key required),
# creates a custom Gym environment for trading, and trains a PPO agent using Stable Baselines3.
# Enhanced with: Predict on new data, RSI technical indicator (optional, requires pip install ta),
# stop-loss risk management, hyperparameter tuning via Optuna (requires pip install optuna),
# and stub for live paper trading via alpaca-py (requires pip install alpaca-py; API keys needed).
# GUI using Tkinter (stdlib, no extra install) for user interaction: change symbols (stocks/crypto like 'BTC-USD'),
# dates, settings, train/test/predict.
# Dependencies: pip install yfinance pandas numpy gymnasium stable-baselines3[extra] torch ta optuna alpaca-py websockets
import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch as th
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
import threading
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
# Conditional imports (to avoid errors if not installed)
try:
    from ta.momentum import RSIIndicator
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("Warning: 'ta' not installed; RSI feature disabled. Run: pip install ta")
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: 'optuna' not installed; hyperparam tuning disabled. Run: pip install optuna")
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    print("Warning: 'alpaca-py' not installed; live stub disabled. Run: pip install alpaca-py")
# Step 1: Fetch historical stock/crypto data
def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical data from Yahoo Finance using yfinance (supports stocks/crypto like 'BTC-USD').
   
    Args:
        symbol (str): Symbol, e.g., 'AAPL' or 'BTC-USD'
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
   
    Returns:
        pd.DataFrame: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
        data = data.dropna()
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
# Step 2: Custom Trading Environment (Enhanced with RSI and Stop-Loss)
class TradingEnv(gym.Env):
    """
    Custom Gym environment for stock/crypto trading.
    - State: Normalized prices + optional RSI (lookback window).
    - Actions: Discrete (0: Hold, 1: Buy, 2: Sell).
    - Reward: Portfolio change, with stop-loss trigger.
    - Episode ends when data processed or stop-loss hit.
    """
   
    def __init__(self, df, initial_balance=10000, lookback_window=30, use_rsi=False, stop_loss_pct=0.10, trade_cost=0.001):
        super(TradingEnv, self).__init__()
       
        self.df = df.copy()
        self.initial_balance = float(initial_balance)
        self.lookback_window = int(lookback_window)
        self.use_rsi = use_rsi and HAS_TA
        self.stop_loss_pct = float(stop_loss_pct)
        self.trade_cost = float(trade_cost)
       
        # Add RSI if requested and available
        if self.use_rsi:
            try:
                rsi_indicator = RSIIndicator(close=self.df['Close'].squeeze(), window=self.lookback_window)
                self.df['RSI'] = rsi_indicator.rsi()
                self.df['RSI'] = self.df['RSI'].fillna(50) # Fill NaN with neutral RSI
            except Exception as e:
                print(f"Error calculating RSI: {e}, proceeding without it.")
                self.use_rsi = False
       
        # Normalize price data
        self.df['Normalized_Close'] = (self.df['Close'] - self.df['Close'].mean()) / self.df['Close'].std()
       
        # Environment variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = 0
        self.terminated = False
       
        # Define action and observation spaces
        obs_features = 2 if self.use_rsi else 1
        obs_size = self.lookback_window * obs_features + 3
       
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )
       
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
       
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = 0
        self.terminated = False
       
        return self._get_observation(), {}
   
    def _get_observation(self):
        """
        Get current observation state.
        """
        if self.current_step < self.lookback_window or self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape[0]).astype(np.float32)
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
       
        # Fill in price data
        price_window = self.df['Normalized_Close'].iloc[start_idx:end_idx].values.flatten()
        obs[:self.lookback_window] = price_window
       
        current_idx = self.lookback_window
       
        # Fill in RSI data if used
        if self.use_rsi:
            rsi_window = self.df['RSI'].iloc[start_idx:end_idx].values.flatten()
            obs[current_idx:current_idx + self.lookback_window] = rsi_window / 100.0
            current_idx += self.lookback_window
           
        # Fill in portfolio state, explicitly converting single-value Series to float
        obs[current_idx] = float(self.balance) / float(self.initial_balance)
        obs[current_idx + 1] = float(self.shares_held) * float(self.df['Close'].iloc[self.current_step - 1]) / float(self.initial_balance)
        obs[current_idx + 2] = float(self.net_worth) / float(self.initial_balance)
        return obs
   
    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
       
        # Get current price
        current_price = float(self.df['Close'].iloc[self.current_step - 1])
       
        reward = 0
        prev_net_worth = self.net_worth
        trade_happened = False
       
        old_shares = self.shares_held
       
        if action == 1: # Buy
            if self.balance > current_price * 0.01:  # Allow buying at least 0.01 shares
                shares_to_buy = self.balance / current_price
                self.shares_held += shares_to_buy
                self.balance = 0.0
                self.trades += 1
                trade_happened = True
                   
        elif action == 2: # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0.0
                self.trades += 1
                trade_happened = True
       
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
       
        if prev_net_worth > 0:
            reward = (self.net_worth - prev_net_worth) / prev_net_worth
       
        if trade_happened:
            reward -= self.trade_cost
        
        if self.net_worth < self.initial_balance * (1 - self.stop_loss_pct):
            self.terminated = True
            reward -= 0.1
       
        # Check if episode is terminated (end of data)
        if self.current_step >= len(self.df):
            self.terminated = True
            final_reward = (self.net_worth - self.initial_balance) / self.initial_balance
            reward += final_reward * 0.1
           
        return self._get_observation(), reward, self.terminated, False, {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'trades': self.trades
        }
# Step 3: Training Functions
def train_ppo_agent(env, timesteps=50000, continue_from_saved=False, model_path=None, log_callback=None):
    """Train PPO agent on the trading environment, optionally continuing from saved model."""
    try:
        env_vec = DummyVecEnv([lambda: env])
        if continue_from_saved and model_path and os.path.exists(model_path):
            model = PPO.load(model_path, env=env_vec)
            if log_callback:
                log_callback("Loaded saved model for continued training...")
        else:
            model = PPO(
                "MlpPolicy",
                env_vec,
                verbose=1,
                tensorboard_log="./trading_tensorboard/",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                device='auto'
            )
            if log_callback:
                log_callback("Created new PPO model...")
        if log_callback:
            log_callback("Starting PPO training...")
        model.learn(total_timesteps=timesteps, progress_bar=True)
        if log_callback:
            log_callback(f"Training completed with {timesteps} timesteps")
        return model
   
    except Exception as e:
        if log_callback:
            log_callback(f"Training error: {e}")
        print(f"Training error: {e}")
        return None
def test_agent(model, env, log_callback=None):
    """Test the trained agent and return results."""
    try:
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        actions_taken = {'hold': 0, 'buy': 0, 'sell': 0}
       
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
           
            action_names = ['hold', 'buy', 'sell']
            action_int = int(action)
            if action_int < len(action_names):
                actions_taken[action_names[action_int]] += 1
           
            total_reward += reward
            steps += 1
           
            if terminated or truncated:
                # FIX: When the loop breaks, we get the final values directly from the environment
                # object, which has the correct state after the last step.
                final_balance = env.balance
                shares_held = env.shares_held
                trades = env.trades
                net_worth = env.net_worth
                break
       
        final_price = float(env.df['Close'].iloc[-1])
        final_value = final_balance + shares_held * final_price
       
        results = {
            'total_reward': total_reward,
            'final_value': final_value,
            'initial_balance': env.initial_balance,
            'return_pct': ((final_value - env.initial_balance) / env.initial_balance) * 100,
            'trades': trades,
            'steps': steps,
            'actions_taken': actions_taken
        }
       
        if log_callback:
            log_callback(f"Test Results:")
            log_callback(f"Final Portfolio Value: ${final_value:.2f}")
            log_callback(f"Return: {results['return_pct']:.2f}%")
            log_callback(f"Total Trades: {trades}")
            log_callback(f"Actions: {actions_taken}")
       
        return results
   
    except Exception as e:
        if log_callback:
            log_callback(f"Testing error: {e}")
        print(f"Testing error: {e}")
        return None
# Step 4: GUI Application
class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot with PPO RL")
        self.root.geometry("900x700")  # Slightly taller for new checkbox
       
        self.model = None
        self.env = None
        self.data = pd.DataFrame()
       
        self.setup_gui()
   
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
       
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
       
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
       
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        params_frame.columnconfigure(5, weight=1)
        params_frame.columnconfigure(7, weight=1)
        ttk.Label(params_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.symbol_var = tk.StringVar(value="AAPL")
        ttk.Entry(params_frame, textvariable=self.symbol_var, width=15).grid(row=0, column=1, sticky=tk.W)
       
        ttk.Label(params_frame, text="Start Date:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.start_date_var = tk.StringVar(value="2022-01-01")
        ttk.Entry(params_frame, textvariable=self.start_date_var, width=15).grid(row=0, column=3, sticky=tk.W)
       
        ttk.Label(params_frame, text="End Date:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.end_date_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(params_frame, textvariable=self.end_date_var, width=15).grid(row=0, column=5, sticky=tk.W)
       
        ttk.Label(params_frame, text="Initial Balance:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.balance_var = tk.StringVar(value="10000")
        ttk.Entry(params_frame, textvariable=self.balance_var, width=15).grid(row=1, column=1, sticky=tk.W)
       
        ttk.Label(params_frame, text="Timesteps:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.timesteps_var = tk.StringVar(value="100000")
        ttk.Entry(params_frame, textvariable=self.timesteps_var, width=15).grid(row=1, column=3, sticky=tk.W)
       
        self.use_rsi_var = tk.BooleanVar(value=HAS_TA)
        ttk.Checkbutton(params_frame, text="Use RSI", variable=self.use_rsi_var).grid(row=1, column=4, sticky=tk.W, padx=5)
       
        ttk.Label(params_frame, text="Trade Cost:").grid(row=1, column=5, sticky=tk.W, padx=5)
        self.trade_cost_var = tk.StringVar(value="0.001")
        ttk.Entry(params_frame, textvariable=self.trade_cost_var, width=15).grid(row=1, column=6, sticky=tk.W)
       
        self.continue_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Continue from Saved", variable=self.continue_var).grid(row=1, column=7, sticky=tk.W, padx=5)
       
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=0, pady=(0, 10))
       
        ttk.Button(buttons_frame, text="Fetch Data", command=self.fetch_data_thread).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="Train Model", command=self.train_model_thread).grid(row=0, column=1, padx=5)
        ttk.Button(buttons_frame, text="Test Model", command=self.test_model_thread).grid(row=0, column=2, padx=5)
        ttk.Button(buttons_frame, text="Save Model", command=self.save_model).grid(row=0, column=3, padx=5)
        ttk.Button(buttons_frame, text="Load Model", command=self.load_model).grid(row=0, column=4, padx=5)
        ttk.Button(buttons_frame, text="Clear Log", command=self.clear_log).grid(row=0, column=5, padx=5)
        ttk.Button(buttons_frame, text="Plot Chart", command=self.plot_chart).grid(row=0, column=6, padx=5)
       
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="5")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
       
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
       
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
   
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
   
    def save_model(self):
        if self.model is None:
            self.log_message("Error: No model to save. Please train a model first.")
            return
       
        try:
            model_path = f"trading_model_{self.symbol_var.get()}.zip"
            self.model.save(model_path)
            self.log_message(f"Model saved successfully to {model_path}")
        except Exception as e:
            self.log_message(f"Error saving model: {e}")
   
    def load_model(self):
        try:
            model_path = f"trading_model_{self.symbol_var.get()}.zip"
            if not os.path.exists(model_path):
                self.log_message(f"Error: Model file {model_path} not found")
                return
           
            if self.env is None:
                self.log_message("Please fetch data first to create environment")
                return
           
            self.model = PPO.load(model_path)
            self.log_message(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.log_message(f"Error loading model: {e}")
   
    def fetch_data_thread(self):
        threading.Thread(target=self.fetch_data, daemon=True).start()
   
    def fetch_data(self):
        try:
            symbol = self.symbol_var.get().strip().upper()
            start_date = self.start_date_var.get().strip()
            end_date = self.end_date_var.get().strip()
           
            if not symbol or not start_date or not end_date:
                self.log_message("Error: Please fill in all required fields")
                return
           
            # Fix for crypto symbols: replace '_' with '-' (e.g., BTC_USD -> BTC-USD)
            symbol = symbol.replace('_', '-')
           
            self.log_message(f"Fetching data for {symbol} from {start_date} to {end_date}...")
           
            self.data = fetch_stock_data(symbol, start_date, end_date)
           
            if self.data.empty:
                self.log_message(f"Error: No data found for symbol {symbol}")
                return
           
            self.log_message(f"Successfully fetched {len(self.data)} data points")
            self.log_message(f"Data range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            self.log_message(f"Price range: ${float(self.data['Close'].min()):.2f} - ${float(self.data['Close'].max()):.2f}")
           
            initial_balance = float(self.balance_var.get())
            use_rsi = self.use_rsi_var.get()
            trade_cost = float(self.trade_cost_var.get())
           
            self.env = TradingEnv(
                self.data,
                initial_balance=initial_balance,
                use_rsi=use_rsi,
                trade_cost=trade_cost
            )
           
            self.log_message("Trading environment created successfully")
           
        except Exception as e:
            self.log_message(f"Error fetching data: {e}")
   
    def train_model_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()
   
    def train_model(self):
        try:
            if self.env is None:
                self.log_message("Error: Please fetch data first")
                return
           
            timesteps = int(self.timesteps_var.get())
            continue_training = self.continue_var.get()
            model_path = f"trading_model_{self.symbol_var.get()}.zip"
           
            self.log_message("Starting model training...")
            self.model = train_ppo_agent(
                self.env, 
                timesteps=timesteps, 
                continue_from_saved=continue_training, 
                model_path=model_path,
                log_callback=self.log_message
            )
           
            if self.model:
                self.log_message("Model training completed successfully!")
                self.log_message("You can now test the model or save it for later use.")
            else:
                self.log_message("Model training failed")
               
        except Exception as e:
            self.log_message(f"Error training model: {e}")
   
    def test_model_thread(self):
        threading.Thread(target=self.test_model, daemon=True).start()
   
    def test_model(self):
        try:
            if self.model is None:
                self.log_message("Error: Please train or load a model first")
                return
           
            if self.env is None:
                self.log_message("Error: Please fetch data first")
                return
           
            self.log_message("Testing model performance...")
            results = test_agent(self.model, self.env, log_callback=self.log_message)
           
            if results:
                self.log_message("Testing completed successfully!")
               
                if hasattr(self.env, 'df') and len(self.env.df) > 0:
                    initial_price = float(self.env.df['Close'].iloc[self.env.lookback_window])
                    final_price = float(self.env.df['Close'].iloc[-1])
                    buy_hold_return = ((final_price - initial_price) / initial_price) * 100
                    self.log_message(f"Buy & Hold Return: {buy_hold_return:.2f}%")
                    self.log_message(f"Strategy Outperformance: {results['return_pct'] - buy_hold_return:.2f}%")
            else:
                self.log_message("Testing failed")
               
        except Exception as e:
            self.log_message(f"Error testing model: {e}")
   
    def plot_chart(self):
        try:
            if self.model is None or self.env is None:
                self.log_message("Error: Model and data required")
                return
            self.log_message("Generating chart...")
            obs, _ = self.env.reset()
            prices_list = []
            net_worth_list = []
            buy_indices = []
            sell_indices = []
            dates_list = self.data.index[self.env.lookback_window : ].tolist()
            i = 0
            done = False
            while not done:
                prev_shares = self.env.shares_held
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                if self.env.shares_held != prev_shares:
                    if self.env.shares_held > prev_shares:
                        buy_indices.append(i)
                    else:
                        sell_indices.append(i)
                prices_list.append(float(self.data['Close'].iloc[self.env.lookback_window + i]))
                net_worth_list.append(self.env.net_worth)
                i += 1
            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            # Price plot
            ax1.plot(dates_list, prices_list, label='Close Price', color='black')
            buy_idx = buy_indices
            if buy_idx:
                ax1.scatter([dates_list[j] for j in buy_idx], [prices_list[j] for j in buy_idx], color='green', marker='^', s=100, label='Buy', zorder=5)
            sell_idx = sell_indices
            if sell_idx:
                ax1.scatter([dates_list[j] for j in sell_idx], [prices_list[j] for j in sell_idx], color='red', marker='v', s=100, label='Sell', zorder=5)
            ax1.set_title(f'Trading Buy/Sell Points for {self.symbol_var.get()}')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True)
            # Net worth plot
            ax2.plot(dates_list, net_worth_list, label='Portfolio Value', color='blue')
            ax2.set_ylabel('Portfolio Value ($)')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            plt.show(block=False)  # non-blocking
            self.root.update_idletasks()
            self.log_message("Chart displayed. Close the window to continue.")
        except Exception as e:
            self.log_message(f"Error plotting chart: {e}")
def main():
    root = tk.Tk()
    app = TradingBotGUI(root)
   
    app.log_message("Welcome to Trading Bot with PPO Reinforcement Learning!")
    app.log_message("=" * 60)
    app.log_message("Instructions:")
    app.log_message("1. Enter a stock symbol (e.g., AAPL, MSFT) or crypto (e.g., BTC-USD)")
    app.log_message("2. Set date range and parameters")
    app.log_message("3. Click 'Fetch Data' to download historical data")
    app.log_message("4. Click 'Train Model' to train the PPO agent")
    app.log_message("5. Click 'Test Model' to see performance results")
    app.log_message("6. Use 'Save Model' to save your trained model")
    app.log_message("7. Use 'Load Model' to load a previously saved model")
    app.log_message("8. Check 'Continue from Saved' before training to resume from a saved model")
    app.log_message("-" * 60)
   
    try:
        root.mainloop()
    except Exception as e:
        print(f"GUI error: {e}")
if __name__ == "__main__":
    main()