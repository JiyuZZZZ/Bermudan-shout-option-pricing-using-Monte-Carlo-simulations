# 💥 Bermudan Shout Option Pricing via Monte Carlo Simulation

This project implements the pricing of a **Bermudan shout option** using simulated stock price paths under a geometric Brownian motion model.

A Bermudan shout option allows the holder to “lock in” the intrinsic value of the asset at certain times before maturity. This code estimates its fair value via **Monte Carlo simulation**.

---

### 🧮 Numerical Inversion via Bisection Method

The script implements a **bisection method** to estimate the optimal strike price `K*`  
such that the theoretical price of the Bermudan shout option equals the initial stock price `S₀`.

This inversion process simulates a common task in financial modeling:  
**deriving implied parameters** from market-observed values.
---

## 🧠 Key Features

- Simulates asset price paths using the **Euler–Maruyama method**
- Allows user-defined financial parameters:
  - Initial price, risk-free rate, volatility, drift, maturity, strike, shout level
- Supports **multiple exercise opportunities** (Bermudan-style)
- Implements “shout” logic to fix payoffs dynamically along the path
- Compares optimal shout strategies and resulting payoffs
- Visualizes sample paths and option value distribution

---

## 📦 Dependencies

- `numpy`
- `matplotlib`
- `scipy`

---

## 🚀 How to Run

```bash
python bermudan_shout_mc.py
