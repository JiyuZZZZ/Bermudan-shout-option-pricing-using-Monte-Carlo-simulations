#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:05:24 2024

@author: zhangjiyu
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
from scipy.stats import norm

# Set the standard parameters.
S_0 = 100
r = 0.005
mu = 0.005
sigma = np.sqrt(2*r)
T = 1.5
K = 100
E = 110

################ 1.Inputting data
def input_data():
    """
    This is a function responsible for inputting data and it can deal with 
    situations when an inputted parameter is clearly wrong.

    Returns
    -------
    tuple
        The tuple contains all of the parameter values.
    """
    #Ask user to input the value of parameters.
    S_0 = input("Please input the initial price of the stock:")
    r = input("Please input the risk-free interest rate r(per annum):")
    mu = input("Please input the stock price drift μ:")
    sigma = input("Please input the stock price volatility σ(per annum):")
    t0 = input("Please input the start time t0 of the option:")
    T = input("Please input the expiry time T of the option:")
    K = input("Please input the strike price K of the option:")
    E = input("Please input the shout price E of the option:")
    N = input("Please input the number of time steps:")
    M = input("Please input the number of samples:")
    
    try:
        S_0 = float(S_0)
        r = float(r)
        mu = float(mu)
        sigma = float(sigma)
        t0 = float(t0)
        T = float(T)
        K = float(K)
        E = float(E)
        N = int(N)
        M = int(M)
    except ValueError as e:
        return f"Error: The inputted parameter {e} is not a numeric value."
    
    #Verify the correctness of parameters.
    for param_name, param_value in [('S_0', S_0), ('r', r), ('mu', mu), 
                                    ('sigma', sigma), ('t0', t0),
                                    ('T', T), ('K', K), ('E', E),
                                    ('N', N), ('M', M)]:
        if not isinstance(param_value, (float, int)):
            return f"Error: The inputted parameter {param_name} is not a\
                numeric value."
        if param_value < 0:
            return f"Error: The inputted parameter {param_name} is negative."
        if param_name == 'M' and param_value < 1:
            return "Error: The inputted parameter M should be greater than or\
                equal to 1."
        if param_name == 'M' and param_value %1!=0:
            return "Error: The inputted parameter M should be a integer."
        if param_name == 'N' and param_value < 1:
            return "Error: The inputted parameter N should be greater than or\
                equal to 1."
        if param_name == 'N' and param_value %1!=0:
            return "Error: The inputted parameter N should be a integer."
    return S_0, r, sigma, T, K, M



################ 2.Modalling the stock price
def price_euler(S_0, mu, sigma, t0, T, N):
    '''
    This is a function that generates sample paths of the Seasonal Volatility
    model using the Euler-Maruyama scheme.

    Parameters
    ----------
    S_0 : TYPE
        The initial price of the stock.
    mu : TYPE
        The stock price drift μ.
    sigma : TYPE
        The stock price volatility σ(per annum).
    t0 : TYPE
        The start time t0 of the option.
    T : TYPE
        The expiry time T of the option.
    N : TYPE
        The number of time steps.

    Returns
    -------
    S: The sample paths.

    '''
    
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S_0

    for i in range(N):
        S[i + 1] = S[i] + mu * S[i] * dt + sigma * S[i] * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi )) * np.random.normal(0, np.sqrt(dt))

    return S

# Set parameters.
N = 50
M = 1000
N_values = np.arange(0,N+1,1)

# Calculate the stock prices for t0=0 and t0=0.5.
P_Euler_0 = price_euler(S_0, mu, sigma, 0, T, N)
P_Euler_05 = price_euler(S_0, mu, sigma, 0.5, T, N)
print(P_Euler_0, P_Euler_05)

# Plot a histgram to show the prices distribution. (Figure 3 in report)
prices_0 = []
prices_1 = []
for i in range(5000):
    a = price_euler(S_0, mu, sigma, 0, T, N)
    b = price_euler(S_0, mu, sigma, 0.5, T, N)
    prices_0.append(a[N])
    prices_1.append(b[N])

plt.hist(prices_0, bins=40, label='t0=0', density=True)
plt.hist(prices_1, bins=40, label='t0=0.5', alpha=0.5, density=True)
plt.title('Histgram of 5000 sample stock prices')
plt.xlabel('Stock prices')
plt.ylabel('Density')
plt.legend()
plt.show()


# Plot the sample paths to show the volatility differences. (Figure 1 in report)
plt.plot(N_values, P_Euler_0 , label='t0=0')
plt.plot(N_values, P_Euler_05, label='t0=0.5')
plt.title('Sample paths of t0=0 and to=0.5')
plt.xlabel('Steps')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()

# Plot the volatility terms to show the differences. (Figure 2 in report)
X_values_0 = np.arange(0, T, 0.001)
X_values_05 = np.arange(0.5, 0.5+T, 0.001)
Y_0 = sigma*(0.9*np.sin(2*np.pi*X_values_0)+1)
Y_05 = sigma*(0.9*np.sin(2*np.pi*X_values_05)+1)

plt.plot(X_values_0, Y_0 , label='t0=0')
plt.plot(X_values_0, Y_05, label='t0=0.5')
plt.xlim(0, T)
plt.title('Volatility function images')
plt.xlabel('Time from t_0')
plt.ylabel('Volatility')
plt.legend()
plt.show()


# Monte Carlo.
def price_euler_mc(S_0, mu, sigma, t0, T, N, M):
    '''
    This is a function that generates M Monte Carlo samples of the Seasonal Volatility
    model using the Euler-Maruyama scheme.

    Parameters
    ----------
    S_0 : float
        The initial price of the stock.
    mu : float
        The stock price drift μ.
    sigma : float
        The stock price volatility σ(per annum).
    t0 : float
        The start time t0 of the option.
    T : float
        The expiry time T of the option.
    N : int
        The number of time steps.
    M : int
        The number of samples.
    Returns
    -------
    S: The Monte Carlo samples.

    '''
    dt = T / N
    S = S_0 * np.ones(M)

    for i in range(N):
        S = S + mu * S * dt + sigma * S * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi )) * np.random.normal(0, np.sqrt(dt), M)

    return S

# Plot the time consumed by the two algorithms to compare the complexity. (Figure 4 in report)
M = 500
Time_1 = []
Time_2 = []

# Set M from 50 to avoid the effect of low values.
for i in range(50, M):
    time_start = time.perf_counter()
    price_euler_mc(S_0, mu, sigma, 0, T, N, i)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_1.append(totaltime)

for i in range(50, M):
    time_start = time.perf_counter()
    for j in range(i):
        price_euler(S_0, mu, sigma, 0, T, N)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_2.append(totaltime)

M_values = np.arange(50, M, 1)
# Show the slopes.
coeffs_1 = np.polyfit(np.log(M_values), np.log(Time_1), 1)
slope_1 = coeffs_1[0]
coeffs_2 = np.polyfit(np.log(M_values), np.log(Time_2), 1)
slope_2 = coeffs_2[0]
plt.loglog(M_values, Time_1 , label='Vectorized')
plt.loglog(M_values, Time_2, label='Looping over')
plt.title('Algorithm complexity comparison')
plt.text(100, 0.01, f"Slope={slope_1:6.2f}", fontsize=12, color='blue', ha='center')
plt.text(95, 0.1, f"Slope={slope_2:6.2f}", fontsize=12, color='orange', ha='center')
plt.xlabel('M values')
plt.ylabel('Time')
plt.legend()
plt.show()



################ 3.Runge-Kutta scheme
def price_rk_mc(S_0, mu, sigma, t0, T, N, M):
    '''
    This is a function that generates M Monte Carlo samples of the Seasonal Volatility
    model using the Euler-Maruyama scheme.

    Parameters
    ----------
    S_0 : float
        The initial price of the stock.
    mu : float
        The stock price drift μ.
    sigma : float
        The stock price volatility σ(per annum).
    t0 : float
        The start time t0 of the option.
    T : float
        The expiry time T of the option.
    N : int
        The number of time steps.
    M : int
        The number of samples.
    Returns
    -------
    S: The Monte Carlo samples.

    '''
    dt = T / N
    S = S_0 * np.ones(M)

    for i in range(N):
        b_1 =  sigma * S * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
        S_hat = S + mu * S * dt + b_1 * np.sqrt(dt)
        b_2 = sigma * S_hat * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
        norm_variables = np.random.normal(0, np.sqrt(dt), M)
        S = S + mu * S * dt + b_1 * norm_variables + ((
            norm_variables * norm_variables) - dt
            ) * (b_2 - b_1) / (2 * np.sqrt(dt))
    return S

# Plot to compare the algorithms complexity. (Figure 5 in report)
Time_3 = []
n = 500
for i in range(15,n):
    time_start = time.perf_counter()
    price = price_rk_mc(S_0, mu, sigma, 0, T, i, M)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_3.append(totaltime)

N_values = np.arange(15,n,1)

coeffs = np.polyfit(np.log(N_values), np.log(Time_3), 1)
slope = coeffs[0]

plt.figure()
plt.loglog(N_values, Time_3, markersize=2, label="N")
plt.title('Algorithm complexity for different values of N')
plt.xlabel('Values of N')
plt.ylabel('Time')
plt.text(17, 0.002, f'Slope={slope:6.2f}', fontsize=10, color='blue')
plt.legend()
plt.show()

# Plot to show the convergency. (Figure 6 in report)
Everage_price_N = []
Everage_price_M = []

for i in range(1,2000):
    price_N = np.mean(price_rk_mc(S_0, mu, sigma, 0, T, i, 1000))
    Everage_price_N.append(price_N)
    price_M = np.mean(price_rk_mc(S_0, mu, sigma, 0, T, 200, i))
    Everage_price_M.append(price_M)
    
N_values = np.arange(1,2000,1)
plt.figure()
plt.plot(N_values, Everage_price_N, markersize=2, label="N, when M=1000", alpha=0.5)
plt.plot(N_values, Everage_price_M, markersize=2, label="M, when N=200", alpha=0.5)
plt.title('Everage prices of different N')
plt.xlabel('N or M')
plt.ylabel('Everage Prices')
plt.legend()
plt.show()


# Calculate discounted average prices for different mu values.
Discounted_average_price_mu = []
mu_values = np.arange(0,0.01,0.0001)
for i in mu_values:
    price = np.mean(price_rk_mc(S_0, i, sigma, 0, T, 100, 10000)) * np.exp(-r*T)
    Discounted_average_price_mu.append(price)
    

N_values = np.arange(1,n,1)
#Plot to show the discounted average prices for different mu values. (Figure 7 in report)
plt.figure()
plt.plot(mu_values, Discounted_average_price_mu, markersize=2)
plt.axhline(y=100, color='r', linestyle='--', label='Average price = 100')
plt.axvline(x=0.005, color='black', linestyle='--', label='mu = 0.005')
plt.title('Discounted average prices of different mu')
plt.xlabel('mu')
plt.ylabel('Average prices')
plt.legend()



################ 4.Option price
def option_price_rk_mc(S_0, mu, sigma, t0, T, N, M, E):
    '''
    This is a function that generates M Monte Carlo samples of the Seasonal Volatility
    model using the Euler-Maruyama scheme.

    Parameters
    ----------
    S_0 : float
        The initial price of the stock.
    mu : float
        The stock price drift μ.
    sigma : float
        The stock price volatility σ(per annum).
    t0 : float
        The start time t0 of the option.
    T : float
        The expiry time T of the option.
    N : int
        The number of time steps.
    M : int
        The number of samples.
    E : The shout price of option
    Returns
    -------
    S: The Monte Carlo samples.

    '''
    dt = T / N
    S = S_0 * np.ones(M)
    shout_prices = np.zeros(M)
    V = np.zeros(M)
    
    for j in range(M):
        shout = False
        for i in range(N):
            b_1 =  sigma * S[j] * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
            S_hat = S[j] + mu * S[j] * dt + b_1 * np.sqrt(dt)
            b_2 = sigma * S_hat * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
            norm_variables = np.random.normal(0, np.sqrt(dt))
            S[j] = S[j] + mu * S[j] * dt + b_1 * norm_variables + ((
                norm_variables * norm_variables) - dt
                ) * (b_2 - b_1) / (2 * np.sqrt(dt))
            if S[j] > E and shout == False:
                shout = True
                shout_prices[j] = S[j] - K
        V[j] = max(S[j]-K, shout_prices[j])
        
    aM = np.mean(V)
    bM = np.std(V, ddof=1)
    return aM, bM

# Calculate the Bermudan shout call option price when M=200000 and N=500.
M = 200000
N = 500
aM, bM = option_price_rk_mc(S_0, mu, sigma, 0, T, N, M, E)
confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)  
margin_of_error = z_score * bM / np.sqrt(M)

lower_bound = aM - margin_of_error
upper_bound = aM + margin_of_error
print(2*margin_of_error)
print(f"Confidence Interval ({confidence_level * 100}%): [{lower_bound}, {upper_bound}]")



################ 5.Autithetic variates
def option_price_rk_mc_antithetic(S_0, mu, sigma, t0, T, N, M, E):
    '''
    This is a function that generates M Monte Carlo samples of the Seasonal Volatility
    model using the Euler-Maruyama scheme.

    Parameters
    ----------
    S_0 : float
        The initial price of the stock.
    mu : float
        The stock price drift μ.
    sigma : float
        The stock price volatility σ(per annum).
    t0 : float
        The start time t0 of the option.
    T : float
        The expiry time T of the option.
    N : int
        The number of time steps.
    M : int
        The number of samples.
    Returns
    -------
    S: The Monte Carlo samples.

    '''
    dt = T / N
    S_1 = S_0 * np.ones(M)
    S_2 = S_0 * np.ones(M)
    shout_prices_1 = np.zeros(M)
    shout_prices_2 = np.zeros(M)
    V_1 = np.zeros(M)
    V_2 = np.zeros(M)
    
    for j in range(M):
        shout_1 = False
        shout_2 = False
        for i in range(N):
            b_1_1 =  sigma * S_1[j] * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
            S_hat_1 = S_1[j] + mu * S_1[j] * dt + b_1_1 * np.sqrt(dt)
            b_2_1 = sigma * S_hat_1 * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
            norm_variables = np.random.normal(0, np.sqrt(dt))
            S_1[j] = S_1[j] + mu * S_1[j] * dt + b_1_1 * norm_variables + ((
                norm_variables * norm_variables) - dt
                ) * (b_2_1 - b_1_1) / (2 * np.sqrt(dt))
            if S_1[j] > E and shout_1 == False:
                shout_1 = True
                shout_prices_1[j] = S_1[j] - K
                
            minus_norm_variables = -1 * norm_variables
            b_1_2 =  sigma * S_2[j] * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
            S_hat_2 = S_2[j] + mu * S_2[j] * dt + b_1_2 * np.sqrt(dt)
            b_2_2 = sigma * S_hat_2 * (1 + 0.9 * np.sin(2 * (t0 + i * dt) * np.pi ))
            S_2[j] = S_2[j] + mu * S_2[j] * dt + b_1_2 * minus_norm_variables + ((
                minus_norm_variables * minus_norm_variables) - dt
                ) * (b_2_2 - b_1_2) / (2 * np.sqrt(dt))
            if S_2[j] > E and shout_2 == False:
                shout_2 = True
                shout_prices_2[j] = S_2[j] - K
        V_1[j] = max(S_1[j]-K, shout_prices_1[j])
        V_2[j] = max(S_2[j]-K, shout_prices_2[j])
        
    V = (V_1+V_2)/2
    aM = np.mean(V)
    bM = np.std(V, ddof=1)
    return aM, bM

# Calculate three sets of results to compare algorithm performance. (Table 1 in report)
M=1000
mean_antithetic, std_err_antithetic = option_price_rk_mc_antithetic(S_0, mu, sigma, 0, T, N, M,E)
z_value_antithetic = stats.norm.ppf(0.975)
confidence_interval_antithetic = (mean_antithetic - z_value_antithetic * std_err_antithetic/np.sqrt(M), mean_antithetic + z_value_antithetic * std_err_antithetic/ np.sqrt(M))
print(2*z_value_antithetic*std_err_antithetic/np.sqrt(M))
print(confidence_interval_antithetic)

mean, std_err = option_price_rk_mc(S_0, mu, sigma, 0, T, N, M,E)
z_value = stats.norm.ppf(0.975)
confidence_interval = (mean - z_value * std_err/np.sqrt(M), mean + z_value * std_err/ np.sqrt(M))
print(2*z_value*std_err/np.sqrt(M))
print(confidence_interval)


M=1500
mean_antithetic, std_err_antithetic = option_price_rk_mc_antithetic(S_0, mu, sigma, 0, T, N, M,E)
z_value_antithetic = stats.norm.ppf(0.975)
confidence_interval_antithetic = (mean_antithetic - z_value_antithetic * std_err_antithetic/np.sqrt(M), mean_antithetic + z_value_antithetic * std_err_antithetic/ np.sqrt(M))
print(2*z_value_antithetic*std_err_antithetic/np.sqrt(M))
print(confidence_interval_antithetic)

mean, std_err = option_price_rk_mc(S_0, mu, sigma, 0, T, N, M,E)
z_value = stats.norm.ppf(0.975)
confidence_interval = (mean - z_value * std_err/np.sqrt(M), mean + z_value * std_err/ np.sqrt(M))
print(2*z_value*std_err/np.sqrt(M))
print(confidence_interval)


M=2000
mean_antithetic, std_err_antithetic = option_price_rk_mc_antithetic(S_0, mu, sigma, 0, T, N, M,E)
z_value_antithetic = stats.norm.ppf(0.975)
confidence_interval_antithetic = (mean_antithetic - z_value_antithetic * std_err_antithetic/np.sqrt(M), mean_antithetic + z_value_antithetic * std_err_antithetic/ np.sqrt(M))
print(2*z_value_antithetic*std_err_antithetic/np.sqrt(M))
print(confidence_interval_antithetic)

mean, std_err = option_price_rk_mc(S_0, mu, sigma, 0, T, N, M,E)
z_value = stats.norm.ppf(0.975)
confidence_interval = (mean - z_value * std_err/np.sqrt(M), mean + z_value * std_err/ np.sqrt(M))
print(2*z_value*std_err/np.sqrt(M))
print(confidence_interval)


Time_4 = []
Time_5 = []
price_withoutAV = []
price_withAV = []
M_values = np.arange(100,2000,100)
N = 1000

for i in M_values:
    time_start = time.perf_counter()
    aM, bM = option_price_rk_mc(S_0, mu, sigma, 0, T, N, i, E)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_4.append(totaltime)
    price_withoutAV.append(aM)
    
    time_start = time.perf_counter()
    aM, bM = option_price_rk_mc_antithetic(S_0, mu, sigma, 0, T, N, i, E)
    time_end = time.perf_counter()
    totaltime = time_end - time_start
    Time_5.append(totaltime)
    price_withAV.append(aM)



coeffs_1 = np.polyfit(np.log(M_values), np.log(Time_4), 1)
slope_1 = coeffs_1[0]
coeffs_2 = np.polyfit(np.log(M_values), np.log(Time_5), 1)
slope_2 = coeffs_2[0]

# Plot to show option price volatility. (Figure 8 in report)
plt.figure()
plt.plot(M_values, price_withoutAV, markersize=2, label="without A.V.")
plt.plot(M_values, price_withAV, markersize=2, label="with A.V.")
plt.title('Option prices for different values of M with and without A.V.')
plt.xlabel('Values of M')
plt.ylabel('Option prices')
plt.legend()
plt.show()


# Plot to show algorithms complexity. (Figure 9 in report)
plt.figure()
plt.loglog(M_values, Time_4, markersize=2, label="without A.V.")
plt.loglog(M_values, Time_5, markersize=2, label="with A.V.")
plt.title('Algorithm complexity for different values of M with and without A.V.')
plt.xlabel('Values of M')
plt.ylabel('Time')
plt.text(1000, 5, f'Slope={slope_1:6.2f}', fontsize=10, color='blue')
plt.text(300, 5, f'Slope={slope_2:6.2f}', fontsize=10, color='orange')
plt.legend()
plt.show()




N=10
# Plot the Bermudan shout call option prices for different t0 values. (Figure 10 in report)
M=10000
t0_values = np.arange(0,5,0.01)
prices_t0 = []
for i in t0_values:
    mean_antithetic, std_err_antithetic = option_price_rk_mc_antithetic(S_0, mu, sigma, i, T, N, M,E)
    prices_t0.append(mean_antithetic)

plt.figure()
plt.plot(t0_values, prices_t0, markersize=2)
plt.title('Option prices of different t0')
plt.xlabel('t0')
plt.ylabel('Option prices')
plt.show()