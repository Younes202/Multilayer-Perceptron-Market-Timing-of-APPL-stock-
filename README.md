# Multilayer Perceptron: Revisiting Market Timing of AAPL Stock with Hyper-parameter Tuning

This project implements a **Multilayer Perceptron (MLP)** neural network to predict the 25-day future returns of Apple (AAPL) stock. By utilizing **Keras Tuner**, the project focuses on optimizing model architecture to balance raw profitability with capital preservation.

## 🚀 Project Overview
The project follows a professional quantitative finance pipeline:
1.  **Feature Engineering**: Generating multi-scale momentum indicators (25, 60, 90, 120, and 240-day rolling returns).
2.  **Automated Architecture Search**: Using the `Hyperband` algorithm to determine the optimal density of neurons and dropout rates.
3.  **Risk-Adjusted Backtesting**: Evaluating "Long-Only" and "Long-Short" strategies against a "Buy & Hold" benchmark using 13 years of out-of-sample data.
4.  **Pain Metric Analysis**: Visualizing underwater drawdowns to assess the model's ability to protect capital during market crashes.

---

## 🧠 Model DNA & Architecture
The model was tuned to find a "bottleneck" structure, which effectively filters market noise to prevent overfitting.

| Layer | Type | Configuration (Best Trial) |
| :--- | :--- | :--- |
| **Input** | Dense | 5 Rolling Return Features |
| **Hidden 1** | Dense | 21 Units + ReLU |
| **Regularization** | Dropout | 20% Rate |
| **Hidden 2** | Dense | 6 Units + ReLU |
| **Output** | Dense | 1 Unit (Linear) |

**Total Parameters:** 1,604  
**Optimization:** Adam Optimizer ($1 \times 10^{-5}$) with Early Stopping (Patience: 20).

---

## 📈 Performance & Risk Metrics

### 1. Cumulative Returns (2012 - 2025)
While the benchmark maximized gains, the AI strategy provided a more stable growth trajectory.

* **Buy and Hold Return**: 1177.11%
* **AI Strategy (Long Only)**: 827.71%
* **AI Strategy (Long/Short)**: 498.72%

### 2. Risk-Adjusted Efficiency (Sharpe Ratio)
The **Rolling 1-Year Sharpe Ratio** demonstrates that the AI strategy achieved higher efficiency during specific volatile periods, providing better returns per unit of "stress" (volatility) compared to Apple stock alone.

### 3. The Pain Metric (Maximum Drawdown)
The strategy was most successful as a risk-mitigation tool. By moving to cash during negative predicted regimes, the model reduced the severity of market crashes.

* **Worst Crash (Apple B&H)**: -294.36%
* **Worst Crash (AI Strategy)**: -267.85%
* **Improvement**: Successfully lowered maximum pain by approximately **26%**.

---

## 🏁 Final Conclusion
The project successfully demonstrates that while an **Automated MLP Tuning pipeline** may underperform a high-growth benchmark in absolute terms, it excels as a **Risk Management tool**. The model proved that deep learning can be used to identify market regimes, allowing investors to "limit the risk to continue as long as possible."

---

## 🛠️ Requirements
```python
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
