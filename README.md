# Research Report: Multilayer Perceptron for Market Timing
## Subtitle: Revisiting Market Timing of AAPL Stock with Hyper-parameter Tuning

**Abstract:** This research investigates the efficacy of a Multilayer Perceptron (MLP) in timing the entry and exit points of Apple Inc. (AAPL) equity. By utilizing an automated Hyperband search, we identify a specific neural "bottleneck" architecture that prioritizes capital preservation. The study concludes that while deep learning models may struggle to exceed the absolute returns of a high-growth benchmark in aggressive bull markets, they serve as superior risk-management filters by significantly reducing maximum drawdowns and improving the Reward-to-Variability ratio.

---

## 1. Methodology & Feature Engineering
The research utilizes five rolling return windows as stationary input features to predict a 25-day future return horizon ($Ret25$).

* **Input Features ($X$):** 25, 60, 90, 120, and 240-day rolling product returns.
* **Target Variable ($y$):** Forward-shifted 25-day return ($y_{t} = Ret25_{t+25}$).
* **Data Partitioning:** A non-shuffled 60/40 train-test split was implemented to maintain the temporal integrity of the time-series data and prevent look-ahead bias.



---

## 2. Model Architecture Discovery
We employed the **Hyperband Tuning** algorithm to search the hyper-parameter space (30 trials). The search revealed that a "Bottleneck" structure is optimal for filtering market noise.

| Layer | Research Specification | Parameter Function |
| :--- | :--- | :--- |
| **Input Dense** | 21 Neurons | Feature interaction and expansion. |
| **Regularization** | 20% Dropout | Prevention of weight co-adaptation and overfitting. |
| **Bottleneck** | 6 Neurons | Latent feature compression (signal extraction). |
| **Output** | 1 Neuron | Continuous regression of expected return. |

**Optimization Strategy:** The model utilized the Adam optimizer with a learning rate of $1 \times 10^{-5}$ and Mean Absolute Error (MAE) loss. **Early Stopping** was triggered at epoch 47 (restoring weights from epoch 37) to ensure maximum generalization on unseen test data.

---

## 3. Empirical Results & Performance Attribution

### 3.1 Absolute vs. Risk-Adjusted Returns
The strategy was evaluated against a passive Buy & Hold (B&H) benchmark. While the B&H return was higher in absolute terms, the research focuses on the **Sharpe Ratio** (Risk-Adjusted Efficiency).

* **Benchmark Return:** 1177.11%
* **AI Strategy (Long-Only):** 827.71%
* **Observation:** The AI strategy achieved its returns with lower daily variance, leading to a superior Rolling 1-Year Sharpe Ratio during market corrections (2018 and 2022).



### 3.2 The "Pain Metric" (Drawdown Analysis)
The primary research hypothesis—that AI can limit downside risk—was confirmed via **Underwater Drawdown** analysis.

* **Max Drawdown (B&H):** -294.36%
* **Max Drawdown (AI):** -267.85%
* **Risk Reduction:** The model successfully mitigated downside volatility, reducing the maximum peak-to-trough decline by approximately **26%**. This confirms the model's utility as a capital preservation tool.



---

## 4. Discussion & Conclusion
The results demonstrate a critical trade-off in quantitative finance: **The Liquidity-Risk Trade-off**. 

1.  **Regime Switching:** The MLP successfully identified negative-return regimes, triggering a defensive shift to cash (0% exposure).
2.  **Survivability:** By minimizing drawdowns, the strategy ensures the longevity of capital, which is the foundational requirement for long-term compounding.
3.  **Future Work:** This research serves as a baseline for **Module M2**, where 2D Convolutional Neural Networks (CNNs) will be applied to capture spatial patterns in price action that 1D MLPs may overlook.
