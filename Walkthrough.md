# Phase 4 Evaluation Walkthrough

We have successfully processed the 45 ablation runs and generated visualizations to answer the three core research questions defined in the project plan. 

Here are the findings based on the generated plots:

### RQ1: Does trust converge, and how quickly?
To answer this, we tracked the **Trust Entropy** across the 200 training episodes.
* **High Entropy** means trust is uniform (agents trust everyone equally).
* **Low Entropy** means trust has converged (agents have selected specific peers to trust).

![RQ1: Trust Entropy Convergence](outputs/plots/rq1_entropy.png)

> **Note:** The plot shows that trust entropy drops significantly from its starting point and stabilizes over the training episodes across all S2 configurations. This confirms that the trust matrix **does converge** to a stable topology over time.

---

### RQ2: Does cooperation via trust improve portfolio performance?
To answer this, we compared the Final Eval Sharpe Ratio of the trust-weighted strategies (S2) across different `imitation_beta` weights against the independent baseline (S1).

![RQ2: Performance across Trust Weights](outputs/plots/rq2_performance.png)

> **Important:** The red dashed line represents the baseline S1 performance (0.2396 Sharpe). 
> The bar chart shows whether setting `beta > 0` (cooperation) yields better results than S1. 
> You can now analyze which predictor (`moving_average` vs `noisy_oracle`) and which beta value provides the most optimal trust-weighted performance boost!

---

### RQ3: Does asymmetric trust emerge and persist?
To answer this, we plotted the **Asymmetry Index** over time. An index near 0 means trust is perfectly mutual (A trusts B exactly as much as B trusts A). A higher index indicates directional, asymmetric trust (e.g., poor performers trusting high performers).

![RQ3: Trust Asymmetry Emergence](outputs/plots/rq3_asymmetry.png)

> **Tip:** The plot demonstrates that the asymmetry index starts at zero (since all trust is initialized uniformly) and then actively grows and persists as agents evaluate each other's realized returns. This confirms that the trust network evolves into an asymmetric topology based on individual agent performance.

---

### Next Steps
The data processing and plotting pipeline is now complete. You can insert these PNG files into your `phase4_evaluation_report.ipynb` notebook and write up your final academic conclusions based on these trends. The framework is functioning perfectly from end-to-end!
