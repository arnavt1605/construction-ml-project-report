# Discussion & Conclusions

## Summary of Key Findings

This study developed an Explainable Machine Learning framework to predict two critical construction project outcomes: **Cost Overrun** and **Schedule Delay**. The framework incorporated exploratory analysis, feature engineering, feature selection, predictive modeling, and explainability techniques.

### Key Findings

- Cost Overrun and Delay are fundamentally different prediction problems:
  - Cost Overrun exhibits a weak, distributed, and predominantly linear signal.
  - Delay is driven by a smaller set of strong, non-linear, and interpretable factors.

- Feature engineering improved representation of latent project characteristics through six derived features, including ratio, interaction, complexity, and stress indicators.

- Feature selection affected the two targets differently:
  - Delay prediction benefited from retaining only positively contributing features.
  - Cost prediction required the complete feature set to preserve predictive signal.

- For classical machine learning:
  - Logistic Regression performed best for Cost Overrun prediction.
  - CatBoost significantly outperformed Logistic Regression for Delay prediction.

- Among advanced models:
  - TabPFN achieved the strongest Cost Overrun performance.
  - CatBoost remained the best-performing model for Delay prediction.

- TabNet consistently underperformed across both tasks, suggesting that attention-based tabular architectures require larger datasets to realize their advantages.

- Explainability analysis (PFI, PDP, and SHAP) confirmed model-level findings:
  - Cost Overrun is characterized by diffuse and weak feature contributions.
  - Delay is strongly influenced by Productivity Index and Risk Impact.

- Statistical validation confirmed that observed performance differences between models were significant and not due to random variation.

---

## Conclusions

### Major Contributions

#### Construction Risk Explainability

- SHAP, PFI, and PDP successfully provided interpretable insights into construction project risks.
- Productivity Index and Risk Impact emerged as dominant drivers of project delays.
- Cost Overrun was identified as a multi-factor phenomenon with no single dominant predictor.

#### Evaluation of Modern Tabular Models

- TabPFN demonstrated strong performance on small tabular datasets with weak predictive signal.
- Pretrained in-context learning models represent a promising alternative to conventional machine learning approaches in data-constrained environments.

#### Understanding Prediction Difficulty

- Cost Overrun and Delay differ substantially in their predictability.
- Cost Overrun is influenced by contractual, environmental, and behavioral complexities that are difficult to capture through structured features.
- Delay is more operationally driven and therefore more predictable and actionable.

### Overall Conclusion

Explainable Machine Learning can function not only as a predictive tool but also as a knowledge-discovery framework, enabling deeper understanding of construction project risks and outcome drivers.

---

## Limitations

The findings should be interpreted in light of several limitations:

### Small Dataset Size

- The dataset contains only **150 projects**.
- Limited sample size increases variance in cross-validation results.
- Deep learning models such as TabNet and MLP are likely constrained by insufficient training data.

### Binary Outcome Formulation

- Cost Overrun and Delay were modeled as binary outcomes.
- The actual magnitude of overruns and delays was not considered.
- Regression or ordinal prediction approaches may provide richer insights.

### Feature Coverage

- Only 19 project features were included.
- Important factors such as:
  - Material price volatility
  - Workforce turnover
  - Contract disputes
  - Geopolitical influences
- were not available in the dataset.

### Domain Generalizability

- The study was conducted using a single construction project dataset.
- Results may differ across project types, industries, or geographic regions.


### Lack of Temporal Context

- Projects were treated as independent observations.
- Temporal, organizational, and contractual dependencies were not modeled.

---

## Future Work

Several promising research directions emerge from this study.

### Regression-Based Prediction

- Predict the magnitude of:
  - Cost Overrun (%)
  - Schedule Delay (days)
- rather than binary outcomes.
- Investigate multi-task learning frameworks that jointly model both targets.

### Expanded Feature Space

- Incorporate additional project indicators such as:
  - Labor productivity metrics
  - Material cost indices
  - Weather severity scores
  - Contractor financial health
  - Project management maturity


### Ensemble Modeling

- Combine complementary model strengths through:
  - Model stacking
  - Blending
  - Meta-learning approaches


### Real-Time Risk Prediction

- Evaluate models in prospective settings using information available at project inception or milestone checkpoints.
- Develop time-aware feature engineering and validation strategies.

### Graph-Based Construction Modeling

- Explore Graph Neural Networks (GNNs) for modeling:
  - Subcontractor relationships
  - Workflow dependencies
  - Risk propagation networks
- Particularly relevant for large and complex infrastructure projects.