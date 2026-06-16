# Frequently Asked Questions

1. What was the objective of the project?

    The objective was to investigate whether machine learning could reliably predict cost overruns and schedule delays in construction projects. We also wanted to understand the key factors driving these outcomes using explainable AI techniques.

2. Why were the five models chosen?

    We selected models representing different machine learning paradigms. Logistic Regression represents linear methods, KNN is distance based, SVM is margin based, while XGBoost and CatBoost are ensemble boosting methods. This allowed us to compare diverse learning strategies.

3. Why wasn't hyperparameter tuning performed during baseline comparision?

    The baseline phase was meant to eastablish a fair benchmark. Using default settings allows the inherent strength and weaknesses of each algorithm without optimization bias.

4. Why was Stratified K fold cross validation used?

    - It makes maximum use of the available data by training and testing the model across multiple subsets, ultimately providing a much more reliable and robust estimate of how the model will perform on unseen data. 
    - Every single data point gets to be in the training and testing data sets, which further improves generalization. 
    - Stratification allows original distribution of each class to be maintained across each fold. 

5. Why did you choose only Logistic Regression and CatBoost for final modeling?

    Logistic Regression was the strongest model for cost prediction, while CatBoost was selected as a representative non linear ensemble model due to its strong performance and compatibility with explainability techniques.

6. Why did you create engineered features?

    Raw features often cannot capture complex project relationships. Feature engineering incorporated domain knowledge to better represent productivity, risk exposure, and project complexity.

7. Why did feature selection hurt cost prediction?

    Cost overrun information was distributed across many features. Removing seemingly unimportant variables eliminated useful interactions and reduced performance.

8. What are the key takeaways from this project?

    -  Predictive performance alone should not determine project success since understanding the underlying drivers can generate equal or greater business value.
    - Model complexity does not guarantee better performance.
    - Data quality and data richness often have a greater impact on performance than model complexity
    - Explainable AI should not be treated as a post-processing step but as an integral component of decision support systems

