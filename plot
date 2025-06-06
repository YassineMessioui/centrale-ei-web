To compare model performance, we identified the best hyperparameters for two training setups: one model trained on data from all countries and another trained exclusively on French data. We evaluated both models on four different test sets: a global test set covering all countries, and three country-specific sets (France, Italy, Germany).

The comparison involved four machine learning models — XGBoost, Random Forest, Isolation Forest, and LightGBM — and focused on F1 score, recall, and precision. Our objective was not only to maximize these metrics but also to identify the model that generalizes best across all countries.

This evaluation was conducted in the Dataiku flow (Zone: Model Comparison, recipe: Model_comparison), where we ran each model using the best hyperparameters obtained via Optuna. The results showed that the XGBoost model trained on all countries achieved the best overall performance, with an F1 score of 48%, a recall of 37%, and a precision of 71%.
