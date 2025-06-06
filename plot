An interactive dashboard was built using Power BI. The underlying dataset includes all relevant indicators, with separate columns for the country and a target variable identifying fraudulent and non-fraudulent contracts. This dataset was prepared using Dataiku; the full data flow is available in the BI Zone. A Python recipe processes the input dataset indicators_all_countries and outputs INDICATORS_BI, primarily standardizing column names for clarity.

Given the scale of the data—around 3 million contracts—a random sample of 100,000 non-fraudulent contracts is taken to reduce noise and outliers while preserving the overall distribution, making the data more manageable and insightful in Power BI.

To update the dashboard, the indicators_all_countries recipe must be re-run. The resulting INDICATORS_BI_SAMPLED dataset should then be loaded into Power BI by updating the data source in Power Query.
