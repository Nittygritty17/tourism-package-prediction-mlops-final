1. Business Context
"Visit with Us" is a premier travel agency transition toward a data-centric model to refine its customer engagement. The introduction of the Wellness Tourism Package highlighted a critical bottleneck: the company struggled to pinpoint the ideal audience for this niche offering.

The traditional, manual method for lead selection proved to be unreliable and labor-intensive, often resulting in wasted resources and overlooked prospects. To resolve this, the company is shifting toward a scalable, automated predictive framework. By utilizing MLOps—incorporating automated preprocessing, model lifecycle management, and CI/CD via GitHub Actions—the organization can maintain a high-performing predictive system that evolves alongside shifting consumer trends.

2. Project Objective
The goal is to develop and launch a comprehensive MLOps ecosystem that forecasts a customer's propensity to buy the Wellness Tourism Package prior to any marketing outreach.

As the MLOps Engineer, the mission is to:

Standardize and automate data hygiene and feature engineering.

Develop and validate a high-accuracy classification model.

Establish seamless CI/CD workflows through GitHub Actions.

Ensure the system is reproducible, scalable, and ready for continuous updates.
This framework allows leadership to move from intuitive guessing to data-backed precision, maximizing conversion rates while minimizing marketing spend.

3. Project Scope
The MLOps architecture covers the entire machine learning lifecycle, specifically:

Data Management: Automated ingestion and quality validation.

Preprocessing: Standardized cleaning and feature encoding.

Modeling: Rigorous training, hyperparameter tuning, and evaluation.

Deployment: Preparation of model artifacts for production environments.

Automation: Full CI/CD integration to handle code changes and model retraining.

4. Data Dictionary
The predictive model relies on a blend of demographic profiles and historical engagement metrics.

Target Variable: ProdTaken (Binary: 1 for Purchase, 0 for No Purchase).

Demographic Profile: Includes age, occupation, city tier, gender, marital status, job designation, and monthly income.

Logistics & Preferences: Covers travel party size (adults/children), preferred hotel star rating, passport status, car ownership, and annual trip frequency.

Engagement Metrics: Captures the "sales touchpoints," such as how they were contacted, the specific product pitched, the duration of the pitch, satisfaction scores, and the frequency of follow-up attempts.

5. Technology Stack
Core Language: Python

ML Framework: Scikit-learn

Version Control: Git & GitHub

Automation/DevOps: GitHub Actions

Workflow Standards: MLOps best practices for automated testing, tracking, and deployment.

6. Business Impact
Successfully implementing this pipeline delivers several strategic advantages:

Higher Conversion: Targets only high-probability leads.

Operational Excellence: Eliminates manual errors and reduces staff workload.

Agility: The CI/CD component ensures the model stays relevant as market conditions change.

Consistency: Provides a unified, reproducible framework for all future package launches.

7. Conclusion
By bridging the gap between machine learning and DevOps, "Visit with Us" moves to the forefront of the tourism industry. This MLOps pipeline does more than just predict sales; it establishes a robust, automated infrastructure that turns raw data into a sustainable competitive advantage, ensuring the company remains responsive to its customers and efficient in its operations.---


---

# Travel Prediction MLOps App

This Hugging Face Space hosts a Streamlit app for predicting whether a customer is likely to purchase a tourism package.

## Files
- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker setup for Hugging Face Space

## Model Source
The app loads the trained model from:

`nittygritty2106/travelpredictionmlops`

## Space URL
`nittygritty2106/travelpredictionmlops-app` 
