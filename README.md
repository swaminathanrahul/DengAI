# Dengue Fever Prediction

This repository contains the code and models developed for the [Dengue Prediction competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/) hosted on DrivenData. The goal of this competition is to predict the number of dengue fever cases each week in two cities: San Juan, Puerto Rico, and Iquitos, Peru.

## Outcomes

- Final submission score: 26.1
- Explored and preprocessed the provided data to extract meaningful features.
- Implemented and fine-tuned multiple machine learning models, including regression and time series forecasting algorithms.
- Evaluated model performance using appropriate metrics such as Mean Absolute Error (MAE) to assess accuracy.
- Submitted predictions to the DrivenData platform and achieved competitive results on the leaderboard.

## Learnings

- Importance of feature engineering: Found that incorporating lag features and domain-specific transformations significantly improved model performance.
- Model selection: Experimented with different algorithms such as Random Forest, Gradient Boosting, and LSTM to identify the most suitable approach for the task.
- Dealing with imbalanced data: Employed techniques like oversampling and weighted loss functions to address the class imbalance present in the target variable.
- Collaboration and knowledge sharing: Engaged with the community on forums and leveraged shared insights and strategies to enhance model development and performance.

## Repository Structure

- data/: Contains the raw and processed datasets used in the project.
- notebooks/: Jupyter notebooks documenting the data exploration, preprocessing, modeling, and evaluation stages.
- src/: Python scripts for feature engineering, model training, and evaluation.
- models/: Saved trained models for future use or deployment.
- README.md: Overview of the project, outcomes, and learnings (you're here!).

## Requirements

- Python 3.12
- Required packages listed in requirements.txt

## Getting Started

- Clone this repository: git clone https://github.com/your_username/dengue-prediction.git
- Install dependencies: pip install -r requirements.
- Navigate to the notebooks/ directory to explore the project in detail.
- Refer to the notebooks and scripts for code implementation and model development.
- For any inquiries or feedback, feel free to reach out to the repository owner.

## Acknowledgments

- DrivenData for hosting the Dengue Prediction competition.
- Contributors and participants in the competition for valuable discussions and insights.
- Open-source libraries and resources utilized in the project.

Feel free to customize and expand upon this template as needed to reflect your specific contributions and experiences in the competition!