# Customer churn prediction system

![Churn image](images/churn.webp)
## What is churn?
Customer churn, the loss of clients who cease business with a company, is a critical challenge in the telecom industry. With an annual churn rate of 15-25%, the market is intensely competitive, and customers frequently switch providers.

While personalized retention efforts for all customers are cost-prohibitive, companies can gain a significant advantage by using predictive analytics to identify "high-risk" customers likely to churn. This allows firms to focus retention strategies efficiently, as retaining an existing customer is far less expensive than acquiring a new one.

By developing a holistic view of customer interactions, telecom companies can proactively address churn. Success in this market hinges on reducing attrition and fostering loyalty, which directly lowers operational costs and drives profitability.

## Objectives:
- Finding the percentage of Churn Customers and customers that keep in with the active services.
- Analysing the data in terms of various features responsible for customer Churn
- Building a fully monitored system end-to-end for proactively identify churn using machine learning models
## Dataset: 
- [New Cell2cell dataset](https://www.kaggle.com/datasets/jpacse/telecom-churn-new-cell2cell-dataset)
- Alternative: [Telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 
## Implementation 
- Libraries: Numpy, Pandas, Matplotlib, scikit-learn, pytorch
- Models: Neural networks, XGBoost, Random Forest
- Tuning techniques: Gridsearch, Optuna
- Experiment tracking: MLFlow

## Project structure
```text
├── README.md
├── artifacts
├── churn.html
├── config
│   ├── config_ingest.yaml
│   ├── config_process.yaml
│   ├── config_train.yaml
│   ├── config_train_nn.yaml
│   ├── config_train_rf.yaml
│   └── logging.conf
├── data
│   ├── metadata
│   ├── processed
│   ├── raw
│   └── snapshots.dvc
├── dvc.lock
├── dvc_refresh.sh
├── git.sh
├── images
│   ├── Churn_predicition_system_architecture.png
│   └── churn.webp
├── models
├── push.py
├── requirements.txt
├── retrain.py
├── run_output.log
├── setup.py
├── src
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── ml_models.py
│   │   ├── routers
│   │   │   ├── metrics.py
│   │   │   ├── predict.py
│   │   │   ├── train.py
│   │   │   └── validate.py
│   │   └── templates
│   ├── data
│   ├── data_pipeline
│   │   ├── ingest.py
│   │   ├── pipeline_data.py
│   │   ├── preprocess.py
│   │   └── validate_after_preprocess.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── churn_nn.py
│   │   ├── predict.py
│   │   ├── train.py
│   │   ├── train_NN
│   │   │   └── neural_net.py
│   │   ├── train_RandomForest.py
│   │   ├── train_xgboost.py
│   │   ├── tuning
│   │   │   └── optuna_nn.py
│   │   └── utils
│   │       ├── eval_nn.py
│   │       ├── train_util.py
│   │       └── util_nn.py
│   └── monitoring
│       ├── __init__.py
│       ├── drift.py
│       └── metrics.py
├── tes.py
├── test.py
└── tests
    ├── integration
    │   └── test_pipeline.py
    └── unit
        └── test_preprocess.py
```
## Installation
```bash
pip install -r requirements.txt
```


## Authors
- [Jordan Buwa](https://github.com/Jordan-buwa)
- [Aderonke Ajefolakemi](https://github.com/Ronkecrown)
- [Wycliffe Nzoli Nzomo](https://github.com/wycliffenzomo)
