Files and their purpose:
- testproject.py: The entry point to entire program. Run this file to generate charts.

- StrategyLearner.py: Executes a Random Forest Classifier to determine the optimal trades to make with the provided indicators.

- ManualStrategy.py: Executes manual trading strategy with the indicators provided.

- indicators.py: Implements the indicators used as features within StrategyLearner.py and ManualStrategy.py

- marketsimcode.py: Evaluates portfolio performance when provided a data frame that contains orders to execute.

- BagLearner.py: Accepts another learner and performs bootstrapping + aggregation to return an ensemble model.

- RTLearner.py: Executes a Random Tree Classifier model.

- experiment1.py: Runs Experiment 1 as described in the assignment and produces a related chart.

- experiment2.py: Runs Experiment 2 as described in the assignment and produces a related chart.


Instructions for running code:
- To implement the appropriate calls to the API to return run all experiments and produce charts, run the below command:

PYTHONPATH=../:. python testproject.py
