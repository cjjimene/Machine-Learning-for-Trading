import experiment1 as e1
import experiment2 as e2
import numpy as np

np.random.seed(903650706)


def author():
    """
      :return: The GT username of the student
      :rtype: str
      """
    return "cjimenez36"

if __name__ == "__main__":
    e1.run_experiment_one()
    e2.run_experiment_two()