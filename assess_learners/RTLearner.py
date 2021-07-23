""""""
"""  		  	   		   	 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
"""
import random

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class RTLearner(object):
    """
      This is a Random Decision Tree Regression Learner. It is implemented correctly.

      :param verbose: If “verbose” is True, your code can print out
      information for debugging.
          If verbose = False your code should not generate ANY output. When we
          test your code, verbose will be False.
      :type verbose: bool
      """

    def __init__(self, leaf_size=1, verbose=False):
        """
            Constructor method
            """
        self.leaf_size = leaf_size
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
            :return: The GT username of the student
            :rtype: str
            """
        return "cjimenez36"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
            Add training data to learner

            :param data_x: A set of feature values used to train the learner
            :type data_x: numpy.ndarray
            :param data_y: The value we are attempting to predict given the X data
            :type data_y: numpy.ndarray
            """

        # combine data into a single ndarray
        data = np.append(data_x, data_y.reshape(data_x.shape[0], 1), axis=1)
        # build and save tree
        self.tree = self.build_tree(data)
        return data

    def build_tree(self, data):
        """
            Build tree recursively.

            :param data ndarray containing features and target where data[:, -1] is the target
            :type numpy.ndarray

            :return: numpy.ndarray
        """
        # Ensure m > specified leaf size. If not, return mean of y for remaining samples.
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
        # Ensure y values have more than one possible value. If not, return y.
        if np.all(data[:-1] == data[0, -1]):
            return np.array([[-1, data[0, -1], np.nan, np.nan]])
        else:
            # Determine best value to split on.
            i = range(data.shape[1] - 1)
            i = random.choice(i)
            # Account for edge case where median and max are the same, which would
            # result in infinite loop where max leaf size = 1. Returns mean.
            if np.median(data[:, i]) == np.max(data[:, i]):
                return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
            # Ensure all values of x are different. Take mean of y if not.
            if np.all(data[:, i] == data[0, i]):
                return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
            split_val = np.median(data[:, i])
            left_tree = self.build_tree(data[data[:, i] <= split_val])
            right_tree = self.build_tree(data[data[:, i] > split_val])
            root = np.array([[i, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack([np.vstack([root, left_tree]), right_tree])

    def query(self, points):
        """
            Estimate a set of test points given the model we built.

            :param points: A numpy array with each row corresponding to a specific
            query.
            :type points: numpy.ndarray
            :return: The predicted result of the input data according to the trained
            model
            :rtype: numpy.ndarray
            """
        y = np.empty((points.shape[0],))
        for row in range(points.shape[0]):
            node = 0
            # Loop through nodes until leaf is arrived at.
            while self.tree[node, 0] != -1:
                factor = int(self.tree[node, 0])
                split_value = self.tree[node, 1]
                if points[row, factor] <= split_value:
                    node += int(self.tree[node, 2])
                else:
                    node += int(self.tree[node, 3])
            # Once leaf is arrived at, return split value.
            y[row] = self.tree[node, 1]
        return y


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")