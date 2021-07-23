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
import RTLearner as rt
import DTLearner as dt
import LinRegLearner as lrl
np.seterr(divide='ignore', invalid='ignore')


class BagLearner(object):
    """
      This is a Bagging Learner. It is implemented correctly.

      :param verbose: If “verbose” is True, your code can print out
      information for debugging.
          If verbose = False your code should not generate ANY output. When we
          test your code, verbose will be False.
      :type verbose: bool
      """

    def __init__(self, learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=20, boost=False, verbose=False):
        """
            Constructor method
            """
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        """
            :return: The GT username of the student
            :rtype: str
            """
        return "cjimenez36"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        data = np.append(data_x, data_y.reshape(data_x.shape[0], 1), axis=1)
        for i in range(self.bags):
            n = data.shape[0]
            n_index = np.random.choice(n, n, replace=True)
            X = data[n_index, :-1]
            y = data[n_index, -1]
            # Calling learner's `add_evidence` function. Not recursive logic here.
            self.learners[i].add_evidence(X, y)

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
        agg = np.empty([self.bags, points.shape[0]])
        for j in range(self.bags):
            agg[j] = self.learners[j].query(points)
        return np.mean(agg, axis=0)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")