""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
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

import math
import sys
import time

import BagLearner as bl
import DTLearner as dt
import InsaneLearner as it
import RTLearner as rt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    if sys.argv[1] == 'Data/Istanbul.csv':
        inf = open(sys.argv[1])
        data = np.genfromtxt(inf, delimiter=',')
        data = data[1:, 1:]
    else:
        inf = open(sys.argv[1])
        data = np.array([list(map(float, s.strip().split(","))) for s in inf.readlines()])

    # set seed
    np.random.seed(903650706)

    # shuffle data
    np.random.shuffle(data)

    # compute how much of the data is training and testing  		  	   		   	 			  		 			 	 	 		 		 	
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		   	 			  		 			 	 	 		 		 	
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # experiment 1
    in_sample = np.array([])
    out_sample = np.array([])
    for i in range(50):
        # train learner
        learner = dt.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)
        # in-sample predictions
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample = np.append(in_sample, [rmse])
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_sample = np.append(out_sample, [rmse])
    results = pd.DataFrame([in_sample, out_sample], index=['in_sample', 'out_sample']).T
    ax = results.plot()
    plt.title('Experiment 1: DTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend(['In Sample', 'Out of Sample'])
    plt.tight_layout()
    plt.savefig('figure1.png')
    plt.close()

    # experiment 2
    in_sample = np.array([])
    out_sample = np.array([])
    for i in range(50):
        # train learner
        learner = bl.BagLearner(kwargs={"leaf_size": i})
        learner.add_evidence(train_x, train_y)
        # in-sample predictions
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample = np.append(in_sample, [rmse])
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_sample = np.append(out_sample, [rmse])
    results = pd.DataFrame([in_sample, out_sample], index=['in_sample', 'out_sample']).T
    ax = results.plot()
    plt.title('Experiment 2: BagLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend(['In Sample', 'Out of Sample'])
    plt.tight_layout()
    plt.savefig('figure2.png')
    plt.close()

    # experiment 3
    # use MAE & of out of sample predictions & time to build trees.

    # MAE
    dt_sample = np.array([])
    rt_sample = np.array([])
    for i in range(50):
        # train learner
        learner = dt.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)
        pred_y = learner.query(test_x)
        mae = abs(test_y - pred_y).sum() / test_y.shape[0]
        dt_sample = np.append(dt_sample, [mae])

        # train learner
        learner = rt.RTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)
        pred_y = learner.query(test_x)
        mae = abs(test_y - pred_y).sum() / test_y.shape[0]
        rt_sample = np.append(rt_sample, [mae])

    results = pd.DataFrame([dt_sample, rt_sample], index=['DTLearner', 'RTLearner']).T
    ax = results.plot()
    plt.title('Experiment 3: DTLearner vs RTLearner: Mean Absolute Error')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error')
    plt.legend(['DTLearner', 'RTLearner'])
    plt.tight_layout()
    plt.savefig('figure3.png')
    plt.close()

    # Time
    dt_time = np.array([])
    rt_time = np.array([])
    for i in range(50):
        # train learner
        start = time.time()
        learner = dt.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)
        end = time.time()
        dt_time = np.append(dt_time, [end - start])

        # train learner
        start = time.time()
        learner = rt.RTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)
        end = time.time()
        rt_time = np.append(rt_time, [end - start])

    results = pd.DataFrame([dt_time, rt_time], index=['DTLearner', 'RTLearner']).T
    ax = results.plot()
    plt.title('Experiment 3: DTLearner vs RTLearner: Time to Build Trees')
    plt.xlabel('Leaf Size')
    plt.ylabel('Time')
    plt.legend(['DTLearner', 'RTLearner'])
    plt.tight_layout()
    plt.savefig('figure4.png')
    plt.close()
