import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = []
    def author(self):
        return "cjimenez36"
    def add_evidence(self, data_x, data_y):
        for i in range(20):
            learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)
            learner.add_evidence(data_x, data_y)
            self.learners.append(learner)
    def query(self, points):
        final = np.empty((20, points.shape[0]))
        for i in range(20):
            final[i] = self.learners[i].query(points)
        return np.mean(final, axis=0)