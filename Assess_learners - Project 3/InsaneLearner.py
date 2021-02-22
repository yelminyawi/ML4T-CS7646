import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner:
    def __init__(self, verbose=False):
        self.verbose, self.L, self.y = verbose, [], None
        for i in range(20):
            self.L.append(bl.BagLearner(lrl.LinRegLearner, kwargs={'verbose': verbose} , bags=20, boost=False, verbose=self.verbose))
    def author(self) -> str:
        return 'nriojas3'
    def add_evidence(self, x, y) -> None:
        for learners in self.L:
            learners.add_evidence(x, y)
    def query(self, x) -> np.ndarray:
        for learners in self.L:
            y_i = learners.query(x.T)
            if self.y is None: self.y = y_i
            else: self.y = np.column_stack((self.y, y_i))
        return np.mean(np.atleast_2d(self.y), axis=1)