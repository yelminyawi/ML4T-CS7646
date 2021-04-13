import ManualStrategy as ms
import StrategyLearner as sl
import datetime as dt


def author():
    return 'nriojas3'



# # Manual Strategy
# df_trades = ms.testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)
#
#
# # Strategy Learner
# learner = sl.StrategyLearner(verbose = False, impact = 0.0, commission=0.0) # constructor
# learner.add_evidence(symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
# df_trades = learner.testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # testing phase

ms.run()