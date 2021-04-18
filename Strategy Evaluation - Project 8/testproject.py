import ManualStrategy as ms
import experiment1 as e1
import experiment2 as e2
import datetime as dt
import numpy as np


def author():
    return 'nriojas3'


# seed for the RT Learner
np.random.seed(1234)

stock = 'JPM'

sd_in = dt.datetime(2008, 1, 1)
ed_in = dt.datetime(2009, 12, 31)

sd_out = dt.datetime(2010, 1, 1)
ed_out = dt.datetime(2011, 12, 31)

sv = 100000
commission = 9.95
commission_ex_2 = 0
impact = 0.005

# evaluate manual strategy in and out of sample with vertical lines
# number of plots: 2
ms.generate_manual_plots()

# Exp1: compare manual, benchmark, and strategy learners
# number of plots: 1
e1.compare(stock, sd_in, ed_in, sv, commission, impact)

# Exp2: show the effect of changing impact values on return for the strategy learner
# number of plots: 1
e2.compare(stock, sd_in, ed_in, sv, commission)


