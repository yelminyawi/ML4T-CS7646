import TheoreticallyOptimalStrategy as tos
import indicators as ind
import datetime as dt


def author():
    return 'nriojas3'


# Parameters to test the project
start_date = dt.datetime(2008, 1, 1)
end_date = dt.datetime(2009, 12, 31)
start_value = 100000
symbol = 'JPM'
lookback = 14
# Make calls to the two files which generate plots and dataframe for trades
ind.generate_indicators(symbol, lookback, start_date, end_date)
# generate the trades dataframe required
df_trades = tos.testPolicy(symbol, start_date, end_date, start_value)
