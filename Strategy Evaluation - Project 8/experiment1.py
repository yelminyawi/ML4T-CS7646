import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt
import marketsimcode as mks
import datetime as dt


def author():
    return 'nriojas3'


def compare(stock, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005):
    # evaluate manual strategy and generate dataframe for plotting
    m = ms.testPolicy(stock, sd, ed, sv)
    manual = mks.compute_portvals(m, sd, ed, stock, sv, commission, impact)
    # evaluate benchmark strategy and generate dataframe for plotting
    benchmark = ms.get_benchmark(stock, sd, ed)
    # evaluate strategy learner and generate dataframe for plotting
    learner = sl.StrategyLearner(impact=impact, commission=commission)
    learner.add_evidence(stock, sd, ed, sv)
    s = learner.testPolicy(stock, sd, ed, sv)
    strategy = mks.compute_portvals(s, sd, ed, stock, sv, commission, impact)
    # create the plots
    generate_plot(manual, benchmark, strategy)


def normalize_df(df):
    return df / df.iloc[0, :]


def generate_plot(manual, benchmark, strategy):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(normalize_df(manual), color='red')
    ax.plot(normalize_df(benchmark), color='green')
    ax.plot(normalize_df(strategy), color='blue')
    ax.tick_params(axis='x', rotation=20)
    ax.grid()
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend(['Manual', "Benchmark", "Strategy"])
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.title("JPM In Sample Manual Strategy vs Benchmark Strategy vs Strategy Learner")
    plot_name = "Experiment1.png"
    plt.savefig(plot_name)


if __name__ == "__main__":
    compare()
