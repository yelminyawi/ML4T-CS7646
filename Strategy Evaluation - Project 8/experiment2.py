import StrategyLearner as sl
import matplotlib.pyplot as plt
import marketsimcode as mks
import datetime as dt


def author():
    return 'nriojas3'


def compare(stock, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=0, verbose=False):
    # impact values used for analysis
    impact1 = 0
    impact2 = 0.005  # specified on project page
    impact3 = 0.01
    impact4 = 0.015
    impact5 = 0.02
    # train and test each learner using the various impact values
    learner1 = sl.StrategyLearner(impact=impact1, commission=commission)
    learner1.add_evidence(stock, sd, ed, sv)
    s1 = learner1.testPolicy(stock, sd, ed, sv)

    learner2 = sl.StrategyLearner(impact=impact2, commission=commission)
    learner2.add_evidence(stock, sd, ed, sv)
    s2 = learner2.testPolicy(stock, sd, ed, sv)

    learner3 = sl.StrategyLearner(impact=impact3, commission=commission)
    learner3.add_evidence(stock, sd, ed, sv)
    s3 = learner3.testPolicy(stock, sd, ed, sv)

    learner4 = sl.StrategyLearner(impact=impact4, commission=commission)
    learner4.add_evidence(stock, sd, ed, sv)
    s4 = learner4.testPolicy(stock, sd, ed, sv)

    learner5 = sl.StrategyLearner(impact=impact5, commission=commission)
    learner5.add_evidence(stock, sd, ed, sv)
    s5 = learner5.testPolicy(stock, sd, ed, sv)

    # generate dataframes from each learner to plot for comparison
    strategy1 = mks.compute_portvals(s1, sd, ed, stock, sv, commission, impact1)
    strategy2 = mks.compute_portvals(s2, sd, ed, stock, sv, commission, impact2)
    strategy3 = mks.compute_portvals(s3, sd, ed, stock, sv, commission, impact3)
    strategy4 = mks.compute_portvals(s4, sd, ed, stock, sv, commission, impact4)
    strategy5 = mks.compute_portvals(s5, sd, ed, stock, sv, commission, impact5)
    # plot the learners
    generate_plot(strategy1, strategy2, strategy3, strategy4, strategy5, impact1, impact2, impact3, impact4, impact5)

    # metric analysis for varying the impact
    if verbose:
        analyze_portfolios(strategy1, strategy2, strategy3, strategy4, strategy5)


def normalize_df(df):
    return df / df.iloc[0, :]


def generate_plot(s1, s2, s3, s4, s5, i1, i2, i3, i4, i5):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(normalize_df(s1), color='red')
    ax.plot(normalize_df(s2), color='green')
    ax.plot(normalize_df(s3), color='blue')
    ax.plot(normalize_df(s4), color='orange')
    ax.plot(normalize_df(s5), color='purple')
    ax.tick_params(axis='x', rotation=20)
    ax.grid()
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend(["impact: " + str(i1), "impact: " + str(i2),
                "impact: " + str(i3), "impact: " + str(i4),
                "impact: " + str(i5)])
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.title("JPM In Sample Strategy Learner Using Various Impacts")
    plot_name = "Experiment2.png"
    plt.savefig(plot_name)


# analyze the cumulative return and standard deviation of the varying impacts
def analyze_portfolios(p1, p2, p3, p4, p5):
    c1 = cumulative_return(normalize_df(p1))
    c2 = cumulative_return(normalize_df(p2))
    c3 = cumulative_return(normalize_df(p3))
    c4 = cumulative_return(normalize_df(p4))
    c5 = cumulative_return(normalize_df(p5))
    print('cumulative return ---------------------------------')
    print("impact 0: ", c1)
    print("impact 0.005: ", c2)
    print("impact 0.01: ", c3)
    print("impact 0.015: ", c4)
    print("impact 0.02: ", c5)

    d1 = daily_returns(normalize_df(p1))
    d2 = daily_returns(normalize_df(p2))
    d3 = daily_returns(normalize_df(p3))
    d4 = daily_returns(normalize_df(p4))
    d5 = daily_returns(normalize_df(p5))
    std1 = d1.std()
    std2 = d2.std()
    std3 = d3.std()
    std4 = d4.std()
    std5 = d5.std()
    print('standard deviation -------------------------------')
    print("impact 0: ", std1)
    print("impact 0.005: ", std2)
    print("impact 0.01: ", std3)
    print("impact 0.015: ", std4)
    print("impact 0.02: ", std5)


def cumulative_return(df):
    last = df.iloc[-1][0]
    first = df.iloc[0][0]
    return (last / first) - 1


def daily_returns(df):
    return df.pct_change(1)


if __name__ == "__main__":
    compare()