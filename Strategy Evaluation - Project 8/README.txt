Files included in this Project (author: nriojas3)
1. testproject.py
2. indicators.py
3. marketsimcode.py
4. RTLearner.py
5. BagLearner.py
6. ManualStrategy.py
7. StrategyLearner.py
8. experiment1.py
9. experiment2.py


------------------------------------------------------------------------------------------------------------------
To run the entire project and generate all plots in a single Python call use (~53s runtime) :
PYTHONPATH=../:. python testproject.py
------------------------------------------------------------------------------------------------------------------
To run ManualStrategy.py alone with cumulative return, standard deviation, and mean printed:
PYTHONPATH=../:. python ManualStrategy.py
------------------------------------------------------------------------------------------------------------------
To run experiment1.py or experiment2.py alone:
PYTHONPATH=../:. python experiment1.py
PYTHONPATH=../:. python experiment2.py
------------------------------------------------------------------------------------------------------------------
Overview:
testproject.py imports ManualStrategy.py, experiment1.py, and experiment2.py and subsequently calls
generate_manual_plots() in ManualStrategy.py, compare() in experiment1.py, and compare() in experiment2.py. These three
calls are what generate the plots used in the report.
------------------------------------------------------------------------------------------------------------------
Description of files:
testproject.py - python file that runs everything needed to generate output plots for report. The start date and end date
                for the in sample and out of sample periods can be specified in this file. In addition, the specific stock,
                start value, commission, commission for experiment 2, and impact for the manual strategy and experiment 1
                can be set here. This is also where the seed is set to allow for replicable results.


indicators.py - python file that calculates five technical indicators for the specified stock: Bollinger Band
                Percentage, Standard, Deviation, Simple Moving Average, Exponential Moving Average, and Momentum using the
                generate_indicators function. This is a dataframe of the value of the specific indicator over time given
                a specific look back period. Not all indicators are used for this project.

marketsim.py - python file that contains the compute_portvals() method which takes a trades dataframe and converts it
                to a dataframe of portfolio value at each date in the index of trades. Any file that generates a plot
                uses this to convert its trades dataframe to a dataframe of portfolio values to be plotted.

ManualStrategy.py - python file that imports the indicators.py and marketsim.py files. It contains the testPolicy() method that
                    tests the manual strategy for trading with bollinger band percentage, simple moving average per price,
                    and momentum to apply a trading strategy to an input stock over a given time period. The generate_plot() method
                    is used to take the trading strategy dataframe and input it into the marketsim.py comput_portvals() method
                    to plot the required comparisons in sample and out of sample. This is also the file that is used to generate
                    the benchmark strategy dataframe to be plotted.

RTLearner.py - python file that contains the class for the Random Tree learner. This is a classification learner, meaning
                it uses the scipy stats.mode function to split values accordingly when building the tree. The learner is
                imported and used in the BagLearner.

BagLearner.py - python file that contains the class for the Bag learner. This is what is used to generate the random forrest
                by calling the RTLearner, and is called by the Strategy Learner.

StrategyLearner.py - python file that imports the indicators.py, RTLearner.py, and BagLearner.py files. It builds a random forrest
                    that is trained using the bollinger band percentage, simple moving average per price, and momentum for a stock
                    using the training data. It also contains a testPolicy() method that generates a dataframe to apply the
                    training and make specific trades which are listed and returned in a trades dataframe.

experiment1.py - python file that imports the ManualStrategy.py, StrategyLearner.py, and marketsim.py files to generate dataframes
                 for the manual strategy, strategy learner, and benchmark strategy to be plotted against each other as portfolio values
                 over a given time period. These portfolio value dataframes are created using the compute_portvals() method in marketsim.py

experiment2.py - python file that imports the StrategyLearner.py, and marketsim.py files to generate plots for the in sample
                performance of the strategy learner for various impact values. Additionally, verbose can be set to True
                to print out metrics for comparing the different impact dataframes.