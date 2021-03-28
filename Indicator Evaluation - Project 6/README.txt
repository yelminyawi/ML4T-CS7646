Files included in this Project
1. testproject.py
2. indicators.py
3. TheoreticallyOptimalStrategy.py

------------------------------------------------------------------------------------------------------------------
To run the entire project and generate all plots in a single Python call use:
PYTHONPATH=../:. python testproject.py
------------------------------------------------------------------------------------------------------------------
To run indicators.py or TheoreticallyOptimalStrategy.py alone:
PYTHONPATH=../:. python indicators.py
PYTHONPATH=../:. python TheoreticallyOptimalStrategy.py
------------------------------------------------------------------------------------------------------------------
Overview:
testproject.py generates all plots by importing indicators.py and TheoreticallyOptimalStrategy.py and running
their essential methods that generate plots specific to them. The overarching method indicators.py uses to make all
essentials calls that generates the plots is generate_indicators. For TheoreticallyOptimalStrategy.py, this method is
testpolicy. Within testpolicy, a call is made to the compute_portvals method which is from the previous project's marketsim.py file
------------------------------------------------------------------------------------------------------------------
Description of files:
testproject.py - python file that runs everything needed to generate outputs for report. This
                file imports indicators.py and TheoreticallyOptimalStrategy.py, runs their functions
                generate_indicators and testPolicy, respectively, with the required input parameters
                which generates the plots each individual file is able to produce


indicators.py - python file that calculates five technical indicators for the specified stock: Bollinger Band
                Percentage, Standard, Deviation, Simple Moving Average, Exponential Moving Average, and Momentum using the
                generate_indicators function. Within this function separate function calls are performed
                to generate and save plots used to illustrate each indicator over time.

TheoreticallyOptimalStrategy.py - python file that generates a dataframe of the specified stock that determines
                                  the orders to be made each day during the specified period of time for optimal trading. These orders
                                  are limited to 0, +1000, -1000, +2000, -2000, due to the holdings limitations of the
                                  specified stock specified as only long 1000, short 1000, or 0 (do nothing). Using this
                                  dataframe, a separate dataframe is generated using compute_portvals method to generate the value
                                  of the portfolio each day. A benchmark dataframe is generated where the only order given
                                  is +1000 on the first day of trading. A plot is generated comparing the benchmark and
                                  optimal portfolio. The testpolicy method generates the dataframe, calls compute_portvals
                                  method, and makes a call to a separate function to generate and save the plot.

                                  An additional helper function, analyze_portfolios can be used to compute the performance criteria

                                  Note: this file has absorbed the functionality from the marketsim.py file from the prior
                                  project. This is the functionality utilized.

               compute_portvals function accepts a dataframe of orders for a stock,
               start and end dates, a specified stock, a starting cash value, commission, and impact values. Using
               these parameters the portfolio's value at each day during the range is computed, incorporating the
               stock orders and starting values into the value.
