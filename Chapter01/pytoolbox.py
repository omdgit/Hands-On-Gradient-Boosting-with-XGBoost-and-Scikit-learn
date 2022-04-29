"""
Created on Thu Mar 23 10:31:59 2017

@author: eccrod2

First Version: 03.23.2017
Second Version: 03.27.2017
Third Version: 03.31.2017
Fourth Version: 05.18.2017
Fifth Version: 01.19.2019
sixth Version: 04.06.2019
seventh Version: 11.7.2021
"""

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

__version__ = '11.7.2021'


def set_start_time():
    '''Returns the current local date and time'''
    return datetime.datetime.now()


def get_runtime(start):
    '''Calculates the run time.  The only parameter that needs to be passed
    to this method is the start time which can be created with the
    "set_start_time()" method of this module.'''
    start_time = start
    end_time = datetime.datetime.now()
    run_time = relativedelta(end_time, start)
    print(start_time.strftime('Start time: %Y-%m-%d %H:%M:%S %p'))
    print(end_time.strftime('End time:   %Y-%m-%d %H:%M:%S %p'))
    print('Run time: %d days %d hours %d minutes %d seconds'
        % (run_time.days,
           run_time.hours,
           run_time.minutes,
           run_time.seconds))


def freq_tbl(df, var, nb='Yes'):
    '''This is similar to the PROC FREQ in SAS.
    The first parameter is a pandas dataframe and the second parameter
    is the variable of the dataframe for which the frequencies
    are requested.
    example: ptb.freq_tbl(customers, 'risk_ratings')'''
    df = df[var].fillna('Missing')
    x = pd.DataFrame(df.groupby(var).size())
    x.columns = ['count']
    x['cum_count'] = x['count'].cumsum()
    x['percent'] = x['count'] / df.shape[0]
    x['cum_percent'] = x['cum_count'] / df.shape[0]
    x['count'] = x['count'].map('{:,.0f}'.format)
    x['cum_count'] = x['cum_count'].map('{:,.0f}'.format)
    x['percent'] = x['percent'].map('{:.2%}'.format)
    x['cum_percent'] = x['cum_percent'].map('{:.2%}'.format)

    if nb == 'Yes':
        return x.style.set_table_styles(                                       
            [{"selector": "td", "props": [("text-align", "right")]}]
            ).set_precision(2)
    else:
        return x


def stats_tbl(df, var_categorical, var_analysis):
    '''This is similar to the PROC MEANS in SAS.
    The first parameter is a pandas dataframe, the second parameter is the
    categorical variable for which descriptive statistics are requested,
    the third variable is the variable used to calculate the statistics.
    example: ptb.stats_tbl(customers, 'risk_ratings', 'CrrScore')'''
    from collections import OrderedDict

    x = df.groupby(var_categorical)[var_analysis].agg(OrderedDict([
                ('Minimum', 'min'),
                ('Maximum', 'max'),
                ('Mean', 'mean'),
                ('StDev', 'std'),
                ('Median', 'median'),
                ('Count', 'count'),
                ('Pct_1', lambda x: np.percentile(x, 1)),
                ('Pct_10', lambda x: np.percentile(x, 10)),
                ('Pct_25', lambda x: np.percentile(x, 25)),
                ('Pct_50', lambda x: np.percentile(x, 50)),
                ('Pct_75', lambda x: np.percentile(x, 75)),
                ('Pct_90', lambda x: np.percentile(x, 90)),
                ('Pct_99', lambda x: np.percentile(x, 99))
            ]))
    x['Count'] = x['Count'].map('{:,.0f}'.format)
    x['Mean'] = x['Mean'].map('{:.2f}'.format)
    x['StDev'] = x['StDev'].map('{:.2f}'.format)
    return x


def make_pivot_tbl(df, var_row, var_col, var_analysis, func='count_nonzero',
                   miss_value=None):
    '''This is similar to the PROC TABULATE in SAS.        
        Parameters:
            pandas dataframe: source data for pivot table 
            row variable: variable used for vertical distribution
            column variable: variable used for horizontal distribution
            analysis variable: variable used to calculate the statistic
            (i.e. mean, sum, min, max, ...)
            stat: the statistic that is calculated.  By default, it's 'count'.
             Other values: mean, sum, min,...

    '''    
    pd.options.display.float_format = '{:,.0f}'.format
    return pd.pivot_table(df, values=var_analysis, fill_value=miss_value,
                          index=var_row,
                          columns=var_col,
                          aggfunc=eval('np.'+func),
                          margins=True)


def convert_to_categorical_var(var_continuous, bucket_labels, bucket_cutoffs):
    '''Converts a continuous variable to a categorical variable based on
    cut-off points provided as paramaters, similar to a PROC FORMAT in SAS.
    Parameters: pandas dataframe + variable, list of categories/labels,
    list of cut-off points.  The list of cut-off points needs to have one more
    element than the list of categories/labels.
    example: ptb.convert_to_categorical_var(customers['CrrScore'],
                                            ['LR', 'MR', 'HR'],
                                            [0, 2.499, 3, 5])'''
    risk_buckets = bucket_labels
    bins = bucket_cutoffs
    return pd.cut(var_continuous, bins, labels=risk_buckets)

def strata_sample(df, var, sample_size, seed=1234, split=None):
    '''Creates a stratified random sample of a pandas dataframe.
         Parameters:
                 df: pandas dataframe from which a sample is created
                var: list with strata-variable(s)
        sample_size: the sample size; if parameter is <= 1, then it's
        interpreted as a percentage
               seed: seed for random generator (optional)
              split: determines if a second dataframe (i.e. dataframe used as
              parameter less sample), is returned (optional)'''
    np.random.seed(seed)
    sample = pd.DataFrame()
    
    # Convert sample size from a fraction to an integer.
    # Assumption: if the sample size parameter is <= 1,
    #             then the input paramater was a fraction and needs 
    #             to be converted to the desired sample size as an integer.
    if sample_size <= 1:
        sample_size = int(sample_size * df.shape[0])
 
    # Determine the sample size for each stratum.
    x = pd.DataFrame(df.groupby(var).size())
    x.columns = ['count']
    x['percent'] = x['count'] / df.shape[0]
    x['sample_size'] = [int(sample_size * j) for j in x['percent']]
    ls = list(var)
    
    # Build a random sample for each stratum  
    # and combine the random samples to a stratified random sample.
    for i in x.index:
        if len(ls) == 1:
            sub = df[df[ls[0]] == i].copy()
        else:
            # If multiple variables make up the stratum,
            # build a query string to create the dataframe "sub"
            # which comprises all records of a stratum.
            query_str = '"' + str(list(zip(ls, list(i)))).replace("), ('", " & ").replace("', ", " == ").replace("[(", "").replace(")]", '"')[1:]
            sub = df.query(eval(query_str)).copy()
            
        # Build an index with random numbers which will be used
        # to draw a random sample for a stratum.
        ix = np.random.choice(range(0, sub.shape[0]),
                              int(x.loc[i].sample_size), replace=False)
        ix.sort()
        sample = pd.concat([sample, sub.iloc[ix]])
     
    if split == None:
        return sample
    else:
        ix = df.index.isin(sample.index)        
        return sample, df[~ix]

    
def make_sql_conn(server, database):
    import pyodbc
    return pyodbc.connect('DRIVER={SQL Server}; \
                          DATABASE='+database +'; \
                          SERVER='+server+'; \
                          Trusted_Connection=yes')


def get_loaded_packages(global_values):
    print('\n'.join(f'{m.__name__}=={m.__version__}' for m in global_values if getattr(m, '__version__', None)))