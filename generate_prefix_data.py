import pandas as pd
import numpy as np
name='newbpi11-f2.csv'
data=pd.read_csv('../{}'.format(name),sep=',')

data['case_length'] = data.groupby('Case ID')["Activity code"].transform(len)

def generate_prefix_data(data, min_length, max_length, gap=1):
    # generate prefix data (each possible prefix becomes a trace)
    # l=[]
    data['case_length'] = data.groupby('Case ID')["Activity code"].transform(len)

    # dt_prefixes = data[data['case_length'] >= min_length*1.25].groupby('Case ID').head(min_length)
    # l.append(len(dt_prefixes))
    # dt_prefixes["prefix_nr"] = 1
    # dt_prefixes["orig_case_id"] = dt_prefixes['Case ID']
    for nr_events in range(min_length, max_length + 1, gap):
        tmp = data[data['case_length'] >= nr_events*1.25].groupby('Case ID').head(nr_events)
        del tmp['case_length']
        # del tmp['Unnamed: 0']
        tmp.to_csv('11-f2/{}.csv'.format(str(nr_events)),index=False)
        # l.append(len(tmp))
        # dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
    # del dt_prefixes['case_length']
    # dt_prefixes.to_csv('{}.csv'.format(name))


generate_prefix_data(data,1,40)
