import itertools
import pandas as pd
from qp_ode_simulator.data_growth import nested_events, fresh_split, STRATA


def test_balanced_nested_event_splits():
    rows=[]
    for keys in itertools.product(['circular','elliptical'],['ballistic','diffusive'],
                                   ['inside','edge','outside'],[0,1,2]):
        for i in range(4):
            rows.append(dict(zip(STRATA,keys),event_uid=f'{keys}_{i}'))
    labels=pd.DataFrame(rows)
    subsets=nested_events(labels,[30,60,120])
    assert len(subsets['120'])==120
    assert set(subsets['30']) < set(subsets['60']) < set(subsets['120'])
    selected=labels[labels.event_uid.isin(subsets['120'])]
    assert not ((selected.geometry=='elliptical')&(selected.strength_band==2)).any()
    val,test=fresh_split(labels)
    assert len(val)==len(test)==72
    assert not set(val)&set(test)
