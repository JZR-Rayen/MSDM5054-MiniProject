"""Two deterministic LGB-only feature families; original 39 are immutable."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baseline'))
import numpy as np
import pandas as pd
from baseline_features_v2 import monthly_features

TREND_PAIRS = {
 'OPT_T_LATE_RECENT_MINUS_ALL':('R_RECENT12_LATE_RATE','R_LATE_RATE'),
 'OPT_T_SHORTFALL_RECENT_MINUS_ALL':('R_RECENT12_SHORTFALL_MEAN','R_ASOF_SHORTFALL_MEAN'),
 'OPT_T_CC_UTIL_RECENT_MINUS_ALL':('H_CC_UTILIZATION_RECENT6','H_CC_UTILIZATION_MEAN'),
 'OPT_T_BB_RECENT_MINUS_ALL':('BB_RECENT6_OVERDUE_LOAN_RATE_MEAN','BB_LOAN_OVERDUE_RATE_MEAN')}
STATUS_FIELDS = {'OVERDUE_LOAN_MEAN':'BB_LOAN_OVERDUE_RATE_MEAN',
                 'SEVERE_LOAN_MEAN':'BB_LOAN_SEVERE_RATE_MEAN',
                 'RECENT6_OVERDUE_LOAN_MEAN':'BB_RECENT6_OVERDUE_LOAN_RATE_MEAN'}

def trend_features(frame):
    # Recent and full history overlap; these are level differences, not time slopes.
    return pd.DataFrame({name:frame[recent]-frame[all_history]
                         for name,(recent,all_history) in TREND_PAIRS.items()},index=frame.index)

def status_features(months,bureau):
    eligible=bureau.loc[bureau.DAYS_CREDIT.le(0)&~bureau.DAYS_CREDIT_UPDATE.gt(0)].copy()
    assert eligible.SK_ID_BUREAU.is_unique
    result=pd.DataFrame(index=pd.Index(eligible.SK_ID_CURR.unique(),name='SK_ID_CURR'))
    audit={}
    for state in ['Active','Closed']:
        mapping=eligible.loc[eligible.CREDIT_ACTIVE.eq(state),['SK_ID_BUREAU','SK_ID_CURR']]
        # Reuse exact V2 loan-month de-duplication, C/X denominator and equal-loan aggregation.
        part,info=monthly_features(months,mapping)
        part=part.set_index('SK_ID_CURR');audit[state]=info
        for suffix,source in STATUS_FIELDS.items():
            result[f'OPT_S_{state.upper()}_{suffix}']=part[source]
    return result.reset_index(),audit

def build_extension(root,train,test):
    raw=Path(root)/'home-credit-default-risk'
    bureau=pd.read_csv(raw/'bureau.csv',usecols=['SK_ID_CURR','SK_ID_BUREAU','CREDIT_ACTIVE','DAYS_CREDIT','DAYS_CREDIT_UPDATE'])
    months=pd.read_csv(raw/'bureau_balance.csv',dtype={'SK_ID_BUREAU':'int32','MONTHS_BALANCE':'int16','STATUS':'category'})
    states,audit=status_features(months,bureau)
    result=[]
    for frame in [train,test]:
        extra=frame[['SK_ID_CURR']].copy()
        extra=pd.concat([extra,trend_features(frame)],axis=1)
        extra=extra.merge(states,on='SK_ID_CURR',how='left',validate='one_to_one',sort=False)
        assert extra.SK_ID_CURR.equals(frame.SK_ID_CURR)
        assert not np.isinf(extra.drop(columns='SK_ID_CURR').to_numpy()).any()
        result.append(extra)
    return result,audit
