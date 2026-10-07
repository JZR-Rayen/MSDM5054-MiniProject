"""Bounded, applicant-local histories and training-only nonlinear representations."""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import SplineTransformer, StandardScaler
from baseline_data import Preprocessor, save_json, sha256

SPLINE_COLUMNS=['YEARS_BIRTH','YEARS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                'AMT_CREDIT','AMT_INCOME_TOTAL','ANNUITY_INCOME_RATIO']
FAMILY_ORDER=['R','H','B','N']
PREFIX={'R':'R_','H':'H_','B':'BB_'}
PERIOD=['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER']

def ratio(a,b):
    return a.div(b.where(b>0)).replace([np.inf,-np.inf],np.nan)

def repayment_features(df):
    """One canonical observable schedule per loan/period; sum split payments exactly."""
    audit={'input_rows':len(df),'future_payment_rows':int(df.DAYS_ENTRY_PAYMENT.gt(0).sum()),
           'future_due_rows':int(df.DAYS_INSTALMENT.gt(0).sum())}
    d=df.loc[df.DAYS_INSTALMENT.le(0)&df.NUM_INSTALMENT_VERSION.notna()].copy()
    highest=d.groupby(PERIOD,sort=False).NUM_INSTALMENT_VERSION.transform('max')
    audit['older_version_rows']=int(d.NUM_INSTALMENT_VERSION.ne(highest).sum())
    d=d.loc[d.NUM_INSTALMENT_VERSION.eq(highest)].copy();del highest
    observed=d.DAYS_ENTRY_PAYMENT.le(0)
    d['_bad']=(d.DAYS_ENTRY_PAYMENT.isna()|d.AMT_PAYMENT.isna()|d.AMT_INSTALMENT.isna()|
               (observed&d.AMT_PAYMENT.lt(0))|d.AMT_INSTALMENT.le(0))
    d['_paid']=d.AMT_PAYMENT.where(observed,0).fillna(0)
    d['_ontime']=d['_paid'].where(d.DAYS_ENTRY_PAYMENT.le(d.DAYS_INSTALMENT),0)
    d['_payment_row']=(observed&d.AMT_PAYMENT.gt(0)).astype('int8')
    g=d.groupby(PERIOD,sort=False)
    events=g.agg(due_min=('DAYS_INSTALMENT','min'),due=('DAYS_INSTALMENT','max'),
        plan_min=('AMT_INSTALMENT','min'),plan=('AMT_INSTALMENT','max'),
        bad=('_bad','max'),paid=('_paid','sum'),ontime=('_ontime','sum'),payments=('_payment_row','sum'))
    ambiguous=events.bad|events.due_min.ne(events.due)|~np.isclose(events.plan_min,events.plan,rtol=1e-10,atol=1e-6)
    audit['canonical_events']=len(events);audit['ambiguous_or_invalid_events']=int(ambiguous.sum())
    events=events.loc[~ambiguous].copy()
    # Reject an ambiguous event as a whole, not just its inconvenient payment row.
    keys=pd.MultiIndex.from_frame(d[PERIOD]);d=d.loc[keys.isin(events.index)].copy();del keys,g
    d=d.sort_values(PERIOD+['DAYS_ENTRY_PAYMENT'],kind='stable')
    cumulative=d.groupby(PERIOD,sort=False)['_paid'].cumsum()
    crossed=d.DAYS_ENTRY_PAYMENT.le(0)&cumulative.ge(d.AMT_INSTALMENT-1e-6)
    completion=d.loc[crossed].groupby(PERIOD,sort=False).DAYS_ENTRY_PAYMENT.min()
    events['completion']=completion;del d,cumulative,completion
    events['late']=(events.ontime.lt(events.plan-1e-6)).astype(float)
    events['underpaid']=(events.paid.lt(events.plan-1e-6)).astype(float)
    events['on_short']=((events.plan-events.ontime).clip(lower=0)/events.plan).clip(upper=1)
    events['asof_short']=((events.plan-events.paid).clip(lower=0)/events.plan).clip(upper=1)
    events['delay']=(events.completion-events.due).clip(lower=0)
    events['overdue_asof']=events['delay'].where(events.underpaid.eq(0),(-events.due).clip(lower=0))
    events['split']=events.payments.gt(1).astype(float)
    e=events.reset_index();g=e.groupby('SK_ID_CURR',sort=True)
    a=g.agg(R_DUE_EVENTS_COUNT=('plan','size'),R_SPLIT_PAYMENT_RATE=('split','mean'),
      R_LATE_RATE=('late','mean'),R_ON_TIME_SHORTFALL_MEAN=('on_short','mean'),
      R_UNDERPAID_RATE_ASOF=('underpaid','mean'),R_ASOF_SHORTFALL_MEAN=('asof_short','mean'),
      R_COMPLETION_DELAY_MEAN=('delay','mean'),R_COMPLETION_DELAY_MAX=('delay','max'),
      R_OVERDUE_DAYS_ASOF_MEAN=('overdue_asof','mean'))
    recent=e.loc[e.due.ge(-365)].groupby('SK_ID_CURR')
    a['R_RECENT12_LATE_RATE']=recent.late.mean()
    a['R_RECENT12_SHORTFALL_MEAN']=recent.asof_short.mean()
    a['R_DAYS_SINCE_LAST_LATE_DUE']=-e.loc[e.late.eq(1)].groupby('SK_ID_CURR').due.max()
    audit['eligible_events']=len(e);audit['split_events']=int(e.split.sum())
    audit['complete_events']=int(e.completion.notna().sum())
    audit['observed_asof_incomplete_events']=int(e.underpaid.sum())
    return a.reset_index(),audit

def bureau_structure(df):
    d=df.loc[~df.DAYS_CREDIT_UPDATE.gt(0)&df.DAYS_CREDIT.le(0)].copy()
    d['_active']=d.CREDIT_ACTIVE.eq('Active').astype(float)
    d['_closed']=d.CREDIT_ACTIVE.eq('Closed').astype(float)
    g=d.groupby('SK_ID_CURR');a=pd.DataFrame(index=g.size().index)
    a['H_ACTIVE_RATIO']=g['_active'].mean();a['H_CLOSED_RATIO']=g['_closed'].mean()
    for field,name,active in [('AMT_CREDIT_SUM_DEBT','H_ACTIVE_NET_DEBT_CREDIT_RATIO_CUR1',True),
                             ('AMT_CREDIT_SUM_OVERDUE','H_CURRENT_OVERDUE_CREDIT_RATIO_CUR1',False)]:
        ok=d.CREDIT_CURRENCY.eq('currency 1')&d.AMT_CREDIT_SUM.gt(0)&d[field].notna()
        if active:ok&=d.CREDIT_ACTIVE.eq('Active')
        pairs=d.loc[ok].groupby('SK_ID_CURR')
        a[name]=ratio(pairs[field].sum(),pairs.AMT_CREDIT_SUM.sum())
    recent=d.loc[d.DAYS_CREDIT.ge(-365)].groupby('SK_ID_CURR')
    a['H_NEW_CREDIT_COUNT_12M']=recent.size().reindex(a.index,fill_value=0)
    a['H_NEW_CREDIT_ACTIVE_RATE_12M']=recent['_active'].mean()
    a['H_DAYS_SINCE_LATEST_OVERDUE_UPDATE']=-d.loc[d.CREDIT_DAY_OVERDUE.gt(0)].groupby('SK_ID_CURR').DAYS_CREDIT_UPDATE.max()
    return a.reset_index()

def history_features(raw):
    raw=Path(raw);audit={}
    cols=['SK_ID_CURR','SK_ID_BUREAU','CREDIT_ACTIVE','CREDIT_CURRENCY','DAYS_CREDIT',
          'DAYS_CREDIT_UPDATE','CREDIT_DAY_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE']
    d=pd.read_csv(raw/'bureau.csv',usecols=cols)
    audit['bureau_future_updates_excluded']=int(d.DAYS_CREDIT_UPDATE.gt(0).sum())
    eligible=d.loc[~d.DAYS_CREDIT_UPDATE.gt(0)&d.DAYS_CREDIT.le(0),['SK_ID_BUREAU','SK_ID_CURR']].copy()
    assert eligible.SK_ID_BUREAU.is_unique
    a=bureau_structure(d).set_index('SK_ID_CURR');del d
    d=pd.read_csv(raw/'previous_application.csv',usecols=['SK_ID_CURR','DAYS_DECISION','NAME_CONTRACT_STATUS'])
    audit['previous_future_rows_excluded']=int(d.DAYS_DECISION.gt(0).sum())
    d=d.loc[d.DAYS_DECISION.le(0)].copy();d['_refused']=d.NAME_CONTRACT_STATUS.eq('Refused').astype(float)
    g=d.groupby('SK_ID_CURR');extra=pd.DataFrame({'H_PREV_REFUSAL_RATE':g['_refused'].mean()})
    recent=d.loc[d.DAYS_DECISION.ge(-365)].groupby('SK_ID_CURR')
    extra['H_PREV_COUNT_12M']=recent.size().reindex(extra.index,fill_value=0)
    extra['H_PREV_REFUSAL_RATE_12M']=recent['_refused'].mean();a=a.join(extra,how='outer');del d,g,extra
    for fn,code in [('credit_card_balance.csv','CC'),('POS_CASH_balance.csv','POS')]:
        cols=['SK_ID_CURR','MONTHS_BALANCE','SK_DPD']
        if code=='CC':cols+=['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']
        d=pd.read_csv(raw/fn,usecols=cols)
        audit[code+'_future_month_rows_excluded']=int(d.MONTHS_BALANCE.gt(0).sum())
        d=d.loc[d.MONTHS_BALANCE.le(0)].copy();d['_overdue']=d.SK_DPD.gt(0).where(d.SK_DPD.notna()).astype(float)
        if code=='CC':d['_util']=ratio(d.AMT_BALANCE,d.AMT_CREDIT_LIMIT_ACTUAL)
        g=d.groupby('SK_ID_CURR');extra=pd.DataFrame(index=g.size().index)
        if code=='CC':extra['H_CC_UTILIZATION_MEAN']=g['_util'].mean()
        extra[f'H_{code}_MONTHS_SINCE_LAST_OVERDUE']=-d.loc[d.SK_DPD.gt(0)].groupby('SK_ID_CURR').MONTHS_BALANCE.max()
        recent=d.loc[d.MONTHS_BALANCE.ge(-6)].groupby('SK_ID_CURR')
        extra[f'H_{code}_OVERDUE_RATE_RECENT6']=recent['_overdue'].mean()
        extra[f'H_{code}_DPD_MEAN_RECENT6']=recent.SK_DPD.mean()
        if code=='CC':extra['H_CC_UTILIZATION_RECENT6']=recent['_util'].mean()
        a=a.join(extra,how='outer');del d,g,extra,recent
    return a.reset_index(),eligible,audit

def monthly_features(df,mapping):
    audit={'input_month_rows':len(df),'future_month_rows':int(df.MONTHS_BALANCE.gt(0).sum())}
    d=df.loc[df.MONTHS_BALANCE.le(0)].copy()
    mapped=d.SK_ID_BUREAU.isin(mapping.SK_ID_BUREAU)
    audit['unmapped_month_rows']=int((~mapped).sum());d=d.loc[mapped].copy()
    unknown=set(d.STATUS.dropna().astype(str).unique())-set('012345CX')
    if unknown:raise ValueError('Undefined bureau_balance status codes: '+str(unknown))
    d['_code']=d.STATUS.astype('object').map({'C':-1,'X':-2,**{str(i):i for i in range(6)}}).fillna(-2).astype('int8')
    duplicates=d.duplicated(['SK_ID_BUREAU','MONTHS_BALANCE'],keep=False)
    audit['duplicate_loan_month_rows']=int(duplicates.sum())
    if duplicates.any():
        # No transaction key: one loan/month, largest known severity wins deterministic ties.
        d=d.groupby(['SK_ID_BUREAU','MONTHS_BALANCE'],sort=False,as_index=False)['_code'].max()
    d['_known']=d['_code'].ge(0).astype('int8');d['_overdue']=d['_code'].gt(0).astype('int8')
    d['_severe']=d['_code'].ge(2).astype('int8');d['_severity']=d['_code'].where(d['_known'].eq(1))
    g=d.groupby('SK_ID_BUREAU',sort=False)
    loan=g.agg(known=('_known','sum'),overdue=('_overdue','sum'),severe=('_severe','sum'),
               severity=('_severity','max'),latest_month=('MONTHS_BALANCE','max'))
    loan['rate']=ratio(loan.overdue,loan.known);loan['severe_rate']=ratio(loan.severe,loan.known)
    recent=d.loc[d.MONTHS_BALANCE.ge(-6)].groupby('SK_ID_BUREAU')
    loan['recent_rate']=ratio(recent['_overdue'].sum(),recent['_known'].sum())
    latest=d.loc[d.MONTHS_BALANCE.eq(d.SK_ID_BUREAU.map(loan.latest_month))].groupby('SK_ID_BUREAU')['_code'].max()
    loan['latest_overdue']=latest.gt(0).astype(float).where(latest.ne(-2))
    loan['latest_closed']=latest.eq(-1).astype(float).where(latest.ne(-2))
    loan=loan.reset_index().merge(mapping,on='SK_ID_BUREAU',how='left',validate='one_to_one')
    a=loan.groupby('SK_ID_CURR').agg(BB_HISTORY_LOAN_COUNT=('SK_ID_BUREAU','size'),
      BB_OBSERVED_MONTHS_LOAN_MEAN=('known','mean'),BB_LOAN_OVERDUE_RATE_MEAN=('rate','mean'),
      BB_LOAN_OVERDUE_RATE_MAX=('rate','max'),BB_LOAN_SEVERE_RATE_MEAN=('severe_rate','mean'),
      BB_MAX_SEVERITY=('severity','max'),BB_LATEST_OVERDUE_LOAN_RATE=('latest_overdue','mean'),
      BB_LATEST_CLOSED_LOAN_RATE=('latest_closed','mean'),BB_RECENT6_OVERDUE_LOAN_RATE_MEAN=('recent_rate','mean'))
    audit['history_loans']=len(loan);audit['history_applicants']=len(a)
    return a.reset_index(),audit

def build_histories(root,out):
    root=Path(root);out=Path(out);started=time.perf_counter();raw=root/'home-credit-default-risk'
    d=pd.read_csv(raw/'installments_payments.csv',dtype={'SK_ID_CURR':'int32','SK_ID_PREV':'int32','NUM_INSTALMENT_NUMBER':'int32'})
    r,ra=repayment_features(d);del d
    print('R features built:',ra,flush=True)
    h,mapping,ha=history_features(raw);print('H features built:',ha,flush=True)
    d=pd.read_csv(raw/'bureau_balance.csv',dtype={'SK_ID_BUREAU':'int32','MONTHS_BALANCE':'int16','STATUS':'category'})
    b,ba=monthly_features(d,mapping);del d,mapping
    result=r.set_index('SK_ID_CURR').join(h.set_index('SK_ID_CURR'),how='outer').join(b.set_index('SK_ID_CURR'),how='outer').reset_index()
    assert result.SK_ID_CURR.is_unique and not np.isinf(result.drop(columns='SK_ID_CURR').to_numpy()).any()
    defs={'R':r.columns.drop('SK_ID_CURR').tolist(),'H':h.columns.drop('SK_ID_CURR').tolist(),'B':b.columns.drop('SK_ID_CURR').tolist(),
          'N':{'splines':SPLINE_COLUMNS,'knots':5,'degree':3,'extrapolation':'constant','interactions':['EXT_SOURCE_2*EXT_SOURCE_3','YEARS_BIRTH*EXT_SOURCE_3']}}
    audit={'repayments':ra,'history_structure':ha,'bureau_balance':ba,'seconds':time.perf_counter()-started,
      'raw_features':len(result.columns)-1,'train_test_same_definition':True,'target_accessed':False}
    result.to_pickle(out/'results/history_features_v2.pkl')
    save_json(out/'results/feature_definitions_v2.json',defs);save_json(out/'results/feature_audit_v2.json',audit)
    print('B features built:',ba,flush=True)
    return result,audit

def attach_histories(train,test,histories):
    a=train.merge(histories,on='SK_ID_CURR',how='left',validate='one_to_one',sort=False)
    b=test.merge(histories,on='SK_ID_CURR',how='left',validate='one_to_one',sort=False)
    assert a.SK_ID_CURR.equals(train.SK_ID_CURR) and b.SK_ID_CURR.equals(test.SK_ID_CURR)
    assert a.TARGET.equals(train.TARGET)
    return a,b

def prepare_features(train,test,root,out,reuse=False):
    from cache_runtime import current_runtime
    out=Path(out)
    provenance={'inputs':json.loads((out/'results/input_fingerprints.json').read_text()),
      'feature_code':sha256(Path(__file__)),'runtime':current_runtime()}
    key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    path=out/'results/history_features_v2.pkl';meta=out/'results/history_feature_cache_v2.json'
    valid=reuse and path.exists() and meta.exists() and json.loads(meta.read_text())['key']==key
    if valid:
        assert sha256(path)==json.loads(meta.read_text())['table_sha256']
        h=pd.read_pickle(path);audit=json.loads((out/'results/feature_audit_v2.json').read_text())
        print('Validated matching v2 applicant-history cache',flush=True)
    else:
        h,audit=build_histories(root,out)
        save_json(meta,{'key':key,'provenance':provenance,'table_sha256':sha256(path)})
    a,b=attach_histories(train,test,h)
    a.to_pickle(out/'results/train_features_v2.pkl');b.to_pickle(out/'results/official_test_features.pkl')
    return a,b,audit

class ExtensionPreprocessor:
    def __init__(self,families):self.families=sorted(families)
    def selected(self,x):
        return x[[c for c in x if not any(c.startswith(p) and family not in self.families for family,p in PREFIX.items())]]
    def nonlinear(self,x):
        n=self.base_._filled(x)
        v=n[SPLINE_COLUMNS].to_numpy()
        basis=self.spline_.transform(v)
        interactions=np.column_stack([n.EXT_SOURCE_2*n.EXT_SOURCE_3,n.YEARS_BIRTH*n.EXT_SOURCE_3])
        return np.column_stack([basis,interactions])
    def fit(self,x):
        x=self.selected(x);self.base_=Preprocessor().fit(x);self.names_=self.base_.get_feature_names_out()
        if 'N' in self.families:
            n=self.base_._filled(x)
            self.spline_=SplineTransformer(n_knots=5,degree=3,knots='quantile',include_bias=False,extrapolation='constant').fit(n[SPLINE_COLUMNS].to_numpy())
            self.extra_scaler_=StandardScaler().fit(self.nonlinear(x))
            names=self.spline_.get_feature_names_out(SPLINE_COLUMNS).tolist()+['INTERACT_EXT2_EXT3','INTERACT_AGE_EXT3']
            self.names_=np.r_[self.names_,names]
        self.columns_=self.base_.columns_
        return self
    def transform(self,x):
        x=self.selected(x);z=self.base_.transform(x)
        if 'N' in self.families:z=sparse.hstack([z,sparse.csr_matrix(self.extra_scaler_.transform(self.nonlinear(x)))],format='csr')
        assert np.isfinite(z.data).all()
        return z
    def fit_transform(self,x):return self.fit(x).transform(x)
    def get_feature_names_out(self):return self.names_
    def unknown_counts(self,x):return self.base_.unknown_counts(self.selected(x))
    def metadata(self):
        a=self.base_.metadata();a['feature_count']=len(self.names_);a['families']=self.families
        if 'N' in self.families:
            a['spline_knots']=[b.t.tolist() for b in self.spline_.bsplines_]
            a['nonlinear_extra_features']=len(self.names_)-len(self.base_.names_)
        return a
