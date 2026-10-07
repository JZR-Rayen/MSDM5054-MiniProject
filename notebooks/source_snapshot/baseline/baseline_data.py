"""Reconstruct the agreed input, audit it, and fit preprocessing only on training rows."""
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 5054
LOG_COLUMNS = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
               'INCOME_CREDIT_RATIO','ANNUITY_INCOME_RATIO']
AGGREGATIONS = {
 'bureau.csv': {'BUREAU_COUNT':('SK_ID_BUREAU','count'),
  'BUREAU_ACTIVE':('_active','sum'), 'BUREAU_CLOSED':('_closed','sum'),
  'BUREAU_AVG_CREDIT':('AMT_CREDIT_SUM','mean'), 'BUREAU_MAX_CREDIT':('AMT_CREDIT_SUM','max'),
  'BUREAU_AVG_DEBT':('AMT_CREDIT_SUM_DEBT','mean'), 'BUREAU_MAX_OVERDUE':('AMT_CREDIT_MAX_OVERDUE','max')},
 'previous_application.csv': {'PREV_COUNT':('SK_ID_PREV','count'),
  'PREV_APPROVED':('_approved','sum'), 'PREV_REFUSED':('_refused','sum'),
  'PREV_AVG_AMT':('AMT_APPLICATION','mean'), 'PREV_MAX_AMT':('AMT_APPLICATION','max'),
  'PREV_AVG_ANNUITY':('AMT_ANNUITY','mean'), 'PREV_AVG_DAYS_DECISION':('DAYS_DECISION','mean')},
 'installments_payments.csv': {'INSTALL_COUNT':('SK_ID_PREV','count'),
  'INSTALL_AVG_PAYMENT':('AMT_PAYMENT','mean'), 'INSTALL_MAX_PAYMENT':('AMT_PAYMENT','max'),
  'INSTALL_AVG_DAYS_LATE':('DAYS_ENTRY_PAYMENT','mean')},
 'POS_CASH_balance.csv': {'POS_COUNT':('SK_ID_PREV','count'), 'POS_AVG_DPD':('SK_DPD','mean'),
  'POS_MAX_DPD':('SK_DPD','max'), 'POS_AVG_DPD_DEF':('SK_DPD_DEF','mean')},
 'credit_card_balance.csv': {'CC_COUNT':('SK_ID_PREV','count'),
  'CC_AVG_BALANCE':('AMT_BALANCE','mean'), 'CC_MAX_BALANCE':('AMT_BALANCE','max'),
  'CC_AVG_LIMIT':('AMT_CREDIT_LIMIT_ACTUAL','mean'), 'CC_AVG_DPD':('SK_DPD','mean')}
}
TIME_COLUMNS = {'bureau.csv':['DAYS_CREDIT','DAYS_CREDIT_UPDATE'],
 'previous_application.csv':['DAYS_DECISION'], 'installments_payments.csv':['DAYS_ENTRY_PAYMENT','DAYS_INSTALMENT'],
 'POS_CASH_balance.csv':['MONTHS_BALANCE'], 'credit_card_balance.csv':['MONTHS_BALANCE']}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')

def aggregate(df, filename):
    df=df.copy()
    if filename=='bureau.csv':
        df['_active']=df.CREDIT_ACTIVE.eq('Active').astype(int)
        df['_closed']=df.CREDIT_ACTIVE.eq('Closed').astype(int)
    if filename=='previous_application.csv':
        df['_approved']=df.NAME_CONTRACT_STATUS.eq('Approved').astype(int)
        df['_refused']=df.NAME_CONTRACT_STATUS.eq('Refused').astype(int)
    return df.groupby('SK_ID_CURR',sort=True).agg(**AGGREGATIONS[filename]).reset_index()

def approved_bureau_aggregate(df):
    # User approved 2026-10-05: eligibility change only; original formulas retained.
    return aggregate(df.loc[~df.DAYS_CREDIT_UPDATE.gt(0)], 'bureau.csv')

def derive(df):
    df=df.copy()
    cats=df.select_dtypes(include='object').columns
    df[cats]=df[cats].fillna('Unknown')
    df['INCOME_CREDIT_RATIO']=df.AMT_INCOME_TOTAL/df.AMT_CREDIT
    df['ANNUITY_INCOME_RATIO']=df.AMT_ANNUITY/df.AMT_INCOME_TOTAL
    df['YEARS_BIRTH']=-df.DAYS_BIRTH/365.25
    df['YEARS_EMPLOYED']=(-df.DAYS_EMPLOYED/365.25).mask(df.DAYS_EMPLOYED.eq(365243))
    return df

def check_ids(df):
    if df.SK_ID_CURR.isna().any() or not df.SK_ID_CURR.is_unique:
        raise ValueError('Missing or duplicate applicant ID')

def check_time_audit(rows):
    # Observation updates can affect monetary/state aggregates even when dates are omitted.
    relevant={'DAYS_CREDIT','DAYS_CREDIT_UPDATE','DAYS_DECISION','DAYS_ENTRY_PAYMENT','MONTHS_BALANCE'}
    future=[r for r in rows if r['field'] in relevant and r['positive_records']>0]
    if future:
        raise ValueError('BLOCKER: post-application historical observations; confirmed data definitions need review')


def audit_inputs(root, out):
    """Full raw reconstruction; descriptive checks never learn modelling parameters."""
    started=time.perf_counter(); root=Path(root); out=Path(out); raw=root/'home-credit-default-risk'
    files=[root/'application_train_full.csv']+sorted(raw.glob('*.csv'))
    fingerprints=[{'file':str(p.relative_to(root)), 'bytes':p.stat().st_size, 'sha256':sha256(p)} for p in files]
    save_json(out/'results/input_fingerprints.json',fingerprints)
    shared=pd.read_csv(root/'application_train_full.csv'); check_ids(shared)
    if shared.shape!=(307511,153) or shared.TARGET.isna().any() or set(shared.TARGET)!={0,1}:
        raise ValueError('Input contract differs from confirmed spec')
    if shared.TARGET.value_counts().to_dict()!={0:282686,1:24825}:
        raise ValueError('Unexpected label counts')
    train=pd.read_csv(raw/'application_train.csv'); test=pd.read_csv(raw/'application_test.csv')
    sample=pd.read_csv(raw/'sample_submission.csv')
    for d in (train,test,sample): check_ids(d)
    if len(test)!=48744 or not test.SK_ID_CURR.equals(sample.SK_ID_CURR):
        raise ValueError('Official test IDs differ from sample submission')
    if set(train.SK_ID_CURR)!=set(shared.SK_ID_CURR) or set(train.SK_ID_CURR)&set(test.SK_ID_CURR):
        raise ValueError('Applicant membership differs or train/test overlap')
    time_rows=[]; used_time_rows=[]; future_sources=[]; filtered_bureau=None; excluded_ids=[]
    for fn, spec in AGGREGATIONS.items():
        cols={c for c,op in spec.values() if not c.startswith('_')} | {'SK_ID_CURR'} | set(TIME_COLUMNS[fn])
        if fn=='bureau.csv': cols.add('CREDIT_ACTIVE')
        if fn=='previous_application.csv': cols.add('NAME_CONTRACT_STATUS')
        d=pd.read_csv(raw/fn,usecols=sorted(cols))
        for c in TIME_COLUMNS[fn]:
            s=d[c]; n=int(s.gt(0).sum())
            time_rows.append({'source':fn,'field':c,'records':len(d),'missing':int(s.isna().sum()),
                              'min':float(s.min()),'max':float(s.max()),'positive_records':n})
            # Future scheduled dates/updates are audited but are not themselves input columns.
            if n and c in ('DAYS_CREDIT','DAYS_CREDIT_UPDATE','DAYS_DECISION','DAYS_ENTRY_PAYMENT','MONTHS_BALANCE'):
                future_sources.append({'source':fn,'field':c,'positive_records':n})
        a=aggregate(d,fn)
        used=d
        if fn=='bureau.csv':
            excluded=d.loc[d.DAYS_CREDIT_UPDATE.gt(0)]
            excluded.to_csv(out/'results/bureau_excluded_records.csv',index=False)
            excluded_ids=sorted(excluded.SK_ID_CURR.unique().tolist())
            filtered_bureau=approved_bureau_aggregate(d).set_index('SK_ID_CURR')
            used=d.loc[~d.DAYS_CREDIT_UPDATE.gt(0)]
        for c in TIME_COLUMNS[fn]:
            used_time_rows.append({'source':fn,'field':c,'records':len(used),
                'missing':int(used[c].isna().sum()),'min':float(used[c].min()),
                'max':float(used[c].max()),'positive_records':int(used[c].gt(0).sum())})
        train=train.merge(a,on='SK_ID_CURR',how='left',validate='one_to_one',sort=False)
        test=test.merge(a,on='SK_ID_CURR',how='left',validate='one_to_one',sort=False)
        print(f'Reconstructed {fn}: {len(d):,} records -> {len(a):,} applicants',flush=True)
        del d,a
    pd.DataFrame(time_rows).to_csv(out/'results/time_audit.csv',index=False)
    train=derive(train); test=derive(test)
    a=train.set_index('SK_ID_CURR').loc[shared.SK_ID_CURR].reset_index()
    if set(a.columns)!=set(shared.columns): raise ValueError('Reconstructed schema mismatch')
    comparisons=[]
    for c in shared.columns:
        s=shared[c]; t=a[c]
        if pd.api.types.is_numeric_dtype(s):
            mask=np.isclose(s.to_numpy(),t.to_numpy(),rtol=1e-10,atol=1e-8,equal_nan=True)
            finite=s.notna()&t.notna()
            diff=float(np.abs(s[finite]-t[finite]).max()) if finite.any() else 0.
        else: mask=s.eq(t).to_numpy(); diff=None
        comparisons.append({'field':c,'mismatches':int((~mask).sum()),'max_abs_difference':diff})
    pd.DataFrame(comparisons).to_csv(out/'results/train_reconstruction_comparison.csv',index=False)
    if any(r['mismatches'] for r in comparisons):
        raise ValueError('BLOCKER: reconstructed values differ from shared input; inspect comparison CSV')
    check_time_audit(used_time_rows)
    pd.DataFrame(used_time_rows).to_csv(out/'results/time_audit_after_filter.csv',index=False)
    # The shared table remains the training source. Replace only its seven bureau columns.
    changes=[]
    for frame,role in [(shared,'train'),(test,'official_test')]:
        clean=filtered_bureau.reindex(frame.SK_ID_CURR)
        for c in AGGREGATIONS['bureau.csv']:
            old=frame[c].to_numpy(); new=clean[c].to_numpy()
            different=~np.isclose(old,new,rtol=1e-10,atol=1e-8,equal_nan=True)
            for i in np.flatnonzero(different):
                changes.append({'dataset':role,'SK_ID_CURR':int(frame.SK_ID_CURR.iloc[i]),
                    'field':c,'old':old[i],'new':new[i]})
            frame[c]=new
    pd.DataFrame(changes).to_csv(out/'results/approved_bureau_changes.csv',index=False)
    save_json(out/'results/bureau_filter_policy.json',{'approved':'2026-10-05 by user',
        'rule':'exclude DAYS_CREDIT_UPDATE > 0 before bureau aggregation',
        'excluded_records':len(excluded_ids),'excluded_applicant_ids':excluded_ids,
        'changed_cells':len(changes),'all_applicants_retained':True,'source_files_modified':False})
    numeric=shared.drop(columns=['SK_ID_CURR','TARGET']).select_dtypes(include='number')
    if np.isinf(numeric.to_numpy()).any(): raise ValueError('Non-finite numeric inputs')
    for df in (shared,test):
        for c in LOG_COLUMNS:
            if (df[c].dropna()<=0).any(): raise ValueError(f'Unexpected range for log field {c}')
    summary={'status':'passed_with_authorized_bureau_filter','train_shape':list(shared.shape),'test_shape':list(test.shape),
             'class_counts':{str(k):int(v) for k,v in shared.TARGET.value_counts().items()},
             'numeric_inputs':numeric.shape[1],'categorical_inputs':16,
             'missing_columns':int(shared.isna().any().sum()),
             'high_missing_columns':int((shared.isna().mean()>.4).sum()),
             'constant_columns':shared.nunique().loc[lambda s:s<=1].index.tolist(),
             'reconstruction_mismatches':0,'historical_future_records_in_used_offsets':0,
             'comparison_tolerance':{'rtol':1e-10,'atol':1e-8},
             'bureau_excluded_records':len(excluded_ids),'bureau_changed_cells':len(changes),
             'seconds':time.perf_counter()-started}
    save_json(out/'results/data_audit.json',summary)
    test.to_pickle(out/'results/official_test_features.pkl')
    # Save definitions and dictionary descriptions of source fields, without label access.
    save_json(out/'results/feature_definitions.json',AGGREGATIONS)
    desc=pd.read_csv(raw/'HomeCredit_columns_description.csv',encoding='latin1')
    wanted=set().union(*[{c for c,_ in s.values() if not c.startswith('_')} for s in AGGREGATIONS.values()])|{'DAYS_EMPLOYED','DAYS_BIRTH','MONTHS_BALANCE','DAYS_CREDIT','DAYS_CREDIT_UPDATE','DAYS_INSTALMENT'}
    desc[desc.Row.isin(wanted)].to_csv(out/'results/source_field_dictionary.csv',index=False)
    return shared,test,summary

def make_splits(df):
    a=df[['SK_ID_CURR','TARGET']].sort_values('SK_ID_CURR').reset_index(drop=True)
    check_ids(a)
    dev,hold=train_test_split(np.arange(len(a)),test_size=.05,stratify=a.TARGET,random_state=SEED)
    a['partition']='holdout'; a['fold']=-1; a.loc[dev,'partition']='development'
    skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    dev=np.sort(dev)
    for k,(_,val) in enumerate(skf.split(dev,a.loc[dev,'TARGET'])): a.loc[dev[val],'fold']=k
    assert len(hold)==int(np.ceil(.05*len(a))) and a.SK_ID_CURR.is_unique
    return a

def correct_fields(df):
    x=df.copy()
    if 'DAYS_EMPLOYED' in x:
        special=x.DAYS_EMPLOYED.eq(365243)
        if not np.allclose(x.YEARS_BIRTH,-x.DAYS_BIRTH/365.25,atol=1e-10,rtol=0):
            raise ValueError('Age conversion mismatch')
        expected=(-x.DAYS_EMPLOYED/365.25).mask(special)
        if not np.allclose(x.YEARS_EMPLOYED,expected,atol=1e-10,rtol=0,equal_nan=True):
            raise ValueError('Employment conversion mismatch')
        x['EMPLOYED_SPECIAL']=special.astype(np.float64)
    x=x.drop(columns=['SK_ID_CURR','TARGET','DAYS_BIRTH','DAYS_EMPLOYED'],errors='ignore')
    return x.rename(columns={'INSTALL_AVG_DAYS_LATE':'INSTALL_MEAN_DAYS_ENTRY_PAYMENT'})

class Preprocessor:
    """Training-only medians/scales/vocabulary. Complete dummies, sparse float64.

    Indicators for all numeric columns (even currently complete) ensure future missing
    values are represented; only employment's proven duplicate indicator is omitted.
    All-missing training numeric fields have a documented fallback median of zero.
    """
    def fit(self,x):
        self.columns_=list(x.columns)
        self.numeric_=x.select_dtypes(include='number').columns.tolist()
        self.categorical_=[c for c in x if c not in self.numeric_]
        n=x[self.numeric_]
        if np.isinf(n.to_numpy()).any(): raise ValueError('Infinite numeric inputs')
        self.medians_=n.median().fillna(0.)
        self.all_missing_=[c for c in self.numeric_ if n[c].isna().all()]
        self.binary_=[c for c in self.numeric_ if n[c].dropna().isin([0,1]).all() and c not in self.all_missing_]
        self.scaled_=[c for c in self.numeric_ if c not in self.binary_]
        self.indicator_=[c for c in self.numeric_ if c!='EMPLOYED_SPECIAL']
        self.employment_duplicate_=False
        if 'EMPLOYED_SPECIAL' in x and 'YEARS_EMPLOYED' in x:
            self.employment_duplicate_=np.array_equal(x.YEARS_EMPLOYED.isna().astype(int),x.EMPLOYED_SPECIAL)
            if self.employment_duplicate_: self.indicator_.remove('YEARS_EMPLOYED')
        filled=self._filled(x)
        self.scaler_=StandardScaler().fit(filled[self.scaled_]) if self.scaled_ else None
        self.encoder_=OneHotEncoder(handle_unknown='ignore',drop=None,sparse_output=True,dtype=np.float64)
        if self.categorical_: self.encoder_.fit(x[self.categorical_].fillna('Unknown').astype(str))
        self.names_=np.array(self.numeric_+[c+'__missing' for c in self.indicator_]+
            (self.encoder_.get_feature_names_out(self.categorical_).tolist() if self.categorical_ else []))
        return self
    def _filled(self,x):
        n=x[self.numeric_].fillna(self.medians_).astype(float)
        for c in LOG_COLUMNS:
            if c in n:
                if (n[c]<0).any(): raise ValueError(f'Invalid log range {c}')
                n[c]=np.log1p(n[c])
        return n
    def transform(self,x):
        if list(x.columns)!=self.columns_: raise ValueError('Input feature order/schema mismatch')
        n=self._filled(x)
        if self.scaled_: n[self.scaled_]=self.scaler_.transform(n[self.scaled_])
        flags=x[self.indicator_].isna().to_numpy(dtype=np.float64)
        if self.employment_duplicate_:
            # The frozen deduplication rule must also hold outside training.
            if not np.array_equal(x.YEARS_EMPLOYED.isna().astype(int),x.EMPLOYED_SPECIAL):
                raise ValueError('Employment indicator no longer duplicates missingness')
        parts=[sparse.csr_matrix(n.to_numpy()),sparse.csr_matrix(flags)]
        if self.categorical_: parts.append(self.encoder_.transform(x[self.categorical_].fillna('Unknown').astype(str)))
        z=sparse.hstack(parts,format='csr'); z.sort_indices()
        if not np.isfinite(z.data).all(): raise ValueError('Invalid transformed values')
        return z
    def fit_transform(self,x): return self.fit(x).transform(x)
    def get_feature_names_out(self): return self.names_
    def unknown_counts(self,x):
        return {c:int((~x[c].fillna('Unknown').astype(str).isin(v)).sum())
                for c,v in zip(self.categorical_,self.encoder_.categories_)} if self.categorical_ else {}
    def metadata(self):
        return {'numeric':self.numeric_,'binary':self.binary_,'scaled':self.scaled_,
                'missing_indicators':self.indicator_,'all_missing_training':self.all_missing_,
                'employment_duplicate_removed':bool(self.employment_duplicate_),
                'feature_count':len(self.names_),'one_hot':'complete; unseen all zero; no reference dropped',
                'medians':self.medians_.to_dict(),
                'scale_means':dict(zip(self.scaled_,self.scaler_.mean_)) if self.scaled_ else {},
                'scale_stds':dict(zip(self.scaled_,self.scaler_.scale_)) if self.scaled_ else {},
                'categories':{c:v.tolist() for c,v in zip(self.categorical_,self.encoder_.categories_)} if self.categorical_ else {}}
