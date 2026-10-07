"""Portable, read-only evidence verification for the integrated course notebook."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

BASE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_csv(relative):
    return pd.read_csv(BASE/relative,float_precision='round_trip')

def verify_manifest():
    entries=json.loads((BASE/'support/manifest.json').read_text())
    for r in entries:
        assert sha(BASE/r['file'])==r['sha256'],f"Package file changed: {r['file']}"
    return len(entries)

def check_prediction_alignment(p,membership):
    assert p.SK_ID_CURR.is_unique and set(p.SK_ID_CURR)==set(membership.SK_ID_CURR)
    expected=membership.set_index('SK_ID_CURR').loc[p.SK_ID_CURR]
    assert np.array_equal(p.TARGET,expected.TARGET)
    assert np.array_equal(p.fold,expected.fold)
    assert np.isfinite(p.probability).all() and p.probability.between(0,1).all()

def verify_abc():
    membership=read_csv('support/membership.csv');dev=membership.loc[membership.partition.eq('development')]
    p=read_csv('support/abc/oof_predictions.csv.gz');saved=read_csv('support/abc/cv_folds.csv')
    assert len(p)==2629215 and set(zip(saved.model,saved.group))=={(m,g) for m in ['LR','RF','LGB'] for g in 'ABC'}
    rows=[]
    for (model,group),q in p.groupby(['model','group']):
        check_prediction_alignment(q,dev)
        for fold,f in q.groupby('fold'):
            auc=roc_auc_score(f.TARGET,f.probability);ap=average_precision_score(f.TARGET,f.probability)
            r=saved.loc[saved.model.eq(model)&saved.group.eq(group)&saved.fold.eq(fold)]
            assert len(r)==1
            assert abs(auc-r.roc_auc.iloc[0])<1e-14 and abs(ap-r.ap.iloc[0])<1e-14
            rows.append(dict(model=model,group=group,fold=int(fold),roc_auc=auc,ap=ap))
    cv=pd.DataFrame(rows)
    summary=cv.groupby(['model','group']).agg(auc_mean=('roc_auc','mean'),auc_sd=('roc_auc','std'),ap_mean=('ap','mean'),ap_sd=('ap','std')).reset_index()
    expected=read_csv('support/abc/cv_summary.csv').sort_values(['model','group']).reset_index(drop=True)
    np.testing.assert_allclose(summary[['auc_mean','auc_sd','ap_mean','ap_sd']],expected[['auc_mean','auc_sd','ap_mean','ap_sd']],atol=1e-14,rtol=0)
    deltas=read_csv('support/abc/paired_deltas.csv')
    for r in deltas.itertuples():
        hi,lo=r.contrast.split('-');v=cv.loc[cv.model.eq(r.model)].pivot(index='fold',columns='group',values=['roc_auc','ap'])
        a=v.roc_auc[hi]-v.roc_auc[lo];b=v.ap[hi]-v.ap[lo]
        av=a.mean() if r.fold=='mean' else a.loc[int(r.fold)];bv=b.mean() if r.fold=='mean' else b.loc[int(r.fold)]
        assert abs(av-r.auc_delta)<1e-14 and abs(bv-r.ap_delta)<1e-14
    return cv,summary,deltas,dict(status='PASS',oof_rows=len(p),fold_scores=len(cv),new_model_fits=0,scope='OOF/ID/label/fold and arithmetic verification; no raw reconstruction or model replay in this notebook')

def verify_optimization():
    membership=read_csv('support/membership.csv');dev=membership.loc[membership.partition.eq('development')]
    folds=read_csv('support/optimization/cv_folds.csv');blend_folds=read_csv('support/optimization/blend_folds.csv')
    rows=[]
    for tag in ['C11_base','raw80_20','rank80_20','rank70_20_10']:
        p=read_csv(f'support/optimization/{tag}_oof.csv.gz');check_prediction_alignment(p,dev)
        for fold,q in p.groupby('fold'):
            auc=roc_auc_score(q.TARGET,q.probability);ap=average_precision_score(q.TARGET,q.probability)
            src=folds if tag=='C11_base' else blend_folds
            r=src.loc[src.tag.eq(tag)&src.fold.eq(fold)].iloc[0]
            assert abs(auc-r.roc_auc)<1e-14 and abs(ap-r.ap)<1e-14
            rows.append(dict(tag=tag,fold=fold,roc_auc=auc,ap=ap))
    records=json.loads((BASE/'support/optimization/blend_summary.json').read_text())
    verified=pd.DataFrame(rows)
    reference=verified.loc[verified.tag.eq('C11_base')].set_index('fold').sort_index()
    tol=.0003
    for r in records:
        q=verified.loc[verified.tag.eq(r['tag'])].set_index('fold').sort_index()
        delta=q.roc_auc-reference.roc_auc
        np.testing.assert_allclose(delta,r['fold_deltas'],atol=1e-14,rtol=0)
        for metric,column in [('auc','roc_auc'),('ap','ap')]:
            assert abs(q[column].mean()-r[metric+'_mean'])<1e-14
            assert abs(q[column].std(ddof=1)-r[metric+'_sd'])<1e-14
        assert bool(delta.min()>=0 and delta.mean()>tol)==r['stable_gain']
        assert r['complexity']==[len(r['scheme']['weights']),0 if r['scheme']['method']=='raw' else 1]
    eligible=[r for r in records if r['stable_gain']]
    maximum=max(r['auc_mean'] for r in eligible)
    near=[r for r in eligible if maximum-r['auc_mean']<=tol]
    chosen=sorted(near,key=lambda r:(r['complexity'],-r['auc_mean'],-r['ap_mean'],r['tag']))[0]
    assert chosen['tag']=='rank80_20'
    gap=next(r['auc_mean'] for r in records if r['tag']=='rank70_20_10')-chosen['auc_mean']
    return pd.DataFrame(rows),dict(status='PASS',selection=chosen['tag'],tolerance=tol,higher_score_gap=gap,
        rank_scope='within-validation-fold percentile ranks; holdout/test CDF is a different rule',new_model_fits=0)

def feature_counts():
    groups=json.loads((BASE/'support/abc/feature_sets.json').read_text())['groups']
    rows=[]
    for g,v in groups.items():
        for m,features in v['models'].items():
            rows.append(dict(group=g,table_columns=v['column_count'],model=m,input_columns=features['input_count'],encoded_widths=', '.join(map(str,sorted({f['count'] for f in features['folds'].values()})))))
    return pd.DataFrame(rows)

def verify_frozen_source():
    protocol=json.loads((BASE/'support/original_protocol.json').read_text())['protocol']
    for rel,h in protocol['code'].items():assert sha(BASE/'source_snapshot'/rel)==h,rel
    return len(protocol['code'])


def verify_secondary_candidates():
    """Recalculate every frozen candidate fold and reproduce both selection pools."""
    dev=read_csv('support/membership.csv').query("partition == 'development'")
    saved=read_csv('support/optimization/cv_folds.csv')
    predictions=read_csv('support/optimization/all_candidates_oof.csv.gz')
    summaries=json.loads((BASE/'support/optimization/cv_summary.json').read_text())
    rows=[]
    assert set(predictions.tag)==set(saved.tag)
    for tag,p in predictions.groupby('tag',sort=False):
        check_prediction_alignment(p,dev)
        for fold,q in p.groupby('fold'):
            r=saved.loc[saved.tag.eq(tag)&saved.fold.eq(fold)]
            assert len(r)==1
            auc=roc_auc_score(q.TARGET,q.probability);ap=average_precision_score(q.TARGET,q.probability)
            assert abs(auc-r.iloc[0].roc_auc)<1e-14 and abs(ap-r.iloc[0].ap)<1e-14
            rows.append(dict(tag=tag,fold=int(fold),roc_auc=auc,ap=ap))
    folds=pd.DataFrame(rows)
    for s in summaries:
        q=folds[folds.tag.eq(s['tag'])]
        for metric,column in [('auc','roc_auc'),('ap','ap')]:
            assert abs(q[column].mean()-s[metric+'_mean'])<1e-14
            assert abs(q[column].std(ddof=1)-s[metric+'_sd'])<1e-14
    def choose(pool):
        maximum=max(s['auc_mean'] for s in pool)
        eligible=[s for s in pool if s['auc_mean']>=maximum-.0003]
        return sorted(eligible,key=lambda s:(s['complexity'],-s['auc_mean'],-s['ap_mean'],s['tag']))[0]
    parameter=choose([s for s in summaries if s['family']=='base'])
    combined=choose(summaries);maximum=max(summaries,key=lambda s:s['auc_mean'])
    lock=json.loads((BASE/'support/optimization/model_lock.json').read_text())
    assert parameter['tag']==lock['parameter_winner']['tag']=='C10_base'
    assert combined['tag']==lock['LGB']['tag']=='C11_base'
    deltas=read_csv('support/optimization/feature_paired_deltas.csv')
    keyed=folds.set_index(['tag','fold'])
    for r in deltas.itertuples():
        new=keyed.loc[(r.new_tag,r.fold)];old=keyed.loc[(r.reference_tag,r.fold)]
        assert abs(new.roc_auc-old.roc_auc-r.auc_delta)<1e-14
        assert abs(new.ap-old.ap-r.ap_delta)<1e-14
    return folds,dict(status='PASS',candidate_folds=len(folds),prediction_rows=len(predictions),
        parameter_pool_choice=parameter['tag'],combined_pool_choice=combined['tag'],
        combined_maximum=maximum['tag'],combined_tolerance_floor=maximum['auc_mean']-.0003,
        parameter_reference_below_floor=maximum['auc_mean']-.0003-parameter['auc_mean'],
        feature_contrast_rows=len(deltas),new_model_fits=0,
        scope='Saved-score recalculation and frozen-rule replay, not model replay or retraining')
