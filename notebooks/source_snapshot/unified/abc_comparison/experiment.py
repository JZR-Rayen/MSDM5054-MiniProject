"""Application-only extension of the frozen unified experiment; no final fits."""
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'unified'), str(ROOT/'baseline')]
import run as original
import rf_processing as rf
from baseline_data import derive, sha256, save_json, AGGREGATIONS
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent
HISTORY_PREFIXES = ('BUREAU_', 'PREV_', 'INSTALL_', 'POS_', 'CC_', 'R_', 'H_', 'BB_')
DERIVED = ['INCOME_CREDIT_RATIO','ANNUITY_INCOME_RATIO','YEARS_BIRTH','YEARS_EMPLOYED']
MODELS = ['LR','RF','LGB']
SOURCE = '历史结果核验后复用'

def read_scores(path):
    return pd.read_csv(path, float_precision='round_trip')

def application_frame(raw):
    if any(c.startswith(HISTORY_PREFIXES) for c in raw):
        raise ValueError('Historical field in application source')
    return derive(raw)

def application_inputs(model, frame):
    if any(c.startswith(HISTORY_PREFIXES) for c in frame):
        raise ValueError('Historical field in A')
    if model != 'RF':
        return original.correct_fields(frame)
    # Same primitives/order as f3_features; omit ratios whose operands are absent.
    x = rf.prepare_model_data(frame)
    x = x.drop(columns=[c for c in x if c in ['TARGET','partition','fold'] or c.startswith('SK_ID')])
    redundant = {f'{base}_{suffix}' for base in rf.repeated_property_bases for suffix in ['MODE','MEDI']}
    result = x[[c for c in x if c not in redundant]].copy()
    for name, (numerator, denominator) in rf.business_ratio_definitions.items():
        if numerator in x and denominator in x:
            result[name] = rf.safe_ratio(x, numerator, denominator)
    return result

def digest_object(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def snapshot():
    """Back up only files to be edited; protect original evidence by hash inventory."""
    path=OUT/'results/before_manifest.json'
    if path.exists():return
    import shutil
    changes=sorted((ROOT/'unified/changes').glob('*.md'))
    for p in changes:
        dest=OUT/'before'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    paths=set(changes)
    for pattern in ['unified/*.py','unified/protocol.json','unified/results/*','unified/models/*',
                    'baseline/*.py','baseline/results/model_lock.json','processed_data/*.csv',
                    '*/*.ipynb','unified/optimization_20261007/delivery_manifest.json',
                    'unified/optimization_20261007/results/*','unified/optimization_20261007/models/*']:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    paths.add(ROOT/'home-credit-default-risk/application_train.csv')
    save_json(path,{str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)})

def check_historical():
    frozen=json.loads((ROOT/'unified/protocol.json').read_text())
    p=frozen['protocol'];checks={}
    assert digest_object(p)==frozen['key']
    for path,h in {**p['inputs'],**p['code']}.items():
        assert sha256(ROOT/path)==h, f'Frozen source/input changed: {path}'
        checks[path]=h
    for key,cfg in [('lr',original.LR_CONFIG),('rf',original.RF_CONFIG),('lgb',original.LGB_CONFIG)]:
        assert json.loads(json.dumps(cfg))==p[key]
    for key,current in [('numpy',np.__version__),('pandas',pd.__version__),
                        ('sklearn',original.sklearn.__version__),('lightgbm',original.lgb.__version__),
                        ('python',sys.version),('platform',platform.platform())]:
        assert current==p['runtime'][key],f'Runtime changed: {key}'
    # Old manifest is the immutable pre-optimization inventory of original runs.
    manifest=json.loads((ROOT/'unified/optimization_20261007/before/unified/delivery_manifest.json').read_text())
    inventory={r['file']:r['sha256'] for r in manifest}
    historical=[]
    for model in MODELS:
        for version in ['v1','v2']:
            for fold in range(3):
                tag=f'{model}_{version}_fold{fold}'
                for suffix in ['predictions.csv','fit.json','features.json']:
                    rel=f'unified/results/{tag}_{suffix}';h=sha256(ROOT/rel)
                    assert inventory[rel]==h,rel
                    checks[rel]=h
                meta=json.loads((ROOT/f'unified/results/{tag}_fit.json').read_text())
                assert meta['key']==frozen['key']
                assert meta['prediction_sha256']==checks[f'unified/results/{tag}_predictions.csv']
                historical.append(dict(tag=tag,meta=meta))
    refs=[]
    for path in sorted((ROOT/'unified/models').glob('*.joblib')):
        rel=str(path.relative_to(ROOT));h=sha256(path)
        assert sha256(ROOT/'unified/optimization_20261007/before'/rel)==h
        replay_audit=json.loads((ROOT/'unified/results/model_replay_validation.json').read_text())
        assert h in [r['model_sha256'] for r in replay_audit['models_replayed']]
        refs.append(dict(path=rel,sha256=h,scope='historical selected-v2 final model; NOT a CV-fold model; not used for ABC scores'))
    save_json(OUT/'models/historical_references.json',dict(models=refs,
        outer_fold_models_available=False,note='Original run discarded outer-fold models. B/C reuse authenticated fold predictions and fit/feature audits; no model replay claimed for B/C CV.'))
    return frozen,checks,historical

def load_inputs():
    membership=pd.read_csv(ROOT/'baseline/splits/membership.csv')
    assert len(membership)==307511 and membership.SK_ID_CURR.is_unique
    data={}
    for group,version in [('B','v1'),('C','v2')]:
        d=pd.read_csv(ROOT/f'processed_data/application_train_processed_{version}.csv')
        assert d.SK_ID_CURR.is_unique and set(d.SK_ID_CURR)==set(membership.SK_ID_CURR)
        d=d.set_index('SK_ID_CURR').loc[membership.SK_ID_CURR].reset_index()
        assert np.array_equal(d.TARGET,membership.TARGET)
        assert len(d.columns)==(153 if group=='B' else 192)
        data[group]=d
    pd.testing.assert_frame_equal(data['B'],data['C'][data['B'].columns],check_exact=True)
    raw=pd.read_csv(ROOT/'home-credit-default-risk/application_train.csv')
    assert raw.shape==(307511,122) and raw.SK_ID_CURR.is_unique
    assert set(raw.SK_ID_CURR)==set(membership.SK_ID_CURR)
    reconstructed=application_frame(raw).set_index('SK_ID_CURR').loc[membership.SK_ID_CURR].reset_index()
    # Keep identical CSV-serialized application values across groups, after independently
    # proving every retained column comes from raw application + the shared four formulas.
    a=data['B'][reconstructed.columns].copy()
    pd.testing.assert_frame_equal(a,reconstructed,check_exact=False,rtol=1e-12,atol=1e-10)
    expected_history={c for spec in AGGREGATIONS.values() for c in spec}
    assert set(data['B'])-set(a)==expected_history and len(expected_history)==27
    assert len(set(data['C'])-set(data['B']))==39
    data['A']=a
    assert membership.partition.value_counts().to_dict()=={'development':292135,'holdout':15376}
    assert set(membership.loc[membership.partition.eq('development'),'fold'])=={0,1,2}
    save_json(OUT/'results/data_validation.json',dict(applicants=307511,development=292135,holdout_not_scored=15376,
        same_ids_labels_folds=True,raw_application_reconstructed=True,reconstruction_tolerance=dict(rtol=1e-12,atol=1e-10),
        A_uses_verified_shared_application_projection=True,A_columns=126,B_columns=153,C_columns=192,
        no_applicant_dropped=True,no_history_placeholder_columns=True,
        raw_application_sha256=sha256(ROOT/'home-credit-default-risk/application_train.csv')))
    return data,membership,raw.columns.tolist()

def audit_input_equivalence(data):
    records={}
    for model in MODELS:
        a=application_inputs(model,data['A'])
        for group,version in [('B','v1'),('C','v2')]:
            x=original.model_inputs(model,data[group])
            pd.testing.assert_frame_equal(a,x[a.columns],check_exact=True)
            for fold in range(3):
                audit=json.loads((ROOT/f'unified/results/{model}_{version}_fold{fold}_features.json').read_text())
                assert audit['before']==list(x),f'{model} {group} input schema drift'
            records[f'{model}_{group}']=dict(application_input_columns_identical=True,columns=len(x))
    return records

def make_feature_sets(data,raw_columns):
    aggs={c:dict(source=fn,operation=op,raw_field=field) for fn,spec in AGGREGATIONS.items() for c,(field,op) in spec.items()}
    derivation={'INCOME_CREDIT_RATIO':['AMT_INCOME_TOTAL','AMT_CREDIT'],
                'ANNUITY_INCOME_RATIO':['AMT_ANNUITY','AMT_INCOME_TOTAL'],
                'YEARS_BIRTH':['DAYS_BIRTH'],'YEARS_EMPLOYED':['DAYS_EMPLOYED']}
    groups={}
    for group in ['A','B','C']:
        sources={}
        for c in data[group]:
            if c in raw_columns:sources[c]=dict(source='application_train.csv',role='identifier' if c=='SK_ID_CURR' else 'label' if c=='TARGET' else 'feature')
            elif c in derivation:sources[c]=dict(source='application-derived',parents=derivation[c],function='baseline_data.derive')
            elif c in aggs:sources[c]=aggs[c]
            else:sources[c]=dict(source='installments_payments.csv' if c.startswith('R_') else 'bureau_balance.csv + bureau.csv' if c.startswith('BB_') else 'bureau.csv' if c.startswith('H_ACTIVE') or c.startswith('H_CLOSED') or c.startswith('H_CURRENT') or c.startswith('H_NEW') or c.startswith('H_DAYS') else 'previous_application.csv' if c.startswith('H_PREV') else 'credit_card_balance.csv' if c.startswith('H_CC') else 'POS_CASH_balance.csv',definition='baseline/baseline_features_v2.py')
        groups[group]=dict(columns=list(data[group]),column_count=len(data[group].columns),sources=sources,models={})
        for model in MODELS:
            x=application_inputs(model,data[group]) if group=='A' else original.model_inputs(model,data[group])
            entries={}
            for fold in range(3):
                tag=f'{model}_{group}_fold{fold}' if group=='A' else f'{model}_{"v1" if group=="B" else "v2"}_fold{fold}'
                base=OUT if group=='A' else ROOT/'unified'
                audit=json.loads((base/f'results/{tag}_features.json').read_text())
                entries[str(fold)]=dict(features=audit['after'],count=len(audit['after']),audit_file=str((base/f'results/{tag}_features.json').relative_to(ROOT)))
            groups[group]['models'][model]=dict(input_features=list(x),input_count=len(x.columns),folds=entries)
    save_json(OUT/'results/feature_sets.json',dict(groups=groups,
        application_definition='All raw application columns, including EXT_SOURCE and AMT_REQ_CREDIT_BUREAU fields already in application; no joined history information.',
        learned_features='Indicators, imputation, one-hot, scaling, spline knots and interactions use the current training portion only.',
        application_derived_dependencies=derivation))

def event(record):
    with (OUT/'results/fit_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**record))+'\n')

def run(resume=True,validate_only=False):
    started=time.perf_counter()
    for d in ['results','models','logs','figures']:(OUT/d).mkdir(parents=True,exist_ok=True)
    snapshot()
    frozen,checks,historical=check_historical()
    data,membership,raw_columns=load_inputs()
    equivalent=audit_input_equivalence(data)
    protocol=dict(original_key=frozen['key'],original_protocol=frozen['protocol'],
                  code={str(Path(__file__).relative_to(ROOT)):sha256(__file__)},
                  raw_application=sha256(ROOT/'home-credit-default-risk/application_train.csv'),
                  groups={'A':'raw application + shared four derived fields','B':'V1','C':'V2'},
                  expected_new_outer_fits=9,expected_inner_fits=3,holdout_scored=False,
                  shared_implementation_modified=False)
    key=digest_object(protocol)
    lock=OUT/'protocol.json'
    if lock.exists():assert json.loads(lock.read_text())['key']==key,'ABC protocol changed; use separate output directory'
    else:save_json(lock,dict(key=key,protocol=protocol,frozen_at=time.time()))
    save_json(OUT/'results/provenance.json',dict(original_key=frozen['key'],verified_hashes=checks,
        B_C_source=SOURCE,BC_application_equivalence=equivalent,
        original_cv_fold_models_saved=False,original_cv_prediction_origin='unified/run.py::run_cv; authenticated against pre-optimization manifest and fold fit metadata',
        validation_time=time.time()))
    original_out=original.OUT;original.OUT=OUT
    dev=membership.partition.eq('development').to_numpy();folds=membership.fold.to_numpy()
    rows=[];predictions=[];replays=[]
    try:
        for model in MODELS:
            x=application_inputs(model,data['A'])
            for fold in range(3):
                tag=f'{model}_A_fold{fold}';tr=dev&(folds!=fold);va=dev&(folds==fold)
                predpath=OUT/f'results/{tag}_predictions.csv';mp=OUT/f'models/{tag}.joblib';meta=OUT/f'results/{tag}_fit.json'
                if resume and meta.exists():
                    info=json.loads(meta.read_text());assert info['key']==key
                    assert sha256(mp)==info['model_sha256'] and sha256(predpath)==info['prediction_sha256']
                    print('Verified A cache',tag,flush=True)
                else:
                    if validate_only:raise RuntimeError('A fit missing: '+tag)
                    if mp.exists() or predpath.exists():raise RuntimeError('Incomplete artifact; preserve and inspect before retry: '+tag)
                    print('Training',tag,flush=True)
                    event(dict(tag=tag,stage='start',expected_fits=2 if model=='LGB' else 1))
                    try:
                        bundle=original.fit(model,'v1',x.loc[tr],data['A'].loc[tr,'TARGET'],membership.loc[tr,'SK_ID_CURR'],tag)
                    except Exception as exc:
                        event(dict(tag=tag,stage='failed',error=repr(exc)));raise
                    bundle['version']='A';bundle['key']=key;bundle['training_scope']='outer_training_only'
                    bundle['training_ids']=membership.loc[tr,'SK_ID_CURR'].to_numpy()
                    prob=original.predict(bundle,x.loc[va])
                    p=membership.loc[va,['SK_ID_CURR','TARGET']].copy();p['probability']=prob;p.to_csv(predpath,index=False)
                    original.feature_audit(bundle,x.loc[tr],x.loc[va],x.loc[va],[],tag)
                    auditpath=OUT/f'results/{tag}_features.json';audit=json.loads(auditpath.read_text())
                    audit['schema_check_scope']='training and outer validation only; third slot reuses validation, official test not loaded'
                    audit['training_ids_sha256']=hashlib.sha256(bundle['training_ids'].tobytes()).hexdigest()
                    audit['validation_ids_sha256']=hashlib.sha256(membership.loc[va,'SK_ID_CURR'].to_numpy().tobytes()).hexdigest()
                    audit['history_information_absent']=True;save_json(auditpath,audit)
                    joblib.dump(bundle,mp,compress=3)
                    info=dict(key=key,info=bundle['info'],model_sha256=sha256(mp),prediction_sha256=sha256(predpath))
                    save_json(meta,info);event(dict(tag=tag,stage='complete',fit_count=bundle['info']['fit_count'],
                        optimizer_attempts=len(bundle['info'].get('attempts',[])),seconds=bundle['info']['seconds']))
                    del bundle
                # Full outer-validation replay from deserialized model/preprocessor.
                bundle=joblib.load(mp);assert np.array_equal(bundle['training_ids'],membership.loc[tr,'SK_ID_CURR'])
                assert not set(bundle['training_ids'])&set(membership.loc[va,'SK_ID_CURR'])
                p=read_scores(predpath);assert np.array_equal(p.SK_ID_CURR,membership.loc[va,'SK_ID_CURR'])
                assert np.array_equal(p.TARGET,membership.loc[va,'TARGET'])
                replay=original.predict(bundle,x.loc[va]);error=float(np.max(abs(p.probability.to_numpy()-replay)))
                assert error<1e-14 and np.isfinite(replay).all() and ((replay>=0)&(replay<=1)).all()
                if model=='LGB':
                    inner=pd.read_csv(OUT/f'results/{tag}_inner_ids.csv')
                    assert set(inner.SK_ID_CURR)==set(bundle['training_ids'])
                    assert set(inner.inner_role)=={'training','early_stopping'}
                    assert not set(inner.SK_ID_CURR)&set(membership.loc[~tr,'SK_ID_CURR'])
                replays.append(dict(tag=tag,rows=len(p),max_absolute_difference=error,model_sha256=sha256(mp)))
                rows.append(dict(model=model,group='A',fold=fold,roc_auc=roc_auc_score(p.TARGET,p.probability),ap=average_precision_score(p.TARGET,p.probability),source='本轮A组新训练',**info['info']))
                p['model']=model;p['group']='A';p['fold']=fold;predictions.append(p)
                print(tag,rows[-1]['roc_auc'],rows[-1]['ap'],flush=True)
                del bundle
        for record in historical:
            tag=record['tag'];model,version,foldtext=tag.split('_');fold=int(foldtext[-1]);group='B' if version=='v1' else 'C'
            va=dev&(folds==fold);p=read_scores(ROOT/f'unified/results/{tag}_predictions.csv')
            assert p.SK_ID_CURR.is_unique and np.array_equal(p.SK_ID_CURR,membership.loc[va,'SK_ID_CURR'])
            assert np.array_equal(p.TARGET,membership.loc[va,'TARGET'])
            assert np.isfinite(p.probability).all() and p.probability.between(0,1).all()
            rows.append(dict(model=model,group=group,fold=fold,roc_auc=roc_auc_score(p.TARGET,p.probability),ap=average_precision_score(p.TARGET,p.probability),source=SOURCE,**record['meta']['info']))
            rows[-1]['fitting_origin']='historical_verified_reuse'
            p['model']=model;p['group']=group;p['fold']=fold;predictions.append(p)
        pd.DataFrame(rows).to_csv(OUT/'results/cv_folds.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_csv(OUT/'results/oof_predictions.csv',index=False)
        make_feature_sets(data,raw_columns)
        summarize_and_validate(membership,replays)
    finally:original.OUT=original_out
    events=[json.loads(line) for line in (OUT/'results/fit_events.jsonl').read_text().splitlines()]
    cost=dict(successful_outer_fits=sum(r['stage']=='complete' for r in events),
        successful_inner_fits=sum(r['stage']=='complete' and r['tag'].startswith('LGB') for r in events),
        total_successful_fits=sum(r.get('fit_count',0) for r in events),
        failed_attempts=[r for r in events if r['stage']=='failed'],
        lr_optimizer_extra_attempts=sum(max(0,r.get('optimizer_attempts',0)-1) for r in events if r['stage']=='complete'),
        training_seconds=sum(r.get('seconds',0) for r in events),
        invocation_seconds=time.perf_counter()-started,invocation_validate_only=validate_only,
        reused_BC_outer_folds=18,reused_BC_original_fits_including_inner=24,
        no_holdout_or_kaggle_selection=True)
    if not validate_only:save_json(OUT/'results/cost.json',cost)
    else:save_json(OUT/'results/validation_cost.json',cost)
    print(json.dumps(cost,indent=2),flush=True)
    return pd.read_csv(OUT/'results/cv_summary.csv')

def summarize_and_validate(membership,replays):
    folds=read_scores(OUT/'results/cv_folds.csv');oof=read_scores(OUT/'results/oof_predictions.csv')
    dev=membership.loc[membership.partition.eq('development')]
    assert len(oof)==len(dev)*9 and len(folds)==27
    checks=[]
    for (model,group),p in oof.groupby(['model','group'],sort=True):
        assert p.SK_ID_CURR.is_unique and set(p.SK_ID_CURR)==set(dev.SK_ID_CURR)
        matched=p.merge(dev,on='SK_ID_CURR',validate='one_to_one',suffixes=('','_expected'))
        assert np.array_equal(matched.TARGET,matched.TARGET_expected) and np.array_equal(matched.fold,matched.fold_expected)
        for fold,q in p.groupby('fold'):
            r=folds.loc[(folds.model==model)&(folds.group==group)&(folds.fold==fold)].iloc[0]
            auc=roc_auc_score(q.TARGET,q.probability);ap=average_precision_score(q.TARGET,q.probability)
            assert abs(auc-r.roc_auc)<1e-14 and abs(ap-r.ap)<1e-14
            checks.append(dict(model=model,group=group,fold=int(fold),roc_auc=auc,ap=ap))
    summary=folds.groupby(['model','group']).agg(auc_mean=('roc_auc','mean'),auc_sd=('roc_auc','std'),ap_mean=('ap','mean'),ap_sd=('ap','std'),source=('source','first')).reset_index()
    summary.to_csv(OUT/'results/cv_summary.csv',index=False)
    deltas=[]
    for model in MODELS:
        f=folds.loc[folds.model==model].pivot(index='fold',columns='group',values=['roc_auc','ap'])
        for hi,lo in [('B','A'),('C','B'),('C','A')]:
            auc=f.roc_auc[hi]-f.roc_auc[lo];ap=f.ap[hi]-f.ap[lo]
            for fold in range(3):deltas.append(dict(model=model,contrast=f'{hi}-{lo}',fold=str(fold),auc_delta=auc.loc[fold],ap_delta=ap.loc[fold]))
            deltas.append(dict(model=model,contrast=f'{hi}-{lo}',fold='mean',auc_delta=auc.mean(),ap_delta=ap.mean()))
    pd.DataFrame(deltas).to_csv(OUT/'results/paired_deltas.csv',index=False)
    before=json.loads((OUT/'results/before_manifest.json').read_text())
    unchanged={p:sha256(ROOT/p)==h for p,h in before.items() if not p.startswith('unified/changes/')}
    assert all(unchanged.values()),[p for p,ok in unchanged.items() if not ok]
    save_json(OUT/'results/validation.json',dict(status='PASS',oof_rows=len(oof),fold_metric_recomputations=checks,
        A_full_outer_validation_model_replay=replays,all_nine_groups_same_ids_labels_folds=True,
        A_application_only_input_equivalence=True,preprocessing_fit_scope='fresh current outer training; LGB inner categories fitted to inner train then new full outer fit',
        untouched_original_files=unchanged,aggregation='unweighted mean of three outer-fold AUC/AP; sample SD ddof=1; not pooled OOF AUC',
        BC_outer_models_not_available=True,no_new_final_or_submission_model=True,score_csv_parser='round_trip'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    run(validate_only=args.validate_only)
