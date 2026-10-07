"""Bounded, predeclared optimization. Separate immutable historical evidence."""
from pathlib import Path
import argparse, hashlib, json, os, platform, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baseline'));sys.path.insert(0,str(ROOT/'unified'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'unified/runtime/matplotlib'))
import joblib, numpy as np, pandas as pd, lightgbm as lgb, sklearn
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from baseline_data import sha256, save_json
from baseline_experiments import best_f1_threshold, metrics
from run import NativeCategories, LGB_CONFIG, model_inputs, predict
from optimization_features import build_extension, TREND_PAIRS, STATUS_FIELDS
OUT=ROOT/'unified/optimization_20261007'
TOL=.0003
BLENDS=[dict(tag='raw80_20',method='raw',weights={'LGB':.8,'LR':.2}),
        dict(tag='rank80_20',method='rank',weights={'LGB':.8,'LR':.2}),
        dict(tag='rank70_20_10',method='rank',weights={'LGB':.7,'LR':.2,'RF':.1})]

def sample_weights(y,mode):
    if mode=='unweighted':return None
    y=np.asarray(y,dtype=int);counts=np.bincount(y,minlength=2)
    return np.where(y==1,counts[0]/counts[1],1.)

def select_simple(rows):
    best=max(r['auc_mean'] for r in rows)
    eligible=[r for r in rows if r['auc_mean']>=best-TOL]
    return sorted(eligible,key=lambda r:(r['complexity'],-r['auc_mean'],-r['ap_mean'],r['tag']))[0]

def blend_scores(scores,scheme,references=None):
    values={}
    for name,weight in scheme['weights'].items():
        a=np.asarray(scores[name])
        if scheme['method']=='rank':
            if references is None:a=rankdata(a,method='average')/len(a)
            else:
                ref=np.sort(references[name]);a=(np.searchsorted(ref,a,side='left')+np.searchsorted(ref,a,side='right'))/(2*len(ref))
        values[name]=a*weight
    return sum(values.values())

def initial_configs():
    configs=[]
    for depth,leaves in [(2,31),(4,12),(6,24),(8,48)]:
        for weight in ['balanced','unweighted']:
            configs.append(dict(id=f'C{len(configs):02d}',params={**LGB_CONFIG,'max_depth':depth,'num_leaves':leaves},
                                weight=weight,cap=1000,patience=50,stage='structure'))
    return configs

def refinements(anchor):
    variants=[dict(learning_rate=.05),dict(min_data_in_leaf=100),dict(lambda_l2=5.),
              dict(learning_rate=.05,min_data_in_leaf=100,lambda_l2=5.)]
    return [dict(id=f'C{8+i:02d}',params={**anchor['params'],**v},weight=anchor['weight'],
                 cap=2000 if v.get('learning_rate',.1)==.05 else 1000,
                 patience=100 if v.get('learning_rate',.1)==.05 else 50,stage='refinement')
            for i,v in enumerate(variants)]

def complexity(cfg,family='base'):
    p=cfg['params']
    return [0 if family=='base' else (1 if family in ['T','S'] else 2),p['max_depth'],
            min(p['num_leaves'],2**p['max_depth']),0 if cfg['weight']=='unweighted' else 1,
            0 if p['learning_rate']==.1 else 1,-p['min_data_in_leaf'],p['lambda_l2']]

def freeze():
    paths=[ROOT/f'processed_data/application_{role}_processed_v2.csv' for role in ['train','test']]
    paths += [ROOT/'baseline/splits/membership.csv',ROOT/'home-credit-default-risk/bureau.csv',
              ROOT/'home-credit-default-risk/bureau_balance.csv',ROOT/'home-credit-default-risk/HomeCredit_columns_description.csv']
    paths += [ROOT/'unified/protocol.json',ROOT/'unified/results/model_lock.json',ROOT/'unified/results/oof_predictions.csv']
    paths += [ROOT/'unified'/p for p in ['optimization.py','optimization_features.py','run.py','rf_processing.py']]
    paths += [ROOT/'baseline'/p for p in ['baseline_data.py','baseline_features_v2.py','baseline_experiments.py']]
    protocol=dict(inputs_code={str(p.relative_to(ROOT)):sha256(p) for p in paths},
       runtime=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
                    lightgbm=lgb.__version__,sklearn=sklearn.__version__),
       initial_configs=initial_configs(),refinement_rules=[{'learning_rate':.05},{'min_data_in_leaf':100},
                    {'lambda_l2':5},{'learning_rate':.05,'min_data_in_leaf':100,'lambda_l2':5}],
       refinement_anchor='select_simple over C00-C07, fixed rule; no additions',
       feature_groups=dict(T=TREND_PAIRS,S=STATUS_FIELDS),feature_families=['T','S','TS'],
       feature_configs='deduplicated [structure anchor, final parameter winner]; identical small range for base/T/S/TS',
       fixed_feature_comparison='final parameter winner on base/T/S/TS',blends=BLENDS,
       selection=dict(tolerance=TOL,rule='max mean fold AUC; within tolerance lexicographic simplicity, then AUC/AP/tag',
                      extension_gate='mean gain > tolerance and every fold gain >=0 versus parameter-only winner',
                      blend_gate='mean gain > tolerance and every fold gain >=0 versus selected LGB'),
       early_stopping=dict(inner_fraction=.1,seed=42,stratified=True,metric='unweighted auc',
                           outer_labels_used=False,near_cap='best_iteration >=90% cap OR iterations_run reaches cap'),
       maximum_new_fits=112,cv_fits_max=108,final_fits=4,holdout='previously viewed; lock first; no return to selection',
       diagnostic_threshold='max F1 development OOF; highest threshold tie; no AUC threshold optimization',
       rank_blend='validation-fold percentile ranks; full development OOF empirical CDF for holdout/test; uncalibrated score',
       lr_rf='unchanged V2 original key; verify every OOF fold input/code/runtime/IDs/hash/metric and final model replay',
       final_rounds='separate own-config 10% stratified inner validation on 95%, then on all labels; no mixed medians')
    key=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['key']==key,'Frozen protocol changed'
    else:save_json(path,dict(key=key,frozen_epoch=time.time(),protocol=protocol))
    return key

def load_inputs():
    mem=pd.read_csv(ROOT/'baseline/splits/membership.csv')
    train=pd.read_csv(ROOT/'processed_data/application_train_processed_v2.csv').set_index('SK_ID_CURR').loc[mem.SK_ID_CURR].reset_index()
    test=pd.read_csv(ROOT/'processed_data/application_test_processed_v2.csv')
    assert train.SK_ID_CURR.is_unique and test.SK_ID_CURR.is_unique
    assert np.array_equal(train.TARGET,mem.TARGET)
    assert list(train.drop(columns='TARGET'))==list(test)
    return train,test,mem

def historical_check(train,test,mem):
    old=json.loads((ROOT/'unified/protocol.json').read_text());protocol=old['protocol']
    for group in ['inputs','code']:
        for rel,digest in protocol[group].items():assert sha256(ROOT/rel)==digest,(group,rel)
    current=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
                 sklearn=sklearn.__version__,lightgbm=lgb.__version__)
    assert current==protocol['runtime']
    oof=pd.read_csv(ROOT/'unified/results/oof_predictions.csv');cv=pd.read_csv(ROOT/'unified/results/cv_folds.csv')
    dev=mem[mem.partition.eq('development')].set_index('SK_ID_CURR');hold=mem.partition.eq('holdout')
    # Original script serialized NativeCategories under __main__.
    setattr(sys.modules['__main__'],'NativeCategories',NativeCategories)
    evidence=[]
    for name in ['LR','RF','LGB']:
        p=oof[oof.model.eq(name)&oof.version.eq('v2')].copy()
        assert p.SK_ID_CURR.is_unique and set(p.SK_ID_CURR)==set(dev.index)
        assert np.array_equal(p.TARGET,dev.loc[p.SK_ID_CURR,'TARGET'])
        assert np.array_equal(p.fold,dev.loc[p.SK_ID_CURR,'fold'])
        for fold,q in p.groupby('fold'):
            tag=f'{name}_v2_fold{fold}';meta=json.loads((ROOT/f'unified/results/{tag}_fit.json').read_text())
            assert meta['key']==old['key'] and sha256(ROOT/f'unified/results/{tag}_predictions.csv')==meta['prediction_sha256']
            saved=pd.read_csv(ROOT/f'unified/results/{tag}_predictions.csv').set_index('SK_ID_CURR').loc[q.SK_ID_CURR]
            np.testing.assert_allclose(saved.probability,q.probability,atol=1e-14,rtol=0)
            row=cv[cv.model.eq(name)&cv.version.eq('v2')&cv.fold.eq(fold)].iloc[0]
            assert abs(roc_auc_score(q.TARGET,q.probability)-row.roc_auc)<1e-13
            if name=='LGB':
                ids=pd.read_csv(ROOT/f'unified/results/LGB_v2_fold{fold}_inner_ids.csv')
                assert set(ids.SK_ID_CURR)==set(dev[dev.fold.ne(fold)].index)
        for role in ['development','submission']:
            path=ROOT/f'unified/models/{name}_v2_{role}.joblib';bundle=joblib.load(path)
            assert bundle['key']==old['key'] and bundle['training_scope']==role
            frame=train.loc[hold] if role=='development' else test
            actual=predict(bundle,model_inputs(name,frame))
            if role=='development':saved=pd.read_csv(ROOT/f'unified/results/{name}_holdout_predictions.csv').set_index('SK_ID_CURR').loc[frame.SK_ID_CURR].probability
            else:saved=pd.read_csv(ROOT/f'unified/results/submission_{name}.csv').set_index('SK_ID_CURR').loc[frame.SK_ID_CURR].TARGET
            np.testing.assert_allclose(actual,saved,rtol=0,atol=1e-14)
            evidence.append(dict(model=name,role=role,source_key=old['key'],model_sha256=sha256(path),prediction_replayed=True,retrained_this_round=False))
            del bundle
    save_json(OUT/'results/historical_reuse.json',dict(protocol_inputs_code_runtime_match=True,
               oof_ids_labels_folds_and_fold_hashes_match=True,OOF_origin='run_cv fits dev & folds != validation fold',models=evidence))
    return oof

def prepare_extension(train,test,key,rebuild=False):
    paths=[OUT/f'data/{role}_extension.pkl' for role in ['train','test']];meta=OUT/'data/extension.json'
    if not rebuild and meta.exists():
        m=json.loads(meta.read_text());assert m['key']==key
        for p in paths:assert sha256(p)==m['hashes'][p.name]
        return [pd.read_pickle(p) for p in paths]
    pair,audit=build_extension(ROOT,train,test)
    for frame,path in zip(pair,paths):frame.to_pickle(path)
    save_json(meta,dict(key=key,hashes={p.name:sha256(p) for p in paths},audit=audit,
              no_labels_used=True,original_39_unchanged=True,model_specific='LGB experiment only; common v2 unmodified'))
    return pair

def family_inputs(frame,extra,family):
    x=model_inputs('LGB',frame)
    if family!='base':
        prefixes=tuple('OPT_'+f+'_' for f in family)
        cols=[c for c in extra if c.startswith(prefixes)]
        add=extra.set_index('SK_ID_CURR').loc[frame.SK_ID_CURR,cols].reset_index(drop=True)
        for c in cols:x[c]=add[c].to_numpy()
    return x

def select_rounds(x,y,ids,cfg,tag):
    y=np.asarray(y);ids=np.asarray(ids)
    inner,early=train_test_split(np.arange(len(x)),test_size=.1,stratify=y,random_state=42)
    prep=NativeCategories().fit(x.iloc[inner])
    data=lgb.Dataset(prep.transform(x.iloc[inner]),label=y[inner],weight=sample_weights(y[inner],cfg['weight']))
    valid=lgb.Dataset(prep.transform(x.iloc[early]),label=y[early],reference=data)
    history={}
    model=lgb.train(cfg['params'],data,num_boost_round=cfg['cap'],valid_sets=[valid],valid_names=['inner'],
         callbacks=[lgb.early_stopping(cfg['patience'],first_metric_only=True,verbose=False),lgb.record_evaluation(history)])
    n=len(history['inner']['auc']);best=int(model.best_iteration)
    split=pd.DataFrame({'SK_ID_CURR':ids,'inner_role':np.where(np.isin(np.arange(len(x)),early),'early_stopping','training')})
    split.to_csv(OUT/f'results/{tag}_inner_ids.csv',index=False)
    info=dict(rounds=best,iterations_run=n,cap=cfg['cap'],patience=cfg['patience'],
              possibly_truncated=bool(best>=.9*cfg['cap'] or n>=cfg['cap']),inner_auc=history['inner']['auc'][best-1])
    save_json(OUT/f'results/{tag}_early_stopping.json',dict(**info,history=history))
    return best,info

def fit_fixed(x,y,cfg,rounds):
    prep=NativeCategories().fit(x)
    model=lgb.train(cfg['params'],lgb.Dataset(prep.transform(x),label=y,weight=sample_weights(y,cfg['weight'])),num_boost_round=rounds)
    return prep,model

def candidate(train,test,mem,extension,cfg,family,key,resume):
    tag=cfg['id']+'_'+family;x=family_inputs(train,extension[0],family)
    xt=family_inputs(test,extension[1],family);assert list(x)==list(xt)
    dev=mem.partition.eq('development').to_numpy();folds=mem.fold.to_numpy();rows=[];ps=[]
    for fold in range(3):
        ft=f'{tag}_fold{fold}';path=OUT/f'results/{ft}_predictions.csv';meta=OUT/f'results/{ft}_fit.json'
        tr=dev&(folds!=fold);va=dev&(folds==fold)
        if resume and meta.exists():
            info=json.loads(meta.read_text());assert info['key']==key and info['cfg']==cfg
            assert info['prediction_sha256']==sha256(path);p=pd.read_csv(path)
            assert p.SK_ID_CURR.tolist()==train.loc[va,'SK_ID_CURR'].tolist()
            print('Verified cache',ft,flush=True)
        else:
            started=time.perf_counter();print('Training',ft,flush=True)
            rounds,early=select_rounds(x.loc[tr],train.loc[tr,'TARGET'],train.loc[tr,'SK_ID_CURR'],cfg,ft)
            prep,model=fit_fixed(x.loc[tr],train.loc[tr,'TARGET'],cfg,rounds)
            probability=model.predict(prep.transform(x.loc[va]),num_iteration=rounds,validate_features=True)
            p=train.loc[va,['SK_ID_CURR','TARGET']].copy();p['fold']=fold;p['probability']=probability;p.to_csv(path,index=False)
            info=dict(key=key,cfg=cfg,family=family,seconds=time.perf_counter()-started,fit_count=2,
                      training_rows=int(tr.sum()),validation_rows=int(va.sum()),features=list(x),categories=prep.categories_,
                      prediction_sha256=sha256(path),**early)
            save_json(meta,info);del prep,model
        assert np.array_equal(p.TARGET,train.loc[va,'TARGET'])
        score=dict(tag=tag,config=cfg['id'],family=family,fold=fold,roc_auc=roc_auc_score(p.TARGET,p.probability),
                   ap=average_precision_score(p.TARGET,p.probability),**{k:info[k] for k in ['seconds','fit_count','rounds','cap','iterations_run','possibly_truncated']})
        print(ft,score['roc_auc'],score['ap'],'rounds',score['rounds'],'truncated?',score['possibly_truncated'],flush=True)
        rows.append(score);ps.append(p)
    summary=dict(tag=tag,config=cfg['id'],family=family,auc_mean=float(np.mean([r['roc_auc'] for r in rows])),
                 auc_sd=float(np.std([r['roc_auc'] for r in rows],ddof=1)),ap_mean=float(np.mean([r['ap'] for r in rows])),
                 ap_sd=float(np.std([r['ap'] for r in rows],ddof=1)),seconds=sum(r['seconds'] for r in rows),
                 complexity=complexity(cfg,family),cfg=cfg,possibly_truncated=any(r['possibly_truncated'] for r in rows))
    return summary,rows,pd.concat(ps,ignore_index=True)

def oof_for(tag):
    return pd.concat([pd.read_csv(OUT/f'results/{tag}_fold{k}_predictions.csv') for k in range(3)],ignore_index=True)

def gains(new,old):
    return [float(a-b) for a,b in zip(new,old)]

def fold_auc(p):return [roc_auc_score(q.TARGET,q.probability) for _,q in p.groupby('fold',sort=True)]

def run_search(train,test,mem,extra,key,resume,old_oof):
    summaries=[];rows=[];configs=initial_configs()
    for cfg in configs:
        s,r,p=candidate(train,test,mem,extra,cfg,'base',key,resume);summaries.append(s);rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT/'results/cv_folds.csv',index=False)
    anchor=select_simple(summaries);configs+=refinements(anchor['cfg'])
    save_json(OUT/'results/materialized_configs.json',configs)
    for cfg in configs[8:]:
        s,r,p=candidate(train,test,mem,extra,cfg,'base',key,resume);summaries.append(s);rows.extend(r)
        pd.DataFrame(rows).to_csv(OUT/'results/cv_folds.csv',index=False)
    param=select_simple(summaries)
    small={s['config']:s['cfg'] for s in [anchor,param]}
    save_json(OUT/'results/feature_candidate_range.json',dict(configs=list(small),fixed_config=param['config'],rule='structure anchor and parameter winner'))
    for family in ['T','S','TS']:
        for cfg in small.values():
            s,r,p=candidate(train,test,mem,extra,cfg,family,key,resume);summaries.append(s);rows.extend(r)
            pd.DataFrame(rows).to_csv(OUT/'results/cv_folds.csv',index=False)
    save_json(OUT/'results/cv_summary.json',summaries)
    pd.DataFrame([{k:v for k,v in s.items() if k not in ['cfg','complexity']} for s in summaries]).to_csv(OUT/'results/cv_summary.csv',index=False)
    best=select_simple(summaries)
    base_p=oof_for(param['tag']);best_p=oof_for(best['tag']);delta=gains(fold_auc(best_p),fold_auc(base_p))
    if best['family']!='base' and not (best['auc_mean']-param['auc_mean']>TOL and min(delta)>=0):best=param;best_p=base_p
    pair=[]
    for family in ['T','S','TS']:
        fixed=next(s for s in summaries if s['family']==family and s['config']==param['config'])
        optimized=select_simple([s for s in summaries if s['family']==family and s['config'] in small])
        base_small=select_simple([s for s in summaries if s['family']=='base' and s['config'] in small])
        for kind,new,old in [('fixed_config',fixed,param),('same_small_range_optimized',optimized,base_small)]:
            a=oof_for(new['tag']);b=oof_for(old['tag'])
            for k,(auc,oldauc) in enumerate(zip(fold_auc(a),fold_auc(b))):
                qa=a[a.fold.eq(k)];qb=b[b.fold.eq(k)]
                pair.append(dict(kind=kind,family=family,new_tag=new['tag'],reference_tag=old['tag'],fold=k,
                    auc_delta=auc-oldauc,ap_delta=average_precision_score(qa.TARGET,qa.probability)-average_precision_score(qb.TARGET,qb.probability)))
    pd.DataFrame(pair).to_csv(OUT/'results/feature_paired_deltas.csv',index=False)
    sources={'LGB':best_p}
    for name in ['LR','RF']:sources[name]=old_oof[old_oof.model.eq(name)&old_oof.version.eq('v2')][['SK_ID_CURR','TARGET','fold','probability']]
    aligned=best_p[['SK_ID_CURR','TARGET','fold']].copy()
    for name,p in sources.items():
        p=p.set_index('SK_ID_CURR').loc[aligned.SK_ID_CURR]
        assert np.array_equal(p.TARGET,aligned.TARGET) and np.array_equal(p.fold,aligned.fold)
        aligned[name]=p.probability.to_numpy()
    aligned.to_csv(OUT/'results/aligned_base_oof.csv',index=False)
    blend_rows=[];blend_summary=[];blend_predictions={}
    for scheme in BLENDS:
        p=aligned[['SK_ID_CURR','TARGET','fold']].copy();prob=np.empty(len(p))
        for fold in range(3):
            mask=p.fold.eq(fold).to_numpy();scores={name:aligned.loc[mask,name].to_numpy() for name in scheme['weights']}
            prob[mask]=blend_scores(scores,scheme)
            blend_rows.append(dict(tag=scheme['tag'],fold=fold,roc_auc=roc_auc_score(p.loc[mask,'TARGET'],prob[mask]),
                                  ap=average_precision_score(p.loc[mask,'TARGET'],prob[mask])))
        p['probability']=prob;p.to_csv(OUT/f'results/blend_{scheme["tag"]}_oof.csv',index=False)
        q=[r for r in blend_rows if r['tag']==scheme['tag']];d=gains([r['roc_auc'] for r in q],fold_auc(best_p))
        blend_summary.append(dict(tag=scheme['tag'],auc_mean=float(np.mean([r['roc_auc'] for r in q])),
              auc_sd=float(np.std([r['roc_auc'] for r in q],ddof=1)),ap_mean=float(np.mean([r['ap'] for r in q])),
              ap_sd=float(np.std([r['ap'] for r in q],ddof=1)),fold_deltas=d,
              stable_gain=bool(min(d)>=0 and np.mean(d)>TOL),complexity=[len(scheme['weights']),0 if scheme['method']=='raw' else 1],scheme=scheme))
        blend_predictions[scheme['tag']]=p
    pd.DataFrame(blend_rows).to_csv(OUT/'results/blend_folds.csv',index=False)
    save_json(OUT/'results/blend_summary.json',blend_summary)
    eligible=[s for s in blend_summary if s['stable_gain']]
    blend=select_simple(eligible) if eligible else None
    final_p=blend_predictions[blend['tag']] if blend else best_p
    threshold,_=best_f1_threshold(final_p.TARGET,final_p.probability)
    final_p.to_csv(OUT/'results/final_oof.csv',index=False)
    old=old_oof[old_oof.model.eq('LGB')&old_oof.version.eq('v2')]
    stages=[]
    for label,p,reference in [('parameters',oof_for(param['tag']),old),('features',best_p,oof_for(param['tag'])),('blend',final_p,best_p),('complete',final_p,old)]:
        for fold in range(3):
            a=p[p.fold.eq(fold)];b=reference[reference.fold.eq(fold)]
            stages.append(dict(stage=label,fold=fold,roc_auc=roc_auc_score(a.TARGET,a.probability),
                ap=average_precision_score(a.TARGET,a.probability),auc_delta=roc_auc_score(a.TARGET,a.probability)-roc_auc_score(b.TARGET,b.probability),
                ap_delta=average_precision_score(a.TARGET,a.probability)-average_precision_score(b.TARGET,b.probability)))
    pd.DataFrame(stages).to_csv(OUT/'results/stage_deltas.csv',index=False)
    lock=dict(key=key,locked_epoch=time.time(),structure_anchor=anchor,parameter_winner=param,LGB=best,blend=blend,
              final_name='Fusion' if blend else 'LGB_optimized',diagnostic_threshold=threshold,
              selection_scope='95% development fold-mean AUC; bounded development selection, not nested unbiased evaluation',
              holdout_previously_viewed=True,pooled_oof_auc=roc_auc_score(final_p.TARGET,final_p.probability),
              final_fold_auc=fold_auc(final_p),feature_gate_passed=best['family']!='base')
    path=OUT/'results/model_lock.json'
    if path.exists():
        prior=json.loads(path.read_text());lock['locked_epoch']=prior['locked_epoch'];assert prior==lock
    else:save_json(path,lock)
    return lock

def final_fits(train,test,mem,extra,lock,key,resume):
    chosen=lock['LGB'];cfg=chosen['cfg'];family=chosen['family']
    x=family_inputs(train,extra[0],family);xt=family_inputs(test,extra[1],family)
    dev=mem.partition.eq('development').to_numpy();hold=~dev;probs={};fit_records=[]
    for role,mask in [('development',dev),('submission',np.ones(len(train),dtype=bool))]:
        tag='LGB_'+role;path=OUT/f'models/{tag}.joblib'
        if resume and path.exists():
            b=joblib.load(path);assert b['key']==key and b['cfg']==cfg and b['family']==family
        else:
            start=time.perf_counter();rounds,info=select_rounds(x.loc[mask],train.loc[mask,'TARGET'],train.loc[mask,'SK_ID_CURR'],cfg,tag)
            prep,model=fit_fixed(x.loc[mask],train.loc[mask,'TARGET'],cfg,rounds)
            b=dict(preprocessor=prep,model=model,rounds=rounds,cfg=cfg,family=family,key=key,
                   training_scope=role,training_ids_sha256=hashlib.sha256(train.loc[mask,'SK_ID_CURR'].to_numpy().tobytes()).hexdigest(),
                   features=list(x),info=dict(**info,seconds=time.perf_counter()-start,fit_count=2))
            joblib.dump(b,path,compress=3)
        frame=x.loc[hold] if role=='development' else xt
        probs[role]=b['model'].predict(b['preprocessor'].transform(frame),num_iteration=b['rounds'],validate_features=True)
        fit_records.append(dict(role=role,model_sha256=sha256(path),**b['info']))
    save_json(OUT/'results/final_fit_records.json',fit_records)
    references=pd.read_csv(OUT/'results/aligned_base_oof.csv')
    hm=[]
    for role in ['development','submission']:
        ids=train.loc[hold,'SK_ID_CURR'] if role=='development' else test.SK_ID_CURR
        scores={'LGB':probs[role]}
        for name in ['LR','RF']:
            if role=='development':p=pd.read_csv(ROOT/f'unified/results/{name}_holdout_predictions.csv').set_index('SK_ID_CURR');scores[name]=p.loc[ids,'probability'].to_numpy()
            else:p=pd.read_csv(ROOT/f'unified/results/submission_{name}.csv').set_index('SK_ID_CURR');scores[name]=p.loc[ids,'TARGET'].to_numpy()
        scheme=lock['blend']['scheme'] if lock['blend'] else None
        final=blend_scores(scores,scheme,{n:references[n].to_numpy() for n in scheme['weights']}) if scheme else scores['LGB']
        variants={'LGB_optimized':scores['LGB']}
        if scheme:variants['Fusion']=final
        for name,values in variants.items():
            if role=='development':
                p=pd.DataFrame({'SK_ID_CURR':ids.to_numpy(),'TARGET':train.loc[hold,'TARGET'].to_numpy(),'probability':values});p.to_csv(OUT/f'results/{name}_holdout_predictions.csv',index=False)
                t=lock['diagnostic_threshold'] if name==lock['final_name'] else best_f1_threshold(oof_for(chosen['tag']).TARGET,oof_for(chosen['tag']).probability)[0]
                for threshold in [.5,t]:hm.append(dict(model=name,threshold=threshold,scope='previously viewed 5% supplementary; locked before scoring',**metrics(p.TARGET,values,threshold)))
            else:
                sample=pd.read_csv(ROOT/'home-credit-default-risk/sample_submission.csv')
                sub=sample[['SK_ID_CURR']].merge(pd.DataFrame({'SK_ID_CURR':ids.to_numpy(),'TARGET':values}),on='SK_ID_CURR',how='left',validate='one_to_one')
                sub.to_csv(OUT/f'results/submission_{name}.csv',index=False)
    # Reuse controls explicitly under original names and hashes; original three files remain in place.
    pd.DataFrame(hm).to_csv(OUT/'results/holdout_metrics.csv',index=False)
    save_json(OUT/'results/submission_provenance.json',dict(original_submissions_preserved=True,
       controls={n:dict(path=f'unified/results/submission_{n}.csv',sha256=sha256(ROOT/f'unified/results/submission_{n}.csv'),source_key=json.loads((ROOT/'unified/protocol.json').read_text())['key'],retrained_this_round=False) for n in ['LR','RF']},
       optimized_LGB='separate full-labelled fit; rounds from own inner validation',fusion=lock['blend'],uncalibrated=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--validate-only',action='store_true');parser.add_argument('--rebuild-features',action='store_true');args=parser.parse_args()
    key=freeze();train,test,mem=load_inputs();old=historical_check(train,test,mem)
    extra=prepare_extension(train,test,key,args.rebuild_features)
    if not args.validate_only:
        lock=run_search(train,test,mem,extra,key,args.resume,old)
        final_fits(train,test,mem,extra,lock,key,args.resume)
    print('Experiments and final fits complete; reporting/verification entry is optimization_delivery.py',flush=True)

if __name__=='__main__':main()
