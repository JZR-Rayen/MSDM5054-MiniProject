"""Explicit raw-data reconstruction and frozen training in NEW output directories.

Never called by default Notebook execution. No holdout scoring, final submission
models, additional candidates or changed membership. See README.md.
"""
from pathlib import Path
import argparse,json,sys,time
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE),str(BASE/'source_snapshot/unified/abc_comparison'),
               str(BASE/'source_snapshot/unified'),str(BASE/'source_snapshot/baseline')]

def fresh(path):
    path=Path(path).resolve()
    if path.exists():raise FileExistsError('Use a new output directory; existing evidence is preserved')
    path.mkdir(parents=True);(path/'results').mkdir()
    return path

def rebuild(data_root,output):
    import workflow as w
    import pandas as pd,numpy as np
    from baseline_data import AGGREGATIONS,TIME_COLUMNS,aggregate,derive,check_time_audit,make_splits
    from baseline_features_v2 import build_histories,attach_histories
    w.verify_manifest();w.verify_frozen_source()
    root=Path(data_root).resolve();raw=root/'home-credit-default-risk';out=fresh(output);start=time.perf_counter()
    fingerprints=json.loads((BASE/'support/data/raw_fingerprints.json').read_text())
    for r in fingerprints:
        if r['file'].startswith('home-credit-default-risk/'):
            assert w.sha(root/r['file'])==r['sha256'],r['file']
    train=pd.read_csv(raw/'application_train.csv');test=pd.read_csv(raw/'application_test.csv')
    a=derive(train);time_rows=[]
    for name,spec in AGGREGATIONS.items():
        cols={c for c,_ in spec.values() if not c.startswith('_')}|{'SK_ID_CURR'}|set(TIME_COLUMNS[name])
        if name=='bureau.csv':cols.add('CREDIT_ACTIVE')
        if name=='previous_application.csv':cols.add('NAME_CONTRACT_STATUS')
        frame=pd.read_csv(raw/name,usecols=sorted(cols))
        if name=='bureau.csv':
            assert int(frame.DAYS_CREDIT_UPDATE.gt(0).sum())==17
            frame=frame.loc[~frame.DAYS_CREDIT_UPDATE.gt(0)]
        for col in TIME_COLUMNS[name]:time_rows.append(dict(source=name,field=col,positive_records=int(frame[col].gt(0).sum())))
        agg=aggregate(frame,name)
        train=train.merge(agg,on='SK_ID_CURR',how='left',sort=False,validate='one_to_one')
        test=test.merge(agg,on='SK_ID_CURR',how='left',sort=False,validate='one_to_one')
        print('Rebuilt',name,flush=True)
        del frame,agg
    check_time_audit(time_rows)
    b,bt=derive(train),derive(test)
    h,_=build_histories(root,out);c,ct=attach_histories(b,bt,h);del h,train,test
    schemas=json.loads((BASE/'support/abc/feature_sets.json').read_text())['groups']
    mem=w.read_csv('support/membership.csv');pd.testing.assert_frame_equal(make_splits(b),mem)
    frames={'A':a,'B':b,'C':c};data=out/'data';data.mkdir();manifest={}
    for group,frame in frames.items():
        frame=frame.set_index('SK_ID_CURR').loc[mem.SK_ID_CURR].reset_index()[schemas[group]['columns']]
        assert np.array_equal(frame.TARGET,mem.TARGET) and len(frame)==307511
        path=data/f'{group}.csv';frame.to_csv(path,index=False);manifest[path.name]=w.sha(path)
    ct.to_csv(data/'C_test.csv',index=False);manifest['C_test.csv']=w.sha(data/'C_test.csv')
    mem.to_csv(data/'membership.csv',index=False);manifest['membership.csv']=w.sha(data/'membership.csv')
    pd.DataFrame(time_rows).to_csv(out/'results/time_audit.csv',index=False)
    (out/'reconstruction.json').write_text(json.dumps(dict(files=manifest,raw_fingerprints=fingerprints,
        data_root=str(root),source_manifest_sha256=w.sha(BASE/'support/manifest.json'),
        counts={g:len(f.columns) for g,f in frames.items()},applicants=307511,
        membership_reconstructed_and_matched=True,history_applicants_retained=True,
        new_model_fits=0,seconds=time.perf_counter()-start),indent=2))
    print('Raw reconstruction completed',out,flush=True)
    return out

def prepared(path):
    import pandas as pd
    import workflow as w
    w.verify_manifest();w.verify_frozen_source()
    p=Path(path).resolve();record=json.loads((p/'reconstruction.json').read_text())
    assert record['source_manifest_sha256']==w.sha(BASE/'support/manifest.json'),'Source/support changed since reconstruction'
    for name,digest in record['files'].items():assert w.sha(p/'data'/name)==digest
    return {g:pd.read_csv(p/f'data/{g}.csv') for g in 'ABC'},pd.read_csv(p/'data/membership.csv')

def abc(inputs,output,groups,models,folds):
    from train_fixed import fit_frames
    frames,mem=prepared(inputs)
    fit_frames(frames,mem,Path(output).resolve(),groups,models,folds)

def secondary(inputs,output,tags,abc_results=None):
    import pandas as pd
    import workflow as w
    import optimization as opt
    from optimization_features import build_extension
    frames,mem=prepared(inputs);source=Path(inputs).resolve();out=fresh(output);opt.OUT=out
    train=frames['C'];test=pd.read_csv(source/'data/C_test.csv')
    config={c['id']:c for c in json.loads((BASE/'support/optimization/materialized_configs.json').read_text())}
    permitted={c+'_base' for c in config}|{c+'_'+f for c in ['C01','C10'] for f in ['T','S','TS']}
    tags=sorted(permitted) if tags==['all'] else tags
    if not set(tags)<=permitted:raise ValueError('Only frozen candidate tags are permitted')
    if any(not t.endswith('_base') for t in tags):
        raw_root=json.loads((source/'reconstruction.json').read_text())['data_root']
        extra,_=build_extension(Path(raw_root),train,test)
    else:extra=[pd.DataFrame({'SK_ID_CURR':train.SK_ID_CURR}),pd.DataFrame({'SK_ID_CURR':test.SK_ID_CURR})]
    rows=[]
    for tag in tags:
        cfg,family=tag.split('_');_,fold_rows,_=opt.candidate(train,test,mem,extra,config[cfg],family,'isolated-frozen-reproduction',False)
        rows.extend(fold_rows)
    pd.DataFrame(rows).to_csv(out/'results/cv_folds.csv',index=False)
    if 'C11_base' in tags:
        # Fusion is arithmetic, using the fixed C11 and authenticated historical LR/RF OOF.
        # Historical reuse is explicit; --abc-results supplies fresh fixed-CV scores.
        abc_oof=(pd.read_csv(abc_results,float_precision='round_trip') if abc_results else w.read_csv('support/abc/oof_predictions.csv.gz'))
        lgb=pd.concat([pd.read_csv(out/f'results/C11_base_fold{k}_predictions.csv',float_precision='round_trip') for k in range(3)],ignore_index=True)
        for scheme in opt.BLENDS:
            result=lgb[['SK_ID_CURR','TARGET','fold']].copy();values=[]
            for fold,q in lgb.groupby('fold',sort=True):
                scores={'LGB':q.probability.to_numpy()}
                for model in ['LR','RF']:
                    ref=abc_oof[abc_oof.model.eq(model)&abc_oof.group.eq('C')].set_index('SK_ID_CURR').loc[q.SK_ID_CURR]
                    assert (ref.TARGET.to_numpy()==q.TARGET.to_numpy()).all() and (ref.fold==fold).all()
                    scores[model]=ref.probability.to_numpy()
                values.extend(opt.blend_scores(scores,scheme))
            result['probability']=values;result.to_csv(out/f'results/blend_{scheme["tag"]}_oof.csv',index=False)
    (out/'results/training_scope.json').write_text(json.dumps(dict(tags=tags,model_fits=sum(r['fit_count'] for r in rows),
        selection_changed=False,holdout_scored=False,final_models=False,
        fusion_lr_rf_source=str(abc_results) if abc_results else 'authenticated historical C OOF; not fresh LR/RF training'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['data','abc','secondary'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data-root',type=Path)
    p.add_argument('--inputs',type=Path);p.add_argument('--abc-results',type=Path);p.add_argument('--groups',nargs='+',choices=list('ABC'),default=list('ABC'))
    p.add_argument('--models',nargs='+',choices=['LR','RF','LGB'],default=['LR','RF','LGB'])
    p.add_argument('--folds',nargs='+',type=int,choices=[0,1,2],default=[0,1,2]);p.add_argument('--tags',nargs='+',default=['all'])
    a=p.parse_args()
    if a.stage=='data':
        if a.data_root is None:p.error('--data-root is required for data')
        rebuild(a.data_root,a.output)
    else:
        if a.inputs is None:p.error('--inputs is required for training')
        if a.stage=='abc':abc(a.inputs,a.output,a.groups,a.models,a.folds)
        else:secondary(a.inputs,a.output,a.tags,a.abc_results)
