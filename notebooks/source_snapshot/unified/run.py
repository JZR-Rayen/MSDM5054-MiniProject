"""Frozen six-run comparison; no historical search or holdout tuning."""
from pathlib import Path
import argparse, hashlib, json, os, platform, sys, time
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'unified/runtime/matplotlib'))
sys.path.insert(0, str(ROOT/'baseline'))
sys.path.insert(0, str(ROOT/'unified'))
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from baseline_data import correct_fields, sha256, save_json
from baseline_features_v2 import ExtensionPreprocessor
from baseline_experiments import fit_model, best_f1_threshold, metrics
from rf_processing import f3_features, build_feature_preprocessor

OUT = ROOT/'unified'
LR_CONFIG = dict(penalty='elasticnet', C=.01, l1_ratio=.25, class_weight='balanced')
RF_CONFIG = dict(n_estimators=300, max_depth=15, min_samples_leaf=50,
                 max_features=.10, class_weight={0:1,1:2}, criterion='gini',
                 bootstrap=True, random_state=42, n_jobs=4)
LGB_CONFIG = dict(objective='binary', metric='auc', learning_rate=.1, max_depth=2,
                  num_leaves=31, min_data_in_leaf=20, min_gain_to_split=.01,
                  seed=42, num_threads=4, verbosity=-1, zero_as_missing=False,
                  use_missing=True, deterministic=True, force_col_wise=True,
                  feature_fraction=1., bagging_fraction=1., bagging_freq=0,
                  lambda_l1=0., lambda_l2=0., max_bin=255)

def freeze_protocol():
    inputs=[ROOT/'processed_data'/f'application_{role}_processed_{v}.csv'
            for v in ['v1','v2'] for role in ['train','test']]
    inputs += [ROOT/'baseline/splits/membership.csv']
    source=[Path(__file__),OUT/'rf_processing.py',ROOT/'baseline/baseline_data.py',
            ROOT/'baseline/baseline_features_v2.py',ROOT/'baseline/baseline_experiments.py']
    runtime=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                 pandas=pd.__version__,sklearn=sklearn.__version__,lightgbm=lgb.__version__)
    protocol=dict(inputs={str(p.relative_to(ROOT)):sha256(p) for p in inputs},
                  code={str(p.relative_to(ROOT)):sha256(p) for p in source}, runtime=runtime,
                  lr=LR_CONFIG,rf=RF_CONFIG,lgb=LGB_CONFIG,
                  early_stopping=dict(inner_fraction=.1,seed=42,max_rounds=1000,patience=50,
                                      metric='unweighted AUC',outer_validation_used=False),
                  iv=False,threshold='max OOF F1; exact tie highest score; p>=t',
                  selection='mean fold AUC, exact tie mean AP, exact tie v1',
                  old_lr_reuse=False,rf_missing='all historical non-count fields plus each new39; fixed schema',
                  holdout='previously viewed supplementary evaluation; never selection')
    key=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    path=OUT/'protocol.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['key']!=key: raise ValueError('Protocol/input/environment changed; use a separate result directory')
    else: save_json(path,dict(key=key,protocol=protocol,frozen_at=time.time()))
    return key

def load_data():
    membership=pd.read_csv(ROOT/'baseline/splits/membership.csv').set_index('SK_ID_CURR')
    data={}; new=[]
    for version in ['v1','v2']:
        train=pd.read_csv(ROOT/f'processed_data/application_train_processed_{version}.csv')
        test=pd.read_csv(ROOT/f'processed_data/application_test_processed_{version}.csv')
        assert train.SK_ID_CURR.is_unique and test.SK_ID_CURR.is_unique
        assert not set(train.SK_ID_CURR)&set(test.SK_ID_CURR)
        assert set(train.SK_ID_CURR)==set(membership.index)
        train=train.set_index('SK_ID_CURR').loc[membership.index].reset_index()
        assert np.array_equal(train.TARGET,membership.TARGET)
        assert train.drop(columns='TARGET').columns.tolist()==test.columns.tolist()
        data[version]=(train,test)
    a,b=data['v1'][0],data['v2'][0]
    new=[c for c in b if c not in a]; assert len(new)==39
    for role in [0,1]:
        a,b=data['v1'][role],data['v2'][role]
        b=b.set_index('SK_ID_CURR').loc[a.SK_ID_CURR].reset_index()
        pd.testing.assert_frame_equal(a,b[a.columns],check_exact=False,rtol=1e-12,atol=1e-10)
    save_json(OUT/'results/data_contract.json',dict(new39=new,base_equal=True,
              shapes={v:[list(t.shape),list(s.shape)] for v,(t,s) in data.items()},
              train_test_schema_equal=True))
    return data,membership.reset_index(),new

class NativeCategories:
    def fit(self,x):
        self.columns_=list(x);self.categories_={c:sorted(x[c].dropna().unique().tolist())
            for c in x.select_dtypes(include=['object','string','category'])}
        return self
    def transform(self,x):
        if list(x)!=self.columns_:raise ValueError('LGB schema mismatch')
        x=x.copy()
        for c,levels in self.categories_.items():x[c]=pd.Categorical(x[c],categories=levels)
        return x
    def get_feature_names_out(self):return np.array(self.columns_)


def model_inputs(name,frame):
    if name=='RF':return f3_features(frame)
    return correct_fields(frame)

def fit_lgb_fixed(x,y,rounds,params=None):
    prep=NativeCategories().fit(x);z=prep.transform(x)
    counts=np.bincount(np.asarray(y,dtype=int),minlength=2)
    weights=np.where(np.asarray(y)==1,counts[0]/counts[1],1.)
    model=lgb.train(params or LGB_CONFIG,lgb.Dataset(z,label=y,weight=weights),num_boost_round=int(rounds))
    return prep,model

def select_lgb_rounds(x,y,ids,tag,params=None):
    inner,early=train_test_split(np.arange(len(x)),test_size=.1,stratify=y,random_state=42)
    prep=NativeCategories().fit(x.iloc[inner])
    counts=np.bincount(np.asarray(y)[inner],minlength=2)
    weights=np.where(np.asarray(y)[inner]==1,counts[0]/counts[1],1.)
    train=lgb.Dataset(prep.transform(x.iloc[inner]),label=np.asarray(y)[inner],weight=weights)
    valid=lgb.Dataset(prep.transform(x.iloc[early]),label=np.asarray(y)[early],reference=train)
    model=lgb.train(params or LGB_CONFIG,train,num_boost_round=1000,valid_sets=[valid],
                    valid_names=['inner_early_stopping'],
                    callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=False)])
    split=pd.DataFrame({'SK_ID_CURR':np.asarray(ids),
                        'inner_role':np.where(np.isin(np.arange(len(x)),early),'early_stopping','training')})
    split.to_csv(OUT/f'results/{tag}_inner_ids.csv',index=False)
    return int(model.best_iteration)

def fit(name,version,x,y,ids,tag,rounds=None):
    started=time.perf_counter();info={}
    if name=='LR':
        prep=ExtensionPreprocessor(['N'] if version=='v1' else ['B','H','N','R'])
        z=prep.fit_transform(x);model,info=fit_model(z,np.asarray(y),LR_CONFIG)
    elif name=='RF':
        prep,_,_=build_feature_preprocessor(x);z=prep.fit_transform(x)
        model=RandomForestClassifier(**RF_CONFIG).fit(z,y)
    else:
        if rounds is None:rounds=select_lgb_rounds(x,np.asarray(y),ids,tag)
        prep,model=fit_lgb_fixed(x,y,rounds);info['rounds']=int(rounds)
    info.update(seconds=time.perf_counter()-started,training_rows=len(x),tag=tag,
                fitting_origin='new_training',fit_count=2 if name=='LGB' and 'fold' in tag else 1)
    return dict(preprocessor=prep,model=model,model_name=name,version=version,info=info)

def predict(bundle,x):
    z=bundle['preprocessor'].transform(x)
    if bundle['model_name']=='LGB':return bundle['model'].predict(z,num_iteration=bundle['info']['rounds'],validate_features=True)
    return bundle['model'].predict_proba(z)[:,1]

def feature_audit(bundle,x,validation,test,new39,tag):
    prep=bundle['preprocessor']; names=list(prep.get_feature_names_out())
    assert list(x)==list(validation)==list(test)
    widths=[prep.transform(f.iloc[:3]).shape[1] for f in [x,validation,test]]
    assert len(set(widths))==1 and widths[0]==len(names)
    present=[c for c in new39 if c in x]
    if bundle['version']=='v2':
        assert len(present)==39
        for c in present:assert c in names or 'numeric__'+c in names
    numeric=x.select_dtypes(include='number').columns.tolist()
    assert all(c in numeric for c in present)
    flags=[c for c in x if c.endswith('_MISSING')]
    if bundle['model_name']=='RF' and bundle['version']=='v2':
        assert all(c+'_MISSING' in flags for c in new39)
    record=dict(before=list(x),after=names,numeric=numeric,new39_entered=present,
                missing_flags=flags,train_validation_test_widths=widths,
                same_input_schema=True,info=bundle['info'])
    if bundle['model_name']=='LGB':record.update(nan_preserved=True,zero_as_missing=False,categories=prep.categories_)
    if bundle['model_name']=='RF':
        imputer=prep.named_transformers_['numeric'].named_steps['median_imputer']
        record['training_medians']=dict(zip(numeric,imputer.statistics_.tolist()))
    if bundle['model_name']=='LR':record['preprocessing']=prep.metadata()
    save_json(OUT/f'results/{tag}_features.json',record)

def run_cv(data,membership,new39,key,resume):
    dev=membership.partition.eq('development').to_numpy(); folds=membership.fold.to_numpy()
    rows=[];predictions=[]
    for name in ['LR','RF','LGB']:
        for version in ['v1','v2']:
            train,test=data[version];x=model_inputs(name,train);xt=model_inputs(name,test)
            for fold in range(3):
                tag=f'{name}_{version}_fold{fold}';path=OUT/f'results/{tag}_predictions.csv'
                meta=OUT/f'results/{tag}_fit.json'; tr=dev&(folds!=fold);va=dev&(folds==fold)
                if resume and path.exists() and meta.exists():
                    info=json.loads(meta.read_text()); assert info['key']==key and info['prediction_sha256']==sha256(path)
                    p=pd.read_csv(path);assert np.array_equal(p.SK_ID_CURR,train.loc[va,'SK_ID_CURR'])
                    print('Resume verified',tag,flush=True)
                else:
                    print('Training',tag,flush=True)
                    bundle=fit(name,version,x.loc[tr],train.loc[tr,'TARGET'],train.loc[tr,'SK_ID_CURR'],tag)
                    prob=predict(bundle,x.loc[va]);p=train.loc[va,['SK_ID_CURR','TARGET']].copy();p['probability']=prob
                    p.to_csv(path,index=False)
                    feature_audit(bundle,x.loc[tr],x.loc[va],xt,new39,tag)
                    info=dict(key=key,info=bundle['info'],prediction_sha256=sha256(path))
                    save_json(meta,info)
                    del bundle
                score=dict(model=name,version=version,fold=fold,roc_auc=roc_auc_score(p.TARGET,p.probability),
                           ap=average_precision_score(p.TARGET,p.probability),**info['info'])
                rows.append(score);p['model']=name;p['version']=version;p['fold']=fold;predictions.append(p)
                print(tag,score['roc_auc'],score['ap'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'results/cv_folds.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'results/oof_predictions.csv',index=False)

def lock_models(key):
    cv=pd.read_csv(OUT/'results/cv_folds.csv');oof=pd.read_csv(OUT/'results/oof_predictions.csv')
    summary=cv.groupby(['model','version']).agg(auc_mean=('roc_auc','mean'),auc_sd=('roc_auc','std'),
             ap_mean=('ap','mean'),ap_sd=('ap','std')).reset_index()
    summary.to_csv(OUT/'results/cv_summary.csv',index=False)
    paired=cv.pivot(index=['model','fold'],columns='version',values=['roc_auc','ap'])
    delta=pd.DataFrame({'auc_delta':paired['roc_auc']['v2']-paired['roc_auc']['v1'],
                       'ap_delta':paired['ap']['v2']-paired['ap']['v1']}).reset_index()
    delta.to_csv(OUT/'results/paired_deltas.csv',index=False)
    locks=[];thresholds=[]
    for name in ['LR','RF','LGB']:
        winner=summary[summary.model.eq(name)].sort_values(['auc_mean','ap_mean','version'],ascending=[False,False,True]).iloc[0]
        for version in ['v1','v2']:
            p=oof[oof.model.eq(name)&oof.version.eq(version)]; threshold,f1=best_f1_threshold(p.TARGET,p.probability)
            for t in [.5,threshold]:thresholds.append(dict(model=name,version=version,threshold=t,
                scope='development OOF threshold selection',**metrics(p.TARGET,p.probability,t)))
            if version==winner.version:
                rounds=None
                if name=='LGB':rounds=int(np.median(cv[cv.model.eq(name)&cv.version.eq(version)].rounds))
                locks.append(dict(model=name,version=version,threshold=threshold,auc_mean=winner.auc_mean,
                    ap_mean=winner.ap_mean,rounds=rounds))
    pd.DataFrame(thresholds).to_csv(OUT/'results/oof_threshold_metrics.csv',index=False)
    lock=dict(key=key,models=locks,selection_data='95% development CV only',holdout_previously_viewed=True)
    path=OUT/'results/model_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:save_json(path,lock)
    return lock

def final_fits(data,membership,new39,lock,resume):
    dev=membership.partition.eq('development').to_numpy();hold=~dev
    holdrows=[];subchecks=[]
    for cfg in lock['models']:
        name,version=cfg['model'],cfg['version'];train,test=data[version]
        x=model_inputs(name,train);xt=model_inputs(name,test)
        for role,mask in [('development',dev),('submission',np.ones(len(train),dtype=bool))]:
            tag=f'{name}_{version}_{role}';path=OUT/f'models/{tag}.joblib'
            if resume and path.exists():
                bundle=joblib.load(path);assert bundle['key']==lock['key'];print('Resume verified',tag,flush=True)
            else:
                print('Training',tag,flush=True)
                bundle=fit(name,version,x.loc[mask],train.loc[mask,'TARGET'],train.loc[mask,'SK_ID_CURR'],tag,cfg['rounds'])
                bundle['key']=lock['key'];bundle['training_scope']=role
                feature_audit(bundle,x.loc[mask],x.loc[hold],xt,new39,tag);joblib.dump(bundle,path,compress=3)
            if role=='development':
                prob=predict(bundle,x.loc[hold]);p=train.loc[hold,['SK_ID_CURR','TARGET']].copy();p['probability']=prob
                p.to_csv(OUT/f'results/{name}_holdout_predictions.csv',index=False)
                for t in [.5,cfg['threshold']]:holdrows.append(dict(model=name,version=version,threshold=t,
                    scope='previously viewed 5% supplementary holdout',**metrics(p.TARGET,prob,t)))
            else:
                sample=pd.read_csv(ROOT/'home-credit-default-risk/sample_submission.csv')
                prob=predict(bundle,xt);sub=sample[['SK_ID_CURR']].merge(pd.DataFrame({'SK_ID_CURR':test.SK_ID_CURR,'TARGET':prob}),
                        on='SK_ID_CURR',validate='one_to_one',how='left')
                assert sub.SK_ID_CURR.equals(sample.SK_ID_CURR) and sub.TARGET.between(0,1).all() and np.isfinite(sub.TARGET).all()
                sub.to_csv(OUT/f'results/submission_{name}.csv',index=False)
                subchecks.append(dict(model=name,rows=len(sub),columns=list(sub),ids_in_sample_order=True,
                                      finite_probability=True,min=float(sub.TARGET.min()),max=float(sub.TARGET.max()),
                                      training_scope='all labelled applicants'))
            del bundle
    pd.DataFrame(holdrows).to_csv(OUT/'results/holdout_metrics.csv',index=False)
    save_json(OUT/'results/submission_validation.json',subchecks)

def rebuild_data(data):
    from baseline_data import audit_inputs
    from baseline_features_v2 import build_histories,attach_histories
    out=OUT/'data_build';(out/'results').mkdir(parents=True,exist_ok=True)
    train,test,_=audit_inputs(ROOT,out)
    history,_=build_histories(ROOT,out);v2train,v2test=attach_histories(train,test,history)
    for version,pair in [('v1',(train,test)),('v2',(v2train,v2test))]:
        for i,role in enumerate(['train','test']):
            actual=pair[i].set_index('SK_ID_CURR').loc[data[version][i].SK_ID_CURR].reset_index()
            expected=data[version][i]
            pd.testing.assert_frame_equal(actual[expected.columns],expected,check_exact=False,rtol=1e-10,atol=1e-8)
            actual[expected.columns].to_csv(out/f'application_{role}_processed_{version}.csv',index=False)
    save_json(OUT/'results/raw_reconstruction.json',dict(all_four_shared_files_reproduced=True,rtol=1e-10,atol=1e-8))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--rebuild-data',action='store_true');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();key=freeze_protocol();data,membership,new39=load_data()
    if args.rebuild_data:rebuild_data(data)
    if not args.validate_only:
        run_cv(data,membership,new39,key,args.resume);lock=lock_models(key)
        final_fits(data,membership,new39,lock,args.resume)
    from reporting import report_outputs,validate
    report_outputs(OUT);checks=validate(OUT,membership)
    save_json(OUT/'results/validation.json',checks);print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
