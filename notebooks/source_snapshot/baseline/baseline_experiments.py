"""Fixed experiments, optimization checks, pre-holdout lock and aligned predictions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix)
from threadpoolctl import threadpool_limits
from baseline_data import Preprocessor, correct_fields, sha256, save_json, SEED

KKT_TOL=1e-5
GROUPS=['l2_unweighted','l2_balanced','elasticnet_unweighted','elasticnet_balanced']

class LinearProbabilityModel:
    def __init__(self,coef,intercept):
        self.coef_=np.asarray(coef).reshape(1,-1)
        self.intercept_=np.array([intercept]); self.classes_=np.array([0,1])
    def predict_proba(self,x):
        p=expit(np.asarray(x@self.coef_[0]).ravel()+self.intercept_[0])
        return np.column_stack([1-p,p])

def sample_weights(y,weight):
    if weight is None: return np.ones(len(y))
    if weight!='balanced': raise ValueError('Unsupported class weight')
    counts=np.bincount(y,minlength=2)
    return len(y)/(2*counts[y])

def objective_gradient(x,y,coef,intercept,weights,C,ratio):
    # Same objective as sklearn SAGA: weighted loss SUM + penalty / C.
    # Divide everything by sum(weights) for a numerically stable equivalent objective.
    total=weights.sum(); z=np.asarray(x@coef).ravel()+intercept
    loss=np.dot(weights,np.logaddexp(0,z)-y*z)/total
    residual=weights*(expit(z)-y)/total
    grad=np.asarray(x.T@residual).ravel()+(1-ratio)*coef/(C*total)
    g_intercept=residual.sum()
    smooth=loss+(1-ratio)*np.dot(coef,coef)/(2*C*total)
    return smooth,grad,g_intercept,ratio/(C*total)

def kkt_residual(x,y,coef,intercept,weights,C,ratio):
    _,g,gi,l1=objective_gradient(x,y,coef,intercept,weights,C,ratio)
    r=np.where(coef!=0,abs(g+l1*np.sign(coef)),np.maximum(abs(g)-l1,0))
    return float(max(abs(gi),r.max(initial=0)))

def fit_model(x,y,config):
    y=np.asarray(y,dtype=int); start=time.perf_counter()
    C=float(config['C']); ratio=float(config.get('l1_ratio') or 0)
    weights=sample_weights(y,config['class_weight']); attempts=[]
    with threadpool_limits(limits=1):
        if config['penalty']=='l2':
            model=LogisticRegression(solver='lbfgs',penalty='l2',C=C,
                class_weight=config['class_weight'],tol=1e-7,max_iter=8000,random_state=SEED)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); model.fit(x,y)
            warning_messages=[str(w.message) for w in caught]
            kkt=kkt_residual(x,y,model.coef_[0],model.intercept_[0],weights,C,0)
            success=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            attempts=[{'iterations':int(model.n_iter_[0]),'warnings':warning_messages,'kkt_residual':kkt}]
            solver='sklearn lbfgs'; iterations=int(model.n_iter_[0])
        else:
            # Exact Elastic Net via beta=u-v with u,v>=0; intercept remains unpenalized.
            # Smooth bound-constrained optimization avoids SAGA's outlier-limited step size.
            d=x.shape[1]; q=np.zeros(2*d+1)
            prior=np.average(y,weights=weights); q[-1]=np.log(prior/(1-prior))
            def fun(q):
                beta=q[:d]-q[d:2*d]
                f,g,gi,l1=objective_gradient(x,y,beta,q[-1],weights,C,ratio)
                return f+l1*q[:2*d].sum(),np.r_[g+l1,-g+l1,gi]
            bounds=[(0,None)]*(2*d)+[(None,None)]
            success=False; iterations=0
            for ftol in [1e-12,1e-14,1e-15]:
                result=minimize(fun,q,method='L-BFGS-B',jac=True,bounds=bounds,
                     options={'gtol':1e-7,'ftol':ftol,'maxiter':4000,'maxls':50,'maxcor':30})
                q=result.x; beta=q[:d]-q[d:2*d]
                kkt=kkt_residual(x,y,beta,q[-1],weights,C,ratio)
                attempts.append({'iterations':int(result.nit),'success':bool(result.success),
                                 'message':str(result.message),'ftol':ftol,'kkt_residual':kkt})
                iterations+=result.nit
                success=bool(result.success) and kkt<=KKT_TOL
                if success: break
            model=LinearProbabilityModel(beta,q[-1]); solver='scipy L-BFGS-B positive/negative split'
    info={'solver':solver,'iterations':int(iterations),'attempts':attempts,
          'kkt_residual':kkt,'converged':bool(success and kkt<=KKT_TOL),
          'seconds':time.perf_counter()-start,'features':x.shape[1],
          'nonzero_exact':int(np.count_nonzero(model.coef_)),
          'nonzero_above_1e8':int((abs(model.coef_)>1e-8).sum())}
    if not info['converged']:
        raise RuntimeError('Optimization not converged: '+json.dumps(info))
    return model,info

def best_f1_threshold(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    f=np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],
                out=np.zeros(len(thresholds)),where=(precision[:-1]+recall[:-1])>0)
    # Scores >= threshold are positive. Exact F1 ties use highest observed threshold.
    i=np.flatnonzero(f==f.max())[-1]
    return float(thresholds[i]),float(f[i])

def metrics(y,p,threshold):
    label=np.asarray(p)>=threshold; cm=confusion_matrix(y,label,labels=[0,1]).ravel()
    return {'roc_auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),
        'precision':float(precision_score(y,label,zero_division=0)),
        'recall':float(recall_score(y,label,zero_division=0)),
        'f1':float(f1_score(y,label,zero_division=0)),
        **dict(zip(['tn','fp','fn','tp'],map(int,cm)))}

def make_submission(ids,probabilities,sample):
    s=pd.DataFrame({'SK_ID_CURR':ids,'TARGET':probabilities})
    if s.SK_ID_CURR.isna().any() or not s.SK_ID_CURR.is_unique or not sample.SK_ID_CURR.is_unique:
        raise ValueError('Invalid IDs')
    if set(s.SK_ID_CURR)!=set(sample.SK_ID_CURR): raise ValueError('ID coverage mismatch')
    if not np.isfinite(s.TARGET).all() or not s.TARGET.between(0,1).all():
        raise ValueError('Invalid probabilities')
    return s.set_index('SK_ID_CURR').loc[sample.SK_ID_CURR].reset_index()[['SK_ID_CURR','TARGET']]

def candidate(penalty,weight,C,ratio=None,stage='initial'):
    group=f"{penalty}_{'balanced' if weight else 'unweighted'}"
    return {'candidate':f'{group}_C{C:g}'+(f'_r{ratio:g}' if ratio is not None else ''),
            'group':group,'penalty':penalty,'class_weight':weight,'C':float(C),
            'l1_ratio':ratio,'stage':stage}

def initial_candidates():
    return [candidate(p,w,c,r) for p in ['l2','elasticnet'] for w in [None,'balanced']
            for c in [.01,.1,1.] for r in ([None] if p=='l2' else [.25,.75])]

def prepare_membership(train,membership,out):
    out=Path(out); expected=train[['SK_ID_CURR','TARGET']].set_index('SK_ID_CURR')
    assert membership.SK_ID_CURR.is_unique and set(membership.SK_ID_CURR)==set(train.SK_ID_CURR)
    assert np.array_equal(expected.loc[membership.SK_ID_CURR,'TARGET'],membership.TARGET)
    assert set(membership.partition)=={'development','holdout'}
    dev=membership[membership.partition=='development']; hold=membership[membership.partition=='holdout']
    assert not set(dev.SK_ID_CURR)&set(hold.SK_ID_CURR)
    assert len(hold)==int(np.ceil(.05*len(train))) and set(dev.fold)=={0,1,2} and hold.fold.eq(-1).all()
    table=membership.groupby(['partition','fold']).TARGET.agg(['count','sum','mean']).reset_index()
    table.to_csv(out/'results/split_summary.csv',index=False)
    return train.set_index('SK_ID_CURR').loc[membership.SK_ID_CURR].reset_index()

def run_search(train,membership,out,reuse=False):
    out=Path(out); dev=membership.partition.eq('development').to_numpy()
    d=train.loc[dev].reset_index(drop=True); folds=membership.loc[dev,'fold'].to_numpy()
    x=correct_fields(d); y=d.TARGET.to_numpy(); candidates=initial_candidates(); rows=[]; dummy=[]
    cache=out/'results/cv'; cache.mkdir(exist_ok=True)
    fingerprint={'data':json.loads((out/'results/input_fingerprints.json').read_text()),
      'code':{p.name:sha256(p) for p in [out/'baseline_data.py',out/'baseline_experiments.py']},
      'membership':sha256(out/'splits/membership.csv'),'seed':SEED,'kkt_tol':KKT_TOL}
    key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    manifest=cache/'manifest.json'
    if reuse and manifest.exists() and json.loads(manifest.read_text())['key']!=key:
        raise ValueError('Cached fit provenance mismatch; rerun with reuse=False')
    save_json(manifest,{'key':key,'provenance':fingerprint})
    def run_candidates(configs):
        batch=[]
        for fold in range(3):
            tr=folds!=fold; va=folds==fold
            prep=Preprocessor(); z=prep.fit_transform(x.loc[tr]); v=prep.transform(x.loc[va])
            assert len(set(d.SK_ID_CURR[tr]) & set(d.SK_ID_CURR[va]))==0
            save_json(out/f'results/preprocessing_fold{fold}.json',
                {**prep.metadata(),'fit_applicants':int(tr.sum()),'unknown_validation':prep.unknown_counts(x.loc[va])})
            for cfg in configs:
                name=cfg['candidate']; path=cache/f'{name}_fold{fold}.npz'; meta=path.with_suffix('.json')
                if reuse and path.exists() and meta.exists():
                    record=json.loads(meta.read_text()); a=np.load(path)
                    assert record['key']==key and np.array_equal(a['ids'],d.SK_ID_CURR[va])
                    batch.append(record['row']); continue
                m,info=fit_model(z,y[tr],cfg); prob=m.predict_proba(v)[:,1]
                row={**cfg,'fold':fold,**metrics(y[va],prob,.5),**{k:val for k,val in info.items() if k!='attempts'}}
                np.savez_compressed(path,ids=d.SK_ID_CURR[va].to_numpy(),y=y[va],prob=prob,
                    names=prep.get_feature_names_out(),coef=m.coef_[0],intercept=m.intercept_[0])
                save_json(meta,{'key':key,'row':row,'convergence':info})
                batch.append(row)
                pd.DataFrame(rows+batch).to_csv(out/'results/cv_folds.csv',index=False)
                print(f"{name} fold={fold}: AUC={row['roc_auc']:.6f}, AP={row['ap']:.6f}, "
                      f"iter={info['iterations']}, KKT={info['kkt_residual']:.2g}, seconds={info['seconds']:.1f}",flush=True)
            if configs[0]['stage']=='initial':
                prob=np.full(va.sum(),y[tr].mean())
                dummy.append({'fold':fold,'training_prior':float(y[tr].mean()),**metrics(y[va],prob,.5)})
            del z,v,prep
        return batch
    rows.extend(run_candidates(candidates))
    pd.DataFrame(dummy).to_csv(out/'results/dummy_cv.csv',index=False)
    assert len(rows)==54 and all(r['converged'] for r in rows)
    def summarize():
        frame=pd.DataFrame(rows)
        result=frame.groupby('candidate',sort=False).agg(roc_auc_mean=('roc_auc','mean'),
          roc_auc_std=('roc_auc','std'),ap_mean=('ap','mean'),ap_std=('ap','std'),
          nonzero_mean=('nonzero_exact','mean'),feature_mean=('features','mean')).reset_index()
        return result.merge(pd.DataFrame(candidates),on='candidate',validate='one_to_one')
    summary=summarize(); extensions=[]; decisions=[]
    # One boundary extension per group, only with monotonic CV trend exceeding 1e-4 at both steps.
    # Fixed engineering criterion; no holdout scores are available here.
    for group in GROUPS:
        sub=summary[summary.group==group].sort_values(['roc_auc_mean','C','l1_ratio'],ascending=[False,True,True],na_position='first')
        winner=sub.iloc[0]; directional=None
        trend=sub if winner.penalty=='l2' else sub[sub.l1_ratio==winner.l1_ratio]
        trend=trend.sort_values('C'); scores=trend.roc_auc_mean.to_numpy()
        if winner.C==.01 and np.all(np.diff(scores)<-1e-4): directional=.001
        if winner.C==1 and np.all(np.diff(scores)>1e-4): directional=10.
        decisions.append({'group':group,'initial_best':winner.candidate,'trend_C':trend.C.tolist(),
            'trend_mean_auc':scores.tolist(),'criterion':'both adjacent improvements > 0.0001 toward boundary',
            'extension_C':directional,'reason':'monotonic boundary trend' if directional else 'boundary trend criterion not met'})
        if directional:
            extensions += [candidate(winner.penalty,None if 'unweighted' in group else 'balanced',directional,r,'extension')
                           for r in ([None] if winner.penalty=='l2' else [.25,.75])]
    save_json(out/'results/boundary_decisions.json',decisions)
    if extensions:
        rows.extend(run_candidates(extensions)); candidates.extend(extensions); summary=summarize()
    pd.DataFrame(rows).to_csv(out/'results/cv_folds.csv',index=False)
    summary.to_csv(out/'results/cv_summary.csv',index=False)
    # Deterministic exact AUC ties: L2, unweighted, smaller C, smaller l1_ratio.
    summary['group_order']=summary.group.map({g:i for i,g in enumerate(GROUPS)})
    ordered=summary.sort_values(['roc_auc_mean','group_order','C','l1_ratio'],ascending=[False,True,True,True],na_position='first')
    best=ordered.groupby('group',sort=False).head(1).copy(); locks=[]; oof=d[['SK_ID_CURR','TARGET']].copy()
    for _,r in best.iterrows():
        cfg=next(c for c in candidates if c['candidate']==r.candidate); pred=np.full(len(d),np.nan)
        for fold in range(3):
            a=np.load(cache/f'{r.candidate}_fold{fold}.npz'); va=folds==fold
            assert np.array_equal(a['ids'],d.SK_ID_CURR[va]); pred[va]=a['prob']
        assert np.isfinite(pred).all(); threshold,f1=best_f1_threshold(y,pred)
        oof[r.group]=pred
        locks.append({**cfg,'cv_auc_mean':float(r.roc_auc_mean),'cv_auc_std':float(r.roc_auc_std),
              'cv_ap_mean':float(r.ap_mean),'threshold':threshold,'oof_f1_at_threshold':f1})
    oof.to_csv(out/'results/oof_predictions.csv',index=False)
    best.to_csv(out/'results/group_best.csv',index=False)
    lock={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'winner':ordered.iloc[0].candidate,
      'groups':locks,'selection_metric':'mean fold ROC-AUC','tie_rule':'L2, unweighted, smaller C, smaller l1_ratio',
      'threshold_rule':'maximum OOF F1; exact tie uses highest observed probability; positive if p >= threshold',
      'holdout_used':False,'initial_fits':54,'extension_fits':len(extensions)*3,'cache_key':key}
    save_json(out/'results/model_lock.json',lock)
    print('LOCKED BEFORE HOLDOUT:',lock['winner'],flush=True)
    return lock

def evaluate_holdout(train,membership,out):
    out=Path(out); lock_path=out/'results/model_lock.json'; lock=json.loads(lock_path.read_text())
    lock_hash=sha256(lock_path); assert lock['holdout_used'] is False
    dev=membership.partition.eq('development').to_numpy(); hold=~dev
    xd=correct_fields(train.loc[dev]); xh=correct_fields(train.loc[hold]); y=train.TARGET.to_numpy()
    prep=Preprocessor(); z=prep.fit_transform(xd); v=prep.transform(xh)
    save_json(out/'results/preprocessing_development.json',{**prep.metadata(),'fit_applicants':int(dev.sum()),
             'unknown_holdout':prep.unknown_counts(xh)})
    predictions=train.loc[hold,['SK_ID_CURR','TARGET']].copy(); results=[]; coefficients=[]; convergence=[]
    for cfg in lock['groups']:
        model,info=fit_model(z,y[dev],cfg); prob=model.predict_proba(v)[:,1]; predictions[cfg['group']]=prob
        for kind,t in [('default',.5),('oof_f1',cfg['threshold'])]:
            results.append({'group':cfg['group'],'candidate':cfg['candidate'],'threshold_kind':kind,'threshold':t,
                **metrics(y[hold],prob,t),'nonzero_exact':info['nonzero_exact'],'features':info['features']})
        coefficients.extend({'group':cfg['group'],'feature':n,'coefficient':float(c)}
                      for n,c in zip(prep.get_feature_names_out(),model.coef_[0]))
        convergence.append({'stage':'development_refit','candidate':cfg['candidate'],**info})
        joblib.dump({'preprocessor':prep,'model':model,'config':cfg,'lock_hash':lock_hash},
                    out/f"models/development_{cfg['group']}.joblib",compress=3)
        print('Holdout',cfg['group'],results[-1],flush=True)
    prior=float(y[dev].mean()); p=np.full(hold.sum(),prior); predictions['dummy']=p
    results.append({'group':'dummy','candidate':'training_prior','threshold_kind':'default',
                    'threshold':.5,**metrics(y[hold],p,.5),'nonzero_exact':0,'features':0})
    predictions.to_csv(out/'results/holdout_predictions.csv',index=False)
    pd.DataFrame(results).to_csv(out/'results/holdout_metrics.csv',index=False)
    pd.DataFrame(coefficients).to_csv(out/'results/development_coefficients.csv',index=False)
    save_json(out/'results/holdout_convergence.json',convergence)
    assert sha256(lock_path)==lock_hash
    save_json(out/'results/holdout_evaluation_record.json',{'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),
       'lock_sha256':lock_hash,'holdout_applicants':int(hold.sum()),'holdout_positives':int(y[hold].sum()),
       'development_applicants':int(dev.sum()),'dummy_training_prior':prior,'lock_unchanged':True})
    return pd.DataFrame(results)

def refit_submission(train,test,out):
    out=Path(out); lock=json.loads((out/'results/model_lock.json').read_text())
    assert (out/'results/holdout_evaluation_record.json').exists()
    cfg=next(c for c in lock['groups'] if c['candidate']==lock['winner'])
    prep=Preprocessor(); x=correct_fields(train); xt=correct_fields(test)
    xt=xt[x.columns]; z=prep.fit_transform(x); v=prep.transform(xt)
    model,info=fit_model(z,train.TARGET.to_numpy(),cfg); prob=model.predict_proba(v)[:,1]
    sample=pd.read_csv(out.parent/'home-credit-default-risk/sample_submission.csv')
    submission=make_submission(test.SK_ID_CURR,prob,sample); submission.to_csv(out/'submission.csv',index=False)
    loaded=pd.read_csv(out/'submission.csv'); assert loaded.SK_ID_CURR.equals(sample.SK_ID_CURR)
    assert list(loaded)==['SK_ID_CURR','TARGET'] and len(loaded)==48744
    assert loaded.SK_ID_CURR.is_unique and np.isfinite(loaded.TARGET).all() and loaded.TARGET.between(0,1).all()
    joblib.dump({'preprocessor':prep,'model':model,'config':cfg},out/'models/submission_model.joblib',compress=3)
    save_json(out/'results/submission_convergence.json',info)
    save_json(out/'results/preprocessing_full.json',{**prep.metadata(),'fit_applicants':len(train),
       'unknown_official_test':prep.unknown_counts(xt)})
    pd.DataFrame({'feature':prep.get_feature_names_out(),'coefficient':model.coef_[0]}).to_csv(out/'results/submission_coefficients.csv',index=False)
    save_json(out/'results/submission_validation.json',{'rows':len(loaded),'columns':list(loaded),
       'unique_ids':int(loaded.SK_ID_CURR.nunique()),'sample_order_matches':True,'coverage_matches':True,
       'finite_probabilities':True,'probability_range':[float(loaded.TARGET.min()),float(loaded.TARGET.max())],
       'sha256':sha256(out/'submission.csv'),'winner':cfg,'uploaded_to_kaggle':False})
    return submission
