"""Opt-in fixed A/B/C CV, using unchanged original training implementations.

Requires the original verified processed train tables, raw application table and
membership.csv. Saves only to a new directory; never performs tuning/final fits.
"""
from pathlib import Path
import argparse,json,sys,time
BASE=Path(__file__).resolve().parent

def train(data_root,output,groups=('A',)):
    import workflow as w
    data_root=Path(data_root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new output directory to preserve prior results')
    sys.path[:0]=[str(BASE/'source_snapshot/unified/abc_comparison'),str(BASE/'source_snapshot/unified'),str(BASE/'source_snapshot/baseline')]
    import experiment as e
    original=e.original
    protocol=json.loads((BASE/'support/original_protocol.json').read_text())['protocol']
    # All learned processing and fixed parameters are supplied by original code.
    for rel,h in protocol['code'].items():assert w.sha(BASE/'source_snapshot'/rel)==h
    for rel in ['processed_data/application_train_processed_v1.csv','processed_data/application_train_processed_v2.csv','baseline/splits/membership.csv']:
        assert w.sha(data_root/rel)==protocol['inputs'][rel],rel
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import roc_auc_score,average_precision_score
    mem=pd.read_csv(data_root/'baseline/splits/membership.csv')
    frames={}
    for group,version in [('B','v1'),('C','v2')]:
        f=pd.read_csv(data_root/f'processed_data/application_train_processed_{version}.csv')
        frames[group]=f.set_index('SK_ID_CURR').loc[mem.SK_ID_CURR].reset_index()
        assert np.array_equal(frames[group].TARGET,mem.TARGET)
    raw=pd.read_csv(data_root/'home-credit-default-risk/application_train.csv')
    expected=e.application_frame(raw).set_index('SK_ID_CURR').loc[mem.SK_ID_CURR].reset_index()
    frames['A']=frames['B'][expected.columns].copy()
    pd.testing.assert_frame_equal(frames['A'],expected,rtol=1e-12,atol=1e-10,check_exact=False)
    fit_frames(frames,mem,output,groups)

def fit_frames(frames,mem,output,groups=('A',),models=('LR','RF','LGB'),folds=(0,1,2)):
    """Fit frozen learners to verified prepared frames, writing only a new directory."""
    import pandas as pd
    import joblib
    from sklearn.metrics import roc_auc_score,average_precision_score
    sys.path[:0]=[str(BASE/'source_snapshot/unified/abc_comparison'),str(BASE/'source_snapshot/unified'),str(BASE/'source_snapshot/baseline')]
    import experiment as e
    original=e.original
    import workflow as w
    w.verify_manifest();w.verify_frozen_source()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new output directory to preserve prior results')
    output.mkdir(parents=True);(output/'results').mkdir();(output/'models').mkdir();original.OUT=output
    dev=mem.partition.eq('development');rows=[];predictions=[]
    for name in models:
        for group in groups:
            data=frames[group];version='v2' if group=='C' else 'v1'
            x=e.application_inputs(name,data) if group=='A' else original.model_inputs(name,data)
            for fold in folds:
                tr=dev&mem.fold.ne(fold);va=dev&mem.fold.eq(fold);tag=f'{name}_{group}_fold{fold}'
                print('Training',tag,flush=True)
                bundle=original.fit(name,version,x.loc[tr],data.loc[tr,'TARGET'],mem.loc[tr,'SK_ID_CURR'],tag)
                bundle['training_ids']=mem.loc[tr,'SK_ID_CURR'].to_numpy();joblib.dump(bundle,output/f'models/{tag}.joblib',compress=3)
                p=mem.loc[va,['SK_ID_CURR','TARGET','fold']].copy();p['probability']=original.predict(bundle,x.loc[va]);p['model']=name;p['group']=group;predictions.append(p)
                rows.append(dict(model=name,group=group,fold=fold,roc_auc=roc_auc_score(p.TARGET,p.probability),ap=average_precision_score(p.TARGET,p.probability),**bundle['info']))
    pd.DataFrame(rows).to_csv(output/'results/cv_folds.csv',index=False)
    pd.concat(predictions).to_csv(output/'results/oof_predictions.csv',index=False)
    (output/'results/training_scope.json').write_text(json.dumps(dict(groups=list(groups),model_fits=sum((2 if m=='LGB' else 1)*len(folds)*len(groups) for m in models),holdout_scored=False,final_models=False,source='unchanged original unified functions'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--groups',nargs='+',choices=['A','B','C'],default=['A']);a=p.parse_args()
    train(a.data_root,a.output,a.groups)
