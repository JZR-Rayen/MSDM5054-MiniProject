"""Four manuscript figures from immutable evidence; no model fitting.

Keep the original technical plots. This entry point adds a source-defined
workflow and paired quantitative views, reusing the course plotting style.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter
from style import apply,COLORS,DISPLAY,MARKERS
from audit_panel_alignment import require_matplotlib_panel_alignment
B=Path(__file__).resolve().parents[1];D=B/'source_data';O=B/'figures';Q=B/'validation';SOURCES={}

def read(name):
    p=D/name;meta=json.loads((D/'figure_sources.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta[name]['sha256'],name
    SOURCES[name]={'path':'source_data/'+name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return pd.read_csv(p,float_precision='round_trip')

def letter(ax,text):
    ax.annotate(text,xy=(0,1),xycoords='axes fraction',xytext=(-27,10),textcoords='offset points',weight='bold',fontsize=9,annotation_clip=False)

def export(fig,name,rows,rule):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(Q/f'{name}.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(O/f'{name}.pdf')
    fig.savefig(O/f'{name}.svg')
    fig.savefig(O/f'{name}.png',dpi=600)
    if rows is not None:rows.to_csv(D/f'{name}_plotted.csv',index=False)
    SOURCES[name]={'script':'figure_source/main_figures.py','plotted_data':None if rows is None else f'source_data/{name}_plotted.csv','rule':rule,'size_inches':fig.get_size_inches().tolist()}
    plt.close(fig)

def workflow():
    m=read('membership.csv');assert len(m)==307511 and m.SK_ID_CURR.is_unique
    assert (m.partition=='development').sum()==292135 or (m.partition=='dev').sum()==292135
    spec=json.loads((D/'workflow_definition.json').read_text())
    assert spec['development']==292135 and spec['holdout']==15376
    SOURCES['workflow_definition.json']={'path':'source_data/workflow_definition.json','sha256':hashlib.sha256((D/'workflow_definition.json').read_bytes()).hexdigest()}
    fig,ax=plt.subplots(figsize=(5.5,4.4));fig.subplots_adjust(left=0,right=1,bottom=0,top=1);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,txt,color='#65727C',fill='#F5F7F8',fs=7.7):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.008',lw=.7,edgecolor=color,facecolor=fill))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,linespacing=1.25)
    def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,lw=.7,color='#59636C',shrinkA=1,shrinkB=1))
    ax.text(.5,.974,'Deterministic source construction — no cohort-learned fitting',ha='center',va='center',fontsize=8.2,weight='bold')
    box(.03,.838,.285,.108,'A: Application-table only\nOriginal fields + four\nshared derivatives')
    box(.3575,.838,.285,.108,'B: Basic history\nA + 27 historical\naggregates')
    box(.685,.838,.285,.108,'C: Enhanced history\nB + 39 behavioural\nfeatures')
    arrow(.317,.892,.354,.892);arrow(.645,.892,.682,.892)
    box(.03,.755,.94,.052,'Common IDs and labels: 307,511 applicants; fixed 95% / 5% split')
    for x in [.1725,.50,.8275]:arrow(x,.836,x,.81)
    box(.03,.651,.61,.073,'Development: 292,135 applicants\nSame three outer folds for each A/B/C pipeline')
    box(.685,.651,.285,.073,'Viewed holdout: 15,376\nSupplementary only',fill='#FAFAFA',fs=7.4)
    arrow(.335,.751,.335,.727);arrow(.828,.751,.828,.727)
    box(.03,.53,.61,.050,'Outer training: two folds')
    box(.685,.53,.285,.050,'Outer validation: one fold',fs=7.1)
    ax.plot([.335,.335,.8275],[.647,.615,.615],color='#59636C',lw=.7)
    arrow(.335,.615,.335,.584);arrow(.8275,.615,.8275,.584)
    # Training input fans out to the three fitted branches. Validation bypasses them.
    ax.plot([.1725,.8275],[.496,.496],color='#59636C',lw=.7)
    ax.plot([.335,.335],[.526,.496],color='#59636C',lw=.7)
    texts=[('LR','Median + missing flags\nOne-hot + scaling\nApplication splines /\ninteractions\nFit LR on two folds'),('RF','Fixed application pruning\nWithin-table ratios\nMedian + one-hot\nFit RF on two folds\nNo history terms in A'),('LGB','Native categories / NaNs\nInner 90% fit, 10% stop\nChoose boosting rounds\nFresh processor + booster\nRefit full outer training')]
    for x,(model,txt) in zip([.03,.3575,.685],texts):
        ax.add_patch(FancyBboxPatch((x,.224),.285,.241,boxstyle='round,pad=0.004,rounding_size=0.008',lw=.8,edgecolor=COLORS[model],facecolor='white'))
        ax.text(x+.1425,.438,DISPLAY[model],ha='center',weight='bold',fontsize=8.2,color=COLORS[model])
        ax.text(x+.1425,.325,txt,ha='center',va='center',fontsize=7.35,linespacing=1.3)
        arrow(x+.1425,.494,x+.1425,.468)
        arrow(x+.1425,.219,x+.1425,.178)
    box(.03,.113,.94,.061,'Apply each fitted processor/model to its untouched outer validation fold\nScore AUC and AP; retain ID-aligned out-of-fold predictions',fs=7.5)
    # Validation follows the right border directly to evaluation, never to fitting.
    ax.plot([.97,.987,.987,.972],[.553,.553,.145,.145],color='#59636C',lw=.7)
    arrow(.987,.145,.972,.145)
    box(.03,.026,.94,.050,'Across three folds: mean metrics, sample SD and corresponding-fold gains',fs=7.8)
    arrow(.5,.109,.5,.079)
    export(fig,'study_design',None,'Source-defined information, deterministic applicant-level aggregation before split allowed; learned preprocessing only within training boundaries. The LGB inner 90/10 split is entirely within the two outer-training folds. Holdout never feeds this CV.')

def history():
    s=read('abc_summary.csv');f=read('abc_folds.csv');d=read('abc_deltas.csv')
    assert len(s)==9 and len(f)==27
    for (model,group),q in f.groupby(['model','group']):
        r=s[s.model.eq(model)&s.group.eq(group)].iloc[0]
        assert np.isclose(r.auc_mean,q.roc_auc.mean(),atol=1e-14,rtol=0)
        assert np.isclose(r.auc_sd,q.roc_auc.std(ddof=1),atol=1e-14,rtol=0)
    fig=plt.figure(figsize=(5.5,3.85));gs=fig.add_gridspec(2,2,left=.14,right=.975,bottom=.11,top=.86,hspace=.70,wspace=.42)
    a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1],sharey=b)
    fig.legend([Line2D([],[],color=COLORS[m],lw=1.1) for m in ['LR','RF','LGB']],['LR','RF','LightGBM'],loc='upper center',bbox_to_anchor=(.57,1.0),ncol=3)
    for i,m in enumerate(['LR','RF','LGB']):
        q=s[s.model.eq(m)].set_index('group').loc[list('ABC')];x=np.arange(3)+(i-1)*.055
        a.plot(x,q.auc_mean,color=COLORS[m])
        for j in range(3):a.errorbar(x[j],q.auc_mean.iloc[j],yerr=q.auc_sd.iloc[j],marker='o',color=COLORS[m],capsize=2.5)
    a.set(xticks=range(3),xticklabels=['A: Application-table only','B: Basic history','C: Enhanced history'],ylabel='Mean AUC ± SD\n(zoomed)',ylim=(.746,.786),xlim=(-.20,2.25))
    a.set_yticks([.75,.76,.77,.78]);a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'));a.grid(axis='y',alpha=.15);letter(a,'a')
    rows=[]
    for ax,contrast,panel in [(b,'B-A','b'),(c,'C-B','c')]:
        for i,m in enumerate(['LR','RF','LGB']):
            vals=[]
            for k in range(3):
                z=f[f.model.eq(m)&f.fold.eq(k)].set_index('group');v=z.loc[contrast[0],'roc_auc']-z.loc[contrast[2],'roc_auc'];vals.append(v)
                check=d[d.model.eq(m)&d.contrast.eq(contrast)&d.fold.eq(str(k))].auc_delta.iloc[0];assert abs(check-v)<1e-14
                ax.plot(i+(k-1)*.13,v,marker=MARKERS[k],color=COLORS[m],ls='None')
                rows.append(dict(panel=panel,model=m,contrast=contrast,fold=k,auc_delta=v))
            ax.plot([i-.23,i+.23],[np.mean(vals)]*2,color=COLORS[m])
        ax.set(xticks=range(3),xticklabels=['LR','RF','LGB'],title=contrast.replace('-',' − '),ylim=(-.0007,.0165),xlim=(-.45,2.45))
        ax.axhline(0,color='#777777',lw=.65);ax.grid(axis='y',alpha=.15);ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'));letter(ax,panel)
    b.set_ylabel('Paired AUC gain')
    fig.legend([Line2D([],[],color='#444444',marker=m,ls='None') for m in MARKERS],['Fold 0','Fold 1','Fold 2'],loc='center',bbox_to_anchor=(.57,.49),ncol=3)
    plotted=pd.concat([s.assign(panel='a'),pd.DataFrame(rows)],ignore_index=True)
    export(fig,'history_gains',plotted,'Panel a: nine 3-fold means with sample SD (ddof=1). Panels b/c: all 18 paired B-A/C-B fold gains. No C-A panel because it is the algebraic sum; all 27 source contrasts remain in technical data. No CI or hypothesis test.')

def refinement():
    o=read('optimization_folds.csv');b=read('blend_folds.csv');rows=[]
    original=o[o.tag.eq('C00_base')].set_index('fold');refined=o[o.tag.eq('C11_base')].set_index('fold')
    assert len(original)==len(refined)==3
    for key in ['refinement','raw80_20','rank80_20','rank70_20_10']:
        new=refined if key=='refinement' else b[b.tag.eq(key)].set_index('fold');base=original if key=='refinement' else refined
        for fold in range(3):
            rows.append(dict(contrast=key,fold=fold,reference='C00_base' if key=='refinement' else 'C11_base',auc_new=new.loc[fold,'roc_auc'],auc_reference=base.loc[fold,'roc_auc'],ap_new=new.loc[fold,'ap'],ap_reference=base.loc[fold,'ap'],auc_delta=new.loc[fold,'roc_auc']-base.loc[fold,'roc_auc'],ap_delta=new.loc[fold,'ap']-base.loc[fold,'ap']))
    r=pd.DataFrame(rows);assert len(r)==12
    fig,axes=plt.subplots(1,2,figsize=(5.5,3.2),sharey=True);fig.subplots_adjust(left=.237,right=.927,bottom=.17,top=.84,wspace=.30)
    labels=['Refined LGB\nvs original LGB','LGB–LR raw\n80/20','LGB–LR rank\n80/20 (selected)','LGB–LR–RF rank\n70/20/10'];ys=[3.7,2.4,1.25,.1]
    for ax,metric,panel in zip(axes,['auc','ap'],['a','b']):
        for i,key in enumerate(['refinement','raw80_20','rank80_20','rank70_20_10']):
            vals=r[r.contrast.eq(key)].sort_values('fold')[metric+'_delta'].to_numpy();color=COLORS['LGB' if key=='refinement' else 'Fusion']
            for k,v in enumerate(vals):ax.plot(v,ys[i]+(k-1)*.15,marker=MARKERS[k],color=color,ls='None')
            ax.plot([vals.mean()]*2,[ys[i]-.32,ys[i]+.32],color=color)
        ax.axvline(0,color='#777777',lw=.65);ax.axhline(3.05,color='#AAAAAA',lw=.5,ls=':')
        ax.set(yticks=ys,yticklabels=labels,xlabel='Paired '+metric.upper()+' change',ylim=(-.5,4.3));ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0)
        if metric=='auc':ax.set_xlim(-.00085,.0025);ax.set_xticks([0,.001,.002])
        else:ax.set_xlim(-.001,.0048);ax.set_xticks([0,.002,.004])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'));letter(ax,panel)
    fig.legend([Line2D([],[],color='#444444',marker=m,ls='None') for m in MARKERS],['Fold 0','Fold 1','Fold 2'],loc='upper center',bbox_to_anchor=(.5,.99),ncol=3)
    export(fig,'refinement_effects',r,'All 12 paired fold contrasts for each metric: refinement minus original C LightGBM (top row); three blends minus refined C LightGBM (remaining rows). Three fold points plus vertical mean, no interval; axes include zero. Extra-feature trials use C10 and are deliberately not pooled. No holdout or test scores enter.')

def diagnostics():
    """Descriptive top-five terms from original C full-development models."""
    fig,axes=plt.subplots(3,1,figsize=(5.5,3.7))
    fig.subplots_adjust(left=.285,right=.94,bottom=.10,top=.90,hspace=1.05)
    labels={'AMT_GOODS_PRICE':'Goods price','FLAG_DOCUMENT_16':'Document 16 flag',
            'FLAG_DOCUMENT_18':'Document 18 flag','FLAG_DOCUMENT_13':'Document 13 flag',
            'ORGANIZATION_TYPE_Transport: type 3':'Transport type 3 (org.)',
            'YEARS_EMPLOYED':'Employment years','ORGANIZATION_TYPE':'Organization type',
            'H_ACTIVE_NET_DEBT_CREDIT_RATIO_CUR1':'Active net debt / credit',
            'EXT_SOURCE_1':'External score 1','EXT_SOURCE_2':'External score 2','EXT_SOURCE_3':'External score 3'}
    rows=[]
    for ax,m,unit,panel in zip(axes,['LR','RF','LGB'],
            ['Coefficient (transformed term)','Impurity importance','Split gain (thousands)'],['a','b','c']):
        z=read(m+'_importance.csv');z=z.iloc[np.argsort(z.value.abs())[-5:]].copy()
        z['model']=m;z['display_label']=[labels[c.removeprefix('numeric__')] for c in z.feature]
        z['display_value']=z.value/(1000 if m=='LGB' else 1);rows.append(z)
        ax.barh(z.display_label,z.display_value,color=COLORS[m],height=.58)
        ax.axvline(0,color='#777777',lw=.6);ax.set(xlabel=unit,title=DISPLAY[m])
        ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
        letter(ax,panel)
    export(fig,'model_diagnostics',pd.concat(rows,ignore_index=True),
           'Top five terms by absolute within-model value from each original C full-development model; 15 entries. LR signed transformed coefficients; RF impurity importance; LightGBM split gain /1000. Separate scales, no intervals, no causal or stability inference. Full input tables retained.')

def cv_table():
    """Report-level absolute metrics, checked against all original fold scores."""
    s=read('abc_summary.csv');f=read('abc_folds.csv')
    lines=[r'\begin{tabular}{llccc}',r'\toprule Pipeline & Metric & A: Application-only & B: Basic history & C: Enhanced history \\',r'\midrule']
    for model in ['LR','RF','LGB']:
        for metric,label,col in [('auc','AUC','roc_auc'),('ap','AP','ap')]:
            cells=[]
            for group in 'ABC':
                q=f[f.model.eq(model)&f.group.eq(group)][col]
                row=s[s.model.eq(model)&s.group.eq(group)].iloc[0]
                assert len(q)==3 and np.isclose(q.mean(),row[metric+'_mean'],atol=1e-14,rtol=0)
                assert np.isclose(q.std(ddof=1),row[metric+'_sd'],atol=1e-14,rtol=0)
                cells.append(fr'${q.mean():.4f} \pm {q.std(ddof=1):.4f}$')
            lines.append(' & '.join([DISPLAY[model] if metric=='auc' else '',label,*cells])+r' \\')
        if model!='LGB':lines.append(r'\addlinespace[2pt]')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (B/'table_core_cv.tex').write_text('\n'.join(lines)+'\n')


def main():
    for p in [O,Q]:p.mkdir(exist_ok=True)
    apply();workflow();history();diagnostics();refinement();cv_table()
    (D/'main_figure_sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
if __name__=='__main__':main()
