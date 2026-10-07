"""Read immutable saved results; export shared figures and exact plotted rows.

All paired folds are retained. No model fitting, resampling or significance test.
Figures use separate scales when metrics or importance definitions differ.
"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from style import apply,COLORS,DISPLAY,MARKERS
from audit_panel_alignment import require_matplotlib_panel_alignment
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
OUT=BASE/'figures';DATA=BASE/'source_data';QA=BASE/'validation';sources={}

def read(rel,name):
    p=ROOT/rel;target=DATA/name
    if not p.exists():
        p=target  # Portable report bundle: reuse the exact copied source CSV.
        expected=json.loads((DATA/'figure_sources.json').read_text())[name]['sha256']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
    if p.resolve()!=target.resolve():shutil.copy2(p,target)
    sources[name]={'original':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return pd.read_csv(p,float_precision='round_trip')

def panels(fig,axes):
    for i,ax in enumerate(np.ravel(axes)):
        ax.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-28,10),textcoords='offset points',weight='bold',fontsize=9,annotation_clip=False)

def export(fig,name,rows,rule):
    titles={'abc_auc':'CV AUC','paired_deltas':'Feature gains','eda':'Data overview','fusion_deltas':'Fusion trade-offs','feature_trials':'Feature trials','interpretation':'Model diagnostics','holdout':'Viewed holdout'}
    fig.suptitle(titles[name],x=.5,y=.98,fontsize=9,fontweight='bold')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(QA/f'{name}.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.svg');fig.savefig(OUT/f'{name}.png',dpi=600)
    rows.to_csv(DATA/f'{name}_plotted.csv',index=False)
    sources[name]={'plotted_data':f'{name}_plotted.csv','rule':rule,'width_inches':fig.get_size_inches().tolist(),'n_folds':3 if name not in ['eda','interpretation','holdout'] else None}
    plt.close(fig)

def fold_legend(fig):
    handles=[Line2D([],[],color='#444444',marker=m,linestyle='None',label=f'Fold {i}') for i,m in enumerate(MARKERS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.57,.91),ncol=3)

def main():
    for d in [OUT,DATA,QA]:d.mkdir(parents=True,exist_ok=True)
    apply()
    s=read('unified/abc_comparison/results/cv_summary.csv','abc_summary.csv')
    d=read('unified/abc_comparison/results/paired_deltas.csv','abc_deltas.csv')
    f=read('unified/abc_comparison/results/cv_folds.csv','abc_folds.csv')
    fig,ax=plt.subplots(figsize=(5.5,2.6));fig.subplots_adjust(left=.14,right=.97,bottom=.22,top=.80)
    for i,m in enumerate(['LR','RF','LGB']):
        q=s[s.model.eq(m)].set_index('group').loc[list('ABC')]
        x=np.arange(3)+(i-1)*.08
        ax.plot(x,q.auc_mean,color=COLORS[m],label=DISPLAY[m])
        for j,g in enumerate('ABC'):ax.errorbar(x[j],q.auc_mean.iloc[j],yerr=q.auc_sd.iloc[j],marker=MARKERS[j],color=COLORS[m],capsize=3)
    ax.set(xticks=range(3),xticklabels=['A: Application-table\nonly','B: Basic\nhistory','C: Enhanced\nhistory'],ylabel='Mean AUC ± SD (zoomed)',ylim=(.746,.786))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'));ax.grid(axis='y',alpha=.15)
    fig.legend(loc='upper center',bbox_to_anchor=(.55,.92),ncol=3);export(fig,'abc_auc',s,'Nine means and ddof=1 SD, 3 fixed folds; no CI; y axis zoomed.')
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.5),sharey=True);fig.subplots_adjust(left=.12,right=.98,bottom=.21,top=.73,wspace=.20)
    q=d[d.fold.ne('mean')].copy();assert len(q)==27
    for ax,c in zip(axes,['B-A','C-B','C-A']):
        for i,m in enumerate(['LR','RF','LGB']):
            vals=q[q.contrast.eq(c)&q.model.eq(m)].sort_values('fold').auc_delta.to_numpy()
            for k,v in enumerate(vals):ax.plot(i+(k-1)*.15,v,marker=MARKERS[k],color=COLORS[m],ls='None')
            ax.plot([i-.25,i+.25],[vals.mean()]*2,color=COLORS[m])
        ax.axhline(0,color='#777777',lw=.7);ax.set(title=c.replace('-',' − '),xticks=range(3),xticklabels=['LR','RF','LGB'],ylim=(-.001,.026))
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Paired AUC difference');axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    panels(fig,axes);fold_legend(fig);export(fig,'paired_deltas',q,'All 27 paired differences; horizontal short marks are 3-fold means; zero included.')
    missing=read('unified/wzy_data/missing_statistics.csv','missing.csv')
    mem=read('baseline/splits/membership.csv','membership.csv')
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.5));fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.80,wspace=.55)
    counts=mem.TARGET.value_counts().sort_index();axes[0].bar(['No difficulty','Difficulty'],counts/len(mem)*100,color=['#777777','#0072B2'],width=.55)
    axes[0].set(ylabel='Applications (%)',ylim=(0,100),title='Label imbalance')
    axes[1].hist(missing.missing_ratio*100,bins=np.arange(0,101,10),color='#777777',edgecolor='white')
    axes[1].set(xlabel='Missing values per column (%)',ylabel='Fields with missing values',title='Missingness (67 fields)')
    panels(fig,axes);export(fig,'eda',missing,'Missingness uses 67 application fields with nonzero missingness in saved EDA; excludes zero-missingness and joined-history fields; descriptive full cohort target counts, no inferential error bars.')
    blend=read('unified/optimization_20261007/results/blend_folds.csv','blend_folds.csv')
    opt=read('unified/optimization_20261007/results/cv_folds.csv','optimization_folds.csv')
    reference=opt[opt.tag.eq('C11_base')].set_index('fold');rows=[]
    for r in blend.itertuples():rows.append(dict(tag=r.tag,fold=r.fold,auc_delta=r.roc_auc-reference.loc[r.fold,'roc_auc'],ap_delta=r.ap-reference.loc[r.fold,'ap']))
    b=pd.DataFrame(rows);assert len(b)==9
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.5));fig.subplots_adjust(left=.14,right=.97,bottom=.25,top=.73,wspace=.57)
    tags=['raw80_20','rank80_20','rank70_20_10'];labels=['LGB–LR\nraw\n80/20','LGB–LR\nrank\n80/20','LGB–LR–RF\nrank\n70/20/10']
    for ax,col,title in zip(axes,['auc_delta','ap_delta'],['AUC change','AP change']):
        for i,t in enumerate(tags):
            vals=b[b.tag.eq(t)].sort_values('fold')[col].to_numpy()
            for k,v in enumerate(vals):ax.plot(i+(k-1)*.16,v,marker=MARKERS[k],color=COLORS['Fusion'],ls='None')
            ax.plot([i-.26,i+.26],[vals.mean()]*2,color=COLORS['Fusion'])
        ax.axhline(0,color='#777777',lw=.7);ax.set(title=title,xticks=range(3),xticklabels=labels,ylabel='Change vs refined LGB')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'));ax.grid(axis='y',alpha=.15)
        lo=min(-.0001,b[col].min())-.00012;hi=b[col].max()+.00015;ax.set_ylim(lo,hi)
    panels(fig,axes);fold_legend(fig);export(fig,'fusion_deltas',b,'All 9 blend folds versus same-fold C11; both metrics; no CI.')
    trials=read('unified/optimization_20261007/results/feature_paired_deltas.csv','feature_trials.csv');q=trials[trials.kind.eq('fixed_config')].copy();assert len(q)==9
    fig,ax=plt.subplots(figsize=(5.5,2.4));fig.subplots_adjust(left=.16,right=.97,bottom=.22,top=.76)
    for i,t in enumerate(['T','S','TS']):
        vals=q[q.family.eq(t)].sort_values('fold').auc_delta.to_numpy()
        for k,v in enumerate(vals):ax.plot(i+(k-1)*.14,v,marker=MARKERS[k],color=COLORS['LGB'],ls='None')
        ax.plot([i-.25,i+.25],[vals.mean()]*2,color=COLORS['LGB'])
    ax.axhline(0,color='#777777',lw=.7);ax.set(xticks=range(3),xticklabels=['Recent–overall\ndifferences','Status-specific\ndelinquency','Combined'],ylabel='Paired AUC difference');ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    fold_legend(fig);export(fig,'feature_trials',q,'All 9 fixed-C10 feature-trial differences; T/S/TS not adopted. Full optimized-family comparisons remain in source CSV.')
    fig,axes=plt.subplots(3,1,figsize=(5.5,3.85));fig.subplots_adjust(left=.30,right=.95,bottom=.10,top=.88,hspace=1.03);allrows=[]
    labels_map={'AMT_GOODS_PRICE':'Goods price','FLAG_DOCUMENT_16':'Document 16 flag','FLAG_DOCUMENT_18':'Document 18 flag','FLAG_DOCUMENT_13':'Document 13 flag','ORGANIZATION_TYPE_Transport: type 3':'Transport type 3 (org.)','YEARS_EMPLOYED':'Employment years','ORGANIZATION_TYPE':'Organization type','H_ACTIVE_NET_DEBT_CREDIT_RATIO_CUR1':'Active net debt / credit'}
    for ax,m,unit in zip(axes,['LR','RF','LGB'],['Coefficient (transformed term)','Impurity importance','Split gain (thousands)']):
        z=read(f'unified/results/{m}_importance.csv',f'{m}_importance.csv');z=z.iloc[np.argsort(z.value.abs())[-5:]].copy();z['model']=m;allrows.append(z)
        names=[c.removeprefix('numeric__') for c in z.feature];labs=[labels_map.get(c,c.replace('_',' ')) for c in names];values=z.value/(1000 if m=='LGB' else 1)
        ax.barh(labs,values,color=COLORS[m],height=.6);ax.axvline(0,color='#777777',lw=.6);ax.set(title=DISPLAY[m],xlabel=unit);ax.tick_params(axis='y',labelsize=7.5)
    panels(fig,axes);export(fig,'interpretation',pd.concat(allrows),'Top 5 by absolute within-model value, ascending display; full tables retained. Original C full-development models. LGB gain /1000 for axis only; no common scale or inference.')
    old=read('unified/results/holdout_metrics.csv','original_holdout.csv');new=read('unified/optimization_20261007/results/holdout_metrics.csv','optimized_holdout.csv')
    h=pd.concat([old.drop_duplicates('model'),new.drop_duplicates('model')],ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.4));fig.subplots_adjust(left=.15,right=.97,bottom=.27,top=.80,wspace=.52)
    labs=['LR','RF','Original\nLGB','Refined\nLGB','LGB–LR\nCDF']
    for ax,col,title in zip(axes,['roc_auc','ap'],['AUC (zoomed)','AP (zoomed)']):
        for i,r in enumerate(h.itertuples()):ax.plot(i,getattr(r,col),marker='o',ls='None',color=COLORS['LGB' if r.model=='LGB_optimized' else r.model])
        ax.tick_params(axis='x',labelsize=7)
        ax.set(xticks=range(5),xticklabels=labs,title=title,ylabel='Viewed holdout');ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'));ax.grid(axis='y',alpha=.15)
    panels(fig,axes);export(fig,'holdout',h,'One previously viewed 5% split, no SD or CI; threshold-independent metrics deduplicated across diagnostic thresholds; all five models retained.')
    (DATA/'figure_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
if __name__=='__main__':main()
