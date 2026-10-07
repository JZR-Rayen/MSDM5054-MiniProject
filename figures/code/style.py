"""Shared course figures, sized for the unchanged NeurIPS 5.5-inch text width."""
import matplotlib as mpl
COLORS={'LR':'#0072B2','RF':'#D55E00','LGB':'#009E73','Fusion':'#CC79A7'}
DISPLAY={'LR':'LR','RF':'RF','LGB':'LightGBM','Fusion':'Fusion'}
MARKERS=['o','s','^']
def apply():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'legend.fontsize':7.5,'axes.linewidth':.7,'lines.linewidth':1.1,'lines.markersize':4,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.dpi':600})
