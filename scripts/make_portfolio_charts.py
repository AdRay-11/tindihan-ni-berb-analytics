import pickle, re, sys, warnings, shutil
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import os
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); from style import *; setup()
ROOT=os.path.dirname(HERE)
IN=os.path.join(ROOT,"outputs")+"/"; F=os.path.join(ROOT,"figures")+"/"; PH=os.path.join(ROOT,"figures","phase1_2_originals")+"/"
A=pickle.load(open(IN+"model_artifacts.pkl","rb"))
log=pd.read_csv(IN+"Model_Experiment_Log.csv")
def metric(s,key):
    m=re.search(rf"{key}=([-\d.]+)",s); return float(m.group(1)) if m else np.nan
def save(name): plt.savefig(F+name); plt.close()

# F01 cleaning funnel
steps=[("Raw rows",541909),("a. Cancellations set aside",-9288),("b. Duplicates removed",-5231),
       ("c. Missing CustomerID",-134658),("d. Qty/Price ≤ 0",-40),("e. Non-product codes",-1409),("Clean rows",391283)]
fig,ax=plt.subplots(figsize=(9,4))
run=0
for i,(lab,v) in enumerate(steps):
    if i==0 or i==len(steps)-1:
        ax.barh(i,v,color=BLUE,height=0.6); run=v if i==0 else run
        ax.text(v+5000,i,f"{v:,}",va="center",color=INK,fontsize=9,fontweight="bold")
    else:
        ax.barh(i,-v,left=run+v,color=ORANGE,height=0.6); run+=v
        ax.text(run-v+5000,i,f"−{-v:,}",va="center",color=INK2,fontsize=9)
ax.set_yticks(range(len(steps))); ax.set_yticklabels([s[0] for s in steps]); ax.invert_yaxis()
ax.set_xlim(0,640000); ax.xaxis.set_major_formatter(lambda x,p:f"{x/1000:.0f}k"); ax.grid(axis="y",visible=False)
ax.set_title("Data cleaning: 541,909 raw rows → 391,283 clean rows (72% kept)"); save("F01_cleaning_waterfall.png")

# F05 apriori grid heatmaps
g=log[log.Exp_ID.str.match(r"A(0[1-9]|1[0-6])$")].copy()
g["s"]=g.Hyperparameters.apply(lambda s:metric(s,"min_support")); g["c"]=g.Hyperparameters.apply(lambda s:metric(s,"min_confidence"))
g["rules"]=g.Metrics.apply(lambda s:metric(s,"rules")); g["lift"]=g.Metrics.apply(lambda s:metric(s,"mean_lift"))
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
for ax,col,ttl,fmt in [(axs[0],"rules","Number of rules","{:.0f}"),(axs[1],"lift","Mean lift","{:.1f}")]:
    p=g.pivot(index="s",columns="c",values=col)
    from matplotlib.colors import LinearSegmentedColormap
    cm=LinearSegmentedColormap.from_list("b",SEQ)
    ax.imshow(p.values,cmap=cm,aspect="auto")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v=p.values[i,j]; dark=v>p.values.mean()*1.15
            ax.text(j,i,fmt.format(v),ha="center",va="center",color="white" if dark else INK,fontsize=10)
    ax.set_xticks(range(4)); ax.set_xticklabels(p.columns); ax.set_yticks(range(4)); ax.set_yticklabels(p.index)
    ax.set_xlabel("min_confidence"); ax.set_ylabel("min_support"); ax.set_title(ttl); ax.grid(False)
    i,j=list(p.index).index(0.015),list(p.columns).index(0.6)
    ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,ec=ORANGE,lw=2.5))
fig.suptitle("Apriori tuning grid (16 runs) — orange box = selected setting (support 0.015, confidence 0.6)",x=0.01,ha="left",fontsize=10,color=INK2)
plt.tight_layout(); save("F05_apriori_grid.png")

# F06 top rules (pairwise, unique pairs, by lift)
r=A["association_rules"].copy()
r["a"]=r.antecedents.apply(lambda s:" + ".join(sorted(s))); r["c"]=r.consequents.apply(lambda s:" + ".join(sorted(s)))
def short(t): return (t.title().replace("Regency Teacup And Saucer","Regency Teacup").replace("Alarm Clock Bakelike","Alarm Clock")
    .replace("Gardeners Kneeling Pad","Kneeling Pad").replace("Lunch Bag  Black Skull.","Lunch Bag Black Skull").replace("Regency Cakestand 3 Tier","Regency Cakestand").replace("T-Light Holder","T-Light").replace("Paper Chain Kit 50'S Christmas","Paper Chain 50s Xmas").replace("Paper Chain Kit Vintage Christmas","Paper Chain Vintage Xmas"))
one=r[(r.antecedents.apply(len)==1)&(r.consequents.apply(len)==1)].copy()
one["key"]=one.apply(lambda x:frozenset([x.a,x.c]),axis=1)
one=one.sort_values("confidence",ascending=False).drop_duplicates("key").sort_values("lift",ascending=False).head(10)
fig,ax=plt.subplots(figsize=(9,4.6))
labels=[f"{short(a)}  →  {short(c)}" for a,c in zip(one.a,one.c)]
ax.barh(range(len(one)),one.lift,color=BLUE,height=0.6)
for i,(l,c) in enumerate(zip(one.lift,one.confidence)): ax.text(l+0.2,i,f"lift {l:.1f} · conf {c:.0%}",va="center",fontsize=8.5,color=INK2)
ax.set_yticks(range(len(one))); ax.set_yticklabels(labels,fontsize=9); ax.invert_yaxis(); ax.set_xlim(0,23)
ax.set_xlabel("Lift (times more likely than chance)"); ax.grid(axis="y",visible=False)
ax.set_title("Top 10 product pairs bought together (selected Apriori rules)"); save("F06_top_pairs.png")
one[["a","c","support","confidence","lift"]].to_csv(IN+"top_pairs.csv",index=False)

# F07 network
G=nx.DiGraph()
for _,x in r.iterrows():
    for a in x.antecedents:
        for c in x.consequents:
            w=max(G[a][c]["weight"],x.lift) if G.has_edge(a,c) else x.lift; G.add_edge(a,c,weight=w)
cen=A["product_centrality"]
comms=list(nx.weakly_connected_components(G)); comms.sort(key=len,reverse=True)
cols=[BLUE,ORANGE,AQUA,VIOLET,MAGENTA,GREEN,RED,YELLOW]
ccol={n:cols[i%len(cols)] for i,cc in enumerate(comms) for n in cc}
fig,ax=plt.subplots(figsize=(12,8.5))
pos={}
cells=[(0,0,2.0),(1.55,0,1.2),(2.65,0,1.0),(0.0,-1.5,0.9),(1.0,-1.35,1),(1.85,-1.35,1),(2.7,-1.35,1),(3.55,-1.35,1)]
for (cx,cy,sc),cc in zip(cells,comms):
    sub=G.subgraph(cc).to_undirected()
    pl=nx.kamada_kawai_layout(sub) if len(cc)>2 else {n:(0,0.5-i) for i,n in enumerate(sorted(cc))}
    xs=np.array(list(pl.values()),dtype=float); xs=xs-xs.mean(0); m=np.abs(xs).max(0); m[m==0]=1
    for n,v in zip(pl.keys(),xs): pos[n]=(cx+v[0]/m[0]*0.42*sc, cy+v[1]/m[1]*0.4*sc)
nx.draw_networkx_edges(G,pos,ax=ax,width=[0.5+G[u][v]["weight"]/7 for u,v in G.edges()],edge_color=MUTED,alpha=0.6,arrows=True,arrowsize=10,node_size=[400+6000*cen.loc[n,"pagerank"] for n in G.nodes()])
nx.draw_networkx_nodes(G,pos,ax=ax,node_size=[400+6000*cen.loc[n,"pagerank"] for n in G.nodes()],node_color=[ccol[n] for n in G.nodes()],edgecolors="white",linewidths=2)
for n,(x,y) in pos.items():
    ax.text(x,y-0.09,short(n),ha="center",va="top",fontsize=7.5,color=INK,bbox=dict(fc="white",ec="none",alpha=0.75,pad=0.5))
ax.set_title("Product network from 54 association rules — node size = PageRank, edge width = lift, colour = product cluster")
ax.axis("off"); save("F07_network.png")
pd.DataFrame({"cluster":[i+1 for i,cc in enumerate(comms) for n in cc],"product":[n for cc in comms for n in cc]}).to_csv(IN+"network_clusters.csv",index=False)

# F08 k selection
k=log[log.Exp_ID.str.match(r"K0[1-9]$")].copy()
k["k"]=k.Hyperparameters.apply(lambda s:metric(s,"n_clusters"))
for m in ["inertia","silhouette","davies_bouldin"]: k[m]=k.Metrics.apply(lambda s,m=m:metric(s,m))
fig,axs=plt.subplots(1,3,figsize=(12,3.4))
for ax,m,t in zip(axs,["inertia","silhouette","davies_bouldin"],["Elbow: inertia (lower = tighter)","Silhouette (higher = better)","Davies–Bouldin (lower = better)"]):
    ax.plot(k.k,k[m],color=BLUE,lw=2,marker="o",ms=6); ax.set_title(t,fontsize=10.5); ax.set_xlabel("k (clusters)")
    sel=k[k.k==4][m].iloc[0]; ax.plot(4,sel,"o",ms=11,mfc="none",mec=ORANGE,mew=2.5)
    ax.annotate("k = 4\nselected",(4,sel),xytext=(12,12),textcoords="offset points",fontsize=8.5,color=INK2)
    ax.set_xticks(range(2,11))
plt.tight_layout(); save("F08_k_selection.png")

# F09 segment shares
p=A["cluster_profile"].copy()
order=["Loyal / Big Spenders","Regular Buyers","New / Recent Low Spenders","Inactive / Lapsed"]
p=p.set_index("Segment").loc[order]
fig,ax=plt.subplots(figsize=(9,4))
y=np.arange(4); h=0.36
ax.barh(y-h/2,p["Customer_share_%"],h-0.04,color=BLUE,label="Share of customers")
ax.barh(y+h/2,p["Revenue_share_%"],h-0.04,color=ORANGE,label="Share of revenue")
for i,(c,rv) in enumerate(zip(p["Customer_share_%"],p["Revenue_share_%"])):
    ax.text(c+0.8,i-h/2,f"{c:.1f}%",va="center",fontsize=9,color=INK2); ax.text(rv+0.8,i+h/2,f"{rv:.1f}%",va="center",fontsize=9,color=INK)
ax.set_yticks(y); ax.set_yticklabels([f"{s}\n({int(n):,} customers)" for s,n in zip(order,p.Customers)]); ax.invert_yaxis()
ax.set_xlim(0,75); ax.xaxis.set_major_formatter(lambda x,_:f"{x:.0f}%"); ax.grid(axis="y",visible=False)
ax.legend(loc="lower right"); ax.set_title("16% of customers bring in 64% of revenue (k-Means, k = 4)"); save("F09_segment_shares.png")

# F09b segment RFM medians small multiples
fig,axs=plt.subplots(1,3,figsize=(12,3.3))
segc=[BLUE,AQUA,YELLOW,ORANGE]
for ax,(col,t,f) in zip(axs,[("Recency_median","Median days since last order","{:.0f} d"),("Frequency_median","Median number of orders","{:.0f}"),("Monetary_median","Median total spend","{:,.0f}")]):
    ax.barh(range(4),p[col],color=BLUE,height=0.6)
    for i,v in enumerate(p[col]): ax.text(v,i,"  "+f.format(v),va="center",fontsize=9,color=INK2)
    ax.set_yticks(range(4)); ax.set_yticklabels(order if ax is axs[0] else []); ax.invert_yaxis(); ax.set_title(t,fontsize=10.5)
    ax.set_xlim(0,p[col].max()*1.3); ax.grid(axis="y",visible=False)
plt.tight_layout(); save("F10b_segment_rfm.png")

# F11 classifier comparison (CV)
cv=log[log.Exp_ID.str.match(r"CV\d")].copy()
cv["name"]=cv.Model.str.replace("5-fold CV: ","").str.replace(r" \(from .*\)","",regex=True).str.replace("Neural Network (MLPClassifier)","Neural Network (MLP)").str.replace("Linear Discriminant Analysis","LDA")
def ms(s,k):
    m=re.search(rf"{k}=([\d.]+)\+-([\d.]+)",s); return float(m.group(1)),float(m.group(2))
for m in ["roc_auc","f1","recall"]:
    cv[m]=cv.Metrics.apply(lambda s,m=m:ms(s,m)[0]); cv[m+"_sd"]=cv.Metrics.apply(lambda s,m=m:ms(s,m)[1])
cv=cv.sort_values("roc_auc")
fig,axs=plt.subplots(1,2,figsize=(11,3.4),sharey=True)
for ax,m,t,base in [(axs[0],"roc_auc","ROC-AUC (5-fold CV, mean ± SD)",0.5),(axs[1],"f1","F1 for 'Cancel' class (5-fold CV)",0.0)]:
    colr=[ORANGE if n=="Logistic Regression" else BLUE for n in cv.name]
    ax.barh(range(len(cv)),cv[m],xerr=cv[m+"_sd"],color=colr,height=0.6,error_kw=dict(ecolor=INK2,lw=1,capsize=3))
    for i,(v,sd) in enumerate(zip(cv[m],cv[m+"_sd"])): ax.text(v+sd+0.01,i,f"{v:.3f}",va="center",fontsize=9,color=INK)
    if base: ax.axvline(base,color=MUTED,ls="--",lw=1); ax.text(base+0.005,3.45,"baseline 0.50",fontsize=8,color=MUTED)
    ax.set_xlim(base if base else 0, 0.9 if base else 0.8); ax.set_title(t,fontsize=10.5); ax.grid(axis="y",visible=False)
axs[0].set_yticks(range(len(cv))); axs[0].set_yticklabels(cv.name)
plt.tight_layout(); save("F11_classifier_cv.png")

# F11b all 21 validation runs
pr=log[log.Exp_ID.str.match(r"P(0[1-9]|1\d|2[01])$")].copy()
pr["val_auc"]=pr.Metrics.apply(lambda s:metric(s.split("||")[1],"roc_auc")); pr["val_f1"]=pr.Metrics.apply(lambda s:metric(s.split("||")[1],"f1"))
pr["train_f1"]=pr.Metrics.apply(lambda s:metric(s.split("||")[0],"f1"))
fam={"Logistic Regression":BLUE,"Linear Discriminant Analysis":AQUA,"Decision Tree":ORANGE,"Neural Network (MLPClassifier)":VIOLET}
fig,ax=plt.subplots(figsize=(9,4))
for f,c in fam.items():
    d=pr[pr.Model==f]; ax.scatter(d.val_auc,d.val_f1,s=60,color=c,edgecolor="white",lw=1.5,label=f.replace(" (MLPClassifier)",""),zorder=3)
b=pr[pr.Exp_ID=="P08"].iloc[0]; ax.annotate("P08 selected\nLogReg C=10, balanced",(b.val_auc,b.val_f1),xytext=(-150,-10),textcoords="offset points",fontsize=8.5,color=INK2,arrowprops=dict(arrowstyle="-",color=MUTED))
ax.set_xlabel("Validation ROC-AUC"); ax.set_ylabel("Validation F1"); ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),fontsize=8.5,ncol=4)
ax.set_title("All 21 classifier runs on the validation set"); save("F11b_val_runs.png")
pr[["Exp_ID","Model","Hyperparameters","train_f1","val_f1","val_auc"]].to_csv(IN+"classifier_runs.csv",index=False)

# F12 confusion matrix (test)
cm=np.array([[398,158],[92,219]])
fig,ax=plt.subplots(figsize=(5.2,4.4))
from matplotlib.colors import LinearSegmentedColormap
ax.imshow(cm,cmap=LinearSegmentedColormap.from_list("b",SEQ[:5]),vmin=0,vmax=450)
lab=[["True negative","False positive"],["False negative","True positive"]]
for i in range(2):
    for j in range(2):
        ax.text(j,i-0.08,f"{cm[i,j]}",ha="center",va="center",fontsize=20,fontweight="bold",color="white" if cm[i,j]>300 else INK)
        ax.text(j,i+0.2,lab[i][j],ha="center",va="center",fontsize=8.5,color="white" if cm[i,j]>300 else INK2)
ax.set_xticks([0,1]); ax.set_xticklabels(["No cancel","Cancel"]); ax.set_yticks([0,1]); ax.set_yticklabels(["No cancel","Cancel"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
ax.set_title("Confusion matrix — test set (n = 867)\nLogistic Regression P08",fontsize=11); save("F12_confusion_matrix.png")

# F13 ROC (crop original)
im=Image.open(PH+"nb_roc_confusion.png"); w,h=im.size; im.crop((int(w*0.5),0,w,h)).save(F+"F13_roc_curves.png")

# F14 feature importance
clf=A["classifier"].named_steps["m"]; feats=A["classifier_features"]
coef=pd.Series(clf.coef_[0],index=feats)
dt=pd.Series([0.036,0.175,0.667,0.069,0.052],index=feats)
fig,axs=plt.subplots(1,2,figsize=(11,3.3))
o=coef.sort_values()
axs[0].barh(o.index,o.values,color=[ORANGE if v<0 else BLUE for v in o.values],height=0.6); axs[0].axvline(0,color=MUTED,lw=1)
for i,v in enumerate(o.values): axs[0].text(v+(0.03 if v>=0 else -0.03),i,f"{v:+.2f}",va="center",ha="left" if v>=0 else "right",fontsize=9,color=INK)
axs[0].set_xlim(-0.6,1.25)
axs[0].set_title("Logistic Regression coefficients (scaled features)",fontsize=10.5)
o2=dt.sort_values(); axs[1].barh(o2.index,o2.values,color=BLUE,height=0.6)
for i,v in enumerate(o2.values): axs[1].text(v+0.01,i,f"{v:.2f}",va="center",fontsize=9)
axs[1].set_xlim(0,0.8); axs[1].set_title("Decision Tree (depth 6) feature importance",fontsize=10.5)
for a in axs: a.grid(axis="y",visible=False)
plt.tight_layout(); save("F14_feature_importance.png")

# F16 regression coefs
lin=A["linear_regression"].named_steps["m"]; rc=pd.Series(lin.coef_,index=A["regression_features"]).sort_values()
fig,ax=plt.subplots(figsize=(6,3))
ax.barh(rc.index,rc.values,color=BLUE,height=0.6)
for i,v in enumerate(rc.values): ax.text(v+0.01,i,f"{v:.2f}",va="center",fontsize=9)
ax.set_xlim(0,0.95); ax.grid(axis="y",visible=False)
ax.set_title("What drives spending? Linear regression on log(Monetary)\n(test R² = 0.877; coefficient per 1 SD)",fontsize=10.5); save("F16_regression_coefs.png")

# Phase 1/2 originals (charts whose raw data is not in the outputs folder)
for src,dst in [("p1_monthly_revenue.png","F02_monthly_revenue.png"),("p1_orders_weekday_hour.png","F03_orders_weekday_hour.png"),
                ("p1_spend_distribution.png","F04b_spend_distribution.png"),("nb_correlation.png","F04_correlation.png"),
                ("nb_segment_scatter.png","F10_segment_scatter.png"),("nb_decision_tree.png","F15_decision_tree.png"),
                ("nb_regression_fit.png","F16b_regression_fit.png"),("p1_top_cancelled.png","F17_top_cancelled.png"),("p1_outliers_boxplot.png","F00_outliers_boxplot.png"),("p1_corr_raw.png","F00b_corr_raw.png")]:
    shutil.copy(PH+src,F+dst)
print("done")

# F18 executive dashboard
fig=plt.figure(figsize=(14,8.6))
fig.text(0.03,0.965,"Tindihan ni BERB — Executive analytics dashboard (Dec 2010 – Dec 2011)",fontsize=16,fontweight="bold",color=INK,va="top")
kpis=[("391,283","clean order lines","from 541,909 raw rows"),("4,334","customers profiled","4 RFM segments (k-Means)"),
      ("64%","of revenue from 16% of customers","Loyal / Big Spenders"),("0.77","test ROC-AUC, cancel-risk model","catches 70% of cancellers")]
for i,(v,l,sub) in enumerate(kpis):
    ax=fig.add_axes([0.03+i*0.2425,0.77,0.225,0.14]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=0.06",fc="#f4f3f0",ec="none",transform=ax.transAxes))
    ax.text(0.07,0.66,v,fontsize=24,fontweight="bold",color=INK,transform=ax.transAxes,va="center")
    ax.text(0.07,0.33,l,fontsize=10,color=INK,transform=ax.transAxes,va="center"); ax.text(0.07,0.13,sub,fontsize=8.5,color=MUTED,transform=ax.transAxes,va="center")
ax=fig.add_axes([0.03,0.40,0.43,0.31]); im=Image.open(PH+"p1_monthly_revenue.png"); ax.imshow(im); ax.axis("off")
fig.text(0.03,0.725,"Revenue peaks Sep–Nov (pre-holiday season)",fontsize=11,fontweight="bold")
ax=fig.add_axes([0.70,0.42,0.27,0.28])
ax.barh(np.arange(4)-0.18,p["Customer_share_%"],0.34,color=BLUE,label="Customers"); ax.barh(np.arange(4)+0.18,p["Revenue_share_%"],0.34,color=ORANGE,label="Revenue")
ax.set_yticks(range(4)); ax.set_yticklabels(order,fontsize=9); ax.invert_yaxis(); ax.xaxis.set_major_formatter(lambda x,_:f"{x:.0f}%"); ax.legend(fontsize=8.5,loc="lower right"); ax.grid(axis="y",visible=False)
fig.text(0.53,0.725,"Segment share of customers vs revenue",fontsize=11,fontweight="bold")
ax=fig.add_axes([0.25,0.06,0.23,0.27])
o5=one.head(6); ax.barh(range(6),o5.lift,color=BLUE,height=0.6); ax.set_yticks(range(6)); ax.set_yticklabels([f"{short(a)} → {short(c)}" for a,c in zip(o5.a,o5.c)],fontsize=8.5); ax.invert_yaxis()
for k_,v in enumerate(o5.lift): ax.text(v+0.3,k_,f"{v:.1f}",va="center",fontsize=8.5)
ax.set_xlim(0,21); ax.set_xlabel("lift",fontsize=9); ax.grid(axis="y",visible=False)
fig.text(0.03,0.36,"Strongest bundle candidates (lift)",fontsize=11,fontweight="bold")
ax=fig.add_axes([0.70,0.06,0.27,0.27])
ax.barh(range(len(cv)),cv.roc_auc,color=[ORANGE if n=="Logistic Regression" else BLUE for n in cv.name],height=0.6)
ax.set_yticks(range(len(cv))); ax.set_yticklabels(cv.name,fontsize=9); ax.set_xlim(0.5,0.85); ax.grid(axis="y",visible=False)
for k_,v in enumerate(cv.roc_auc): ax.text(v+0.004,k_,f"{v:.3f}",va="center",fontsize=8.5)
fig.text(0.53,0.36,"Cancel-risk models: 5-fold CV ROC-AUC (orange = selected)",fontsize=11,fontweight="bold")
plt.savefig(F+"F18_dashboard.png",dpi=200); plt.close()
print("dash")
