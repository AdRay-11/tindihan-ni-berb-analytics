"""Phase 3 charts that need the Phase 1 clean data (data/clean_transactions.csv etc.)."""
import os,sys,pickle,warnings; warnings.filterwarnings("ignore")
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); from style import *; setup()
ROOT=os.path.dirname(HERE)
DATA=os.path.join(ROOT,"data"); OUT=os.path.join(ROOT,"figures"); PKL=os.path.join(ROOT,"outputs","model_artifacts.pkl")
def save(n): plt.savefig(os.path.join(OUT,n)); plt.close()
df=pd.read_csv(os.path.join(DATA,"clean_transactions.csv"),dtype={"InvoiceNo":str,"CustomerID":str},parse_dates=["InvoiceDate"])
rfm=pd.read_csv(os.path.join(DATA,"customer_rfm.csv")); canc=pd.read_csv(os.path.join(DATA,"cancellations.csv"),dtype={"InvoiceNo":str})
A=pickle.load(open(PKL,"rb"))
def short(t): return t.title().replace("T-Light Holder","T-Light").replace("Assorted Colour","Asst Colour")
# reversed bulk orders: an order whose exact product+quantity+customer was cancelled within the same day
REV=["581483","541431"]

# F02 monthly revenue
m=df.groupby("YearMonth").TotalPrice.sum()/1e6
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(range(len(m)),m.values,color=BLUE,lw=2.2,marker="o",ms=6)
ax.axvspan(8.6,11.4,color="#fdeee6",zorder=0); ax.text(10,0.38,"Sep–Nov peak",ha="center",fontsize=9,color=INK2)
ax.annotate(f"Nov 2011: {m.iloc[11]:.2f}M",(11,m.iloc[11]),xytext=(-110,4),textcoords="offset points",fontsize=9,color=INK)
ax.text(12,m.iloc[12]-0.06,"data ends\n9 Dec",ha="center",va="top",fontsize=8.5,color=MUTED)
ax.set_xticks(range(len(m))); ax.set_xticklabels([pd.Period(p).strftime("%b\n%Y") if i in (0,1,12) else pd.Period(p).strftime("%b") for i,p in enumerate(m.index)],fontsize=8.5)
ax.set_ylim(0.3,1.25); ax.set_ylabel("Revenue (millions)"); ax.grid(axis="x",visible=False)
ax.set_title("Monthly revenue, Dec 2010 – Dec 2011"); save("F02_monthly_revenue.png")

# F02b top products (net of the two reversed bulk orders)
net=df[~df.InvoiceNo.isin(REV)]
r=net.groupby("Description").TotalPrice.sum().nlargest(10); q=net.groupby("Description").Quantity.sum().nlargest(10)
fig,axs=plt.subplots(1,2,figsize=(13,4.6))
for ax,s,t,f in [(axs[0],r/1000,"Top 10 products by revenue (thousands)","{:.0f}k"),(axs[1],q/1000,"Top 10 products by units sold (thousands)","{:.1f}k")]:
    ax.barh(range(10),s.values,color=BLUE,height=0.6); ax.set_yticks(range(10)); ax.set_yticklabels([short(x) for x in s.index],fontsize=9); ax.invert_yaxis()
    for i,v in enumerate(s.values): ax.text(v+s.max()*0.01,i,f.format(v),va="center",fontsize=8.5,color=INK2)
    ax.set_xlim(0,s.max()*1.15); ax.grid(axis="y",visible=False); ax.set_title(t,fontsize=11)
fig.text(0.01,-0.02,"Excludes two bulk orders that were cancelled the same day (invoices 541431 and 581483; 245,653 in value).",fontsize=8.5,color=MUTED)
plt.tight_layout(); save("F02b_top_products.png")
pd.DataFrame({"product":r.index,"net_revenue":r.values}).to_csv(os.path.join(ROOT,"outputs","top_products_revenue.csv"),index=False)

# F03 weekday/hour
order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["DayOfWeek"]=df.InvoiceDate.dt.day_name(); df["Hour"]=df.InvoiceDate.dt.hour
wd=df.groupby("DayOfWeek").InvoiceNo.nunique().reindex(order).fillna(0); hr=df.groupby("Hour").InvoiceNo.nunique()
fig,axs=plt.subplots(1,2,figsize=(12,3.6),gridspec_kw={"width_ratios":[1,1.6]})
axs[0].bar([d[:3] for d in order],wd.values,color=[ORANGE if d=="Thursday" else BLUE for d in order],width=0.6); axs[0].set_title("Orders per weekday (no Saturday orders)",fontsize=11)
axs[1].bar(hr.index,hr.values,color=[ORANGE if h==12 else BLUE for h in hr.index],width=0.7); axs[1].set_title("Orders per hour of day (peak at 12 noon)",fontsize=11); axs[1].set_xticks(hr.index)
for a in axs: a.grid(axis="x",visible=False)
plt.tight_layout(); save("F03_orders_weekday_hour.png")

# F04 correlation (log features)
lg=rfm.drop(columns="CustomerID").apply(np.log1p); c=lg.corr()
cm=LinearSegmentedColormap.from_list("d",["#eb6834","#f3f2ee","#2a78d6"])
fig,ax=plt.subplots(figsize=(6.4,5)); im=ax.imshow(c.values,cmap=cm,vmin=-1,vmax=1)
for i in range(5):
    for j in range(5): ax.text(j,i,f"{c.values[i,j]:.2f}",ha="center",va="center",fontsize=10,color="white" if abs(c.values[i,j])>0.6 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels(c.columns,rotation=30,ha="right"); ax.set_yticks(range(5)); ax.set_yticklabels(c.columns); ax.grid(False)
plt.colorbar(im,fraction=0.046,pad=0.04); ax.set_title("Correlation of log customer features",fontsize=11); save("F04_correlation.png")

# F13 test ROC
d=rfm.copy()
for k in ["Recency","Frequency","Monetary","AvgBasketSize","UniqueProducts"]: d[k]=np.log1p(d[k])
y=rfm.CustomerID.isin(set(canc.CustomerID.dropna().astype(int))).astype(int)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
_,Xte,_,yte=train_test_split(d[A["classifier_features"]],y,test_size=0.2,stratify=y,random_state=42)
p=A["classifier"].predict_proba(Xte)[:,1]; fpr,tpr,_=roc_curve(yte,p); auc=roc_auc_score(yte,p)
fig,ax=plt.subplots(figsize=(5.4,4.6))
ax.plot([0,1],[0,1],ls="--",color=MUTED,lw=1); ax.text(0.55,0.47,"random guess (AUC 0.50)",fontsize=8.5,color=MUTED,rotation=38)
ax.plot(fpr,tpr,color=BLUE,lw=2.2)
ax.plot(158/556,219/311,"o",ms=9,color=ORANGE,mec="white",mew=1.5); ax.annotate("cutoff 0.5:\nrecall 0.70, FPR 0.28",(158/556,219/311),xytext=(18,-34),textcoords="offset points",fontsize=8.5,color=INK2)
ax.text(0.6,0.12,f"Test AUC = {auc:.3f}",fontsize=11,fontweight="bold",color=INK)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)"); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
ax.set_title("ROC curve — test set (n = 867)\nLogistic Regression P08",fontsize=11); save("F13c_roc_test.png")

# F17 top cancelled products
cc=canc.groupby("Description").Quantity.sum().nlargest(10)/1000
fig,ax=plt.subplots(figsize=(9,4.2))
ax.barh(range(10),cc.values,color=[ORANGE if v>50 else BLUE for v in cc.values],height=0.6); ax.set_yticks(range(10)); ax.set_yticklabels([short(x) for x in cc.index],fontsize=9); ax.invert_yaxis()
for i,v in enumerate(cc.values): ax.text(v+0.8,i,f"{v:.1f}k",va="center",fontsize=8.5,color=INK2)
ax.set_xlim(0,95); ax.grid(axis="y",visible=False); ax.set_xlabel("Cancelled units (thousands)")
ax.set_title("Top 10 most-cancelled products — two single bulk orders dominate (orange)",fontsize=11); save("F17_top_cancelled.png")
print("extra charts ok", round(auc,4))
