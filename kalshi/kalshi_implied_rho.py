"""Kalshi market-implied dependence between US Senate races (2026 midterms).

Reads daily Kalshi prices (data/kalshi_dem_chances.csv: Democratic win chance per
race, in %; data/kalshi_senate_control.csv: Democratic Senate-control price, in %),
and for each day:

  * Q(rho=0): chance of 51+ seats if the 35 races were independent
    (Poisson-binomial, no copula needed),
  * Q(rho=1): chance if one shared swing decided every race,
  * implied rho: the one-factor Gaussian-copula correlation that makes the 35
    race prices reproduce the control price (no solution when the market sits
    outside the [rho=0, rho=1] band, as before mid-September 2026),
  * expected Democratic seats and their standard deviation at rho = 0 and at
    the implied rho.

Democrats hold 34 seats not up in 2026 and need 51 (a 50-50 tie goes to the
Republican Vice President), so they need 17 of the 35 races.

Run from this folder:  python kalshi_implied_rho.py
Outputs: results/control_series.csv, charts/*.png
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import textwrap

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; OUT = HERE / "results"; FIG = HERE / "charts"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
REGIME_START = pd.Timestamp("2026-09-15")   # first day an implied rho exists

# ---------------------------------------------------------------------------
# 1. Data and the one-factor Gaussian copula
# ---------------------------------------------------------------------------

P = pd.read_csv(DATA/"kalshi_dem_chances.csv", parse_dates=["date"]).set_index("date")
C = pd.read_csv(DATA/"kalshi_senate_control.csv", parse_dates=["date"]).set_index("date")["Dem Control %"]/100
P = P/100
NEED = 17
x, w = np.polynomial.hermite_e.hermegauss(80); w = w/w.sum()
def pb(p):  # poisson-binomial pmf
    f = np.zeros(len(p)+1); f[0]=1
    for q in p: f[1:] = f[1:]*(1-q)+f[:-1]*q; f[0]*=(1-q)
    return f
def dist(p, rho):
    p = np.clip(p,1e-6,1-1e-6)
    if rho<=0: return pb(p)
    if rho>=1:
        f=np.zeros(len(p)+1); s=np.sort(p)[::-1]; s=np.r_[1,s,0]
        for k in range(len(p)+1): f[k]=s[k]-s[k+1]
        return f
    t = norm.ppf(p); out=np.zeros(len(p)+1)
    for m,wi in zip(x,w):
        out += wi*pb(norm.cdf((t-np.sqrt(rho)*m)/np.sqrt(1-rho)))
    return out
Q = lambda p,r: dist(p,r)[NEED:].sum()
grid = np.r_[0,np.linspace(0.01,0.99,99)]
rows=[]
for dt,row in P.iterrows():
    if dt not in C.index or row.isna().any(): continue
    p=row.values; c=C[dt]
    q=np.array([Q(p,r) for r in grid])
    g=q-c; rho=np.nan
    idx=np.where(np.sign(g[:-1])!=np.sign(g[1:]))[0]
    if len(idx):
        i=idx[0]; lo,hi=grid[i],grid[i+1]
        for _ in range(30):
            mid=(lo+hi)/2
            if np.sign(Q(p,mid)-c)==np.sign(g[i]): lo=mid
            else: hi=mid
        rho=(lo+hi)/2
    rows.append(dict(date=dt,market=c,q0=q[0],q1=Q(p,1),rho=rho,ed=34+p.sum()))

R = pd.DataFrame(rows).set_index("date")

# ---------------------------------------------------------------------------
# 2. Seat mean and standard deviation per day (rho = 0, implied, rho = 1)
# ---------------------------------------------------------------------------
s = 34 + np.arange(36)
def mean_sd(f):
    m = (f * s).sum(); return m, np.sqrt((f * (s - m) ** 2).sum())
extra = []
for dt, r in R.iterrows():
    p = P.loc[dt].values
    m0, s0 = mean_sd(dist(p, 0)); _, s1 = mean_sd(dist(p, 1))
    si = mean_sd(dist(p, r.rho))[1] if not np.isnan(r.rho) else np.nan
    extra.append(dict(date=dt, mean=m0, sd0=s0, sd1=s1, sdimp=si))
R = R.join(pd.DataFrame(extra).set_index("date"))
R.to_csv(OUT / "control_series.csv")
last = R.iloc[-1]
print(R.tail().round(3))
print(f"\nLatest ({R.index[-1].date()}): control {last.market:.1%}, independent {last.q0:.1%}, "
      f"rho=1 {last.q1:.1%}, implied rho {last.rho:.3f}, E[seats] {last['mean']:.2f}, "
      f"sd {last.sd0:.2f} (rho=0) vs {last.sdimp:.2f} (implied)")

# ---------------------------------------------------------------------------
# 3. Landscape chart (both panels)
# ---------------------------------------------------------------------------
from matplotlib import font_manager as fm
def landscape_chart(R):
    
    for f in fm.findSystemFonts("/usr/share/fonts/opentype/inter"): fm.fontManager.addfont(f)
    mpl.rcParams.update({"font.family":"Inter","font.size":12,"axes.edgecolor":"#c9c8c2","axes.linewidth":0.8,
     "xtick.color":"#52514e","ytick.color":"#52514e","xtick.major.size":0,"ytick.major.size":0,"text.color":"#0b0b0b"})
    INK="#0b0b0b"; SEC="#52514e"; MUT="#8a8984"; GRID="#ebeae6"; SURF="#fcfcfb"; REG="#efeee9"
    BLUE="#2a78d6"; ORG="#eb6834"; AQUA="#1baf7a"; G1="#b9b8b2"; G2="#9d9c96"; G3="#7c7b76"
    
    
    R=R[R.index>="2026-03-01"]
    B=pd.Timestamp("2026-09-15")
    R["sdimp_s"]=R.sdimp.rolling(5,min_periods=3).mean()
    pre=R[R.index<=B]; post=R[R.index>=B]; fl=R.iloc[-1]
    fig=plt.figure(figsize=(16,9),dpi=150,facecolor=SURF)
    gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],left=0.055,right=0.90,top=0.70,bottom=0.13,wspace=0.28)
    a=fig.add_subplot(gs[0]); b=fig.add_subplot(gs[1],sharex=a)
    fig.text(0.055,0.935,"Democrats have a 70% chance of Senate control, but it is a digital bet with big uncertainty",fontsize=22,weight="bold")
    fig.text(0.055,0.885,"Since mid-September the Democratic chance has entered a new regime, driven by a national mood that moves every seat together.",fontsize=14,color=SEC)
    fig.text(0.055,0.855,"That co-movement leaves the expected seat count unchanged but widens the spread: had the gains come seat by seat, it would be less than half as wide.",fontsize=14,color=SEC)
    fl=R.iloc[-1]
    for ax in (a,b):
        ax.set_facecolor(SURF)
        for s in ["top","right","left"]: ax.spines[s].set_visible(False)
        ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
        pass
        ax.axvline(B,color=MUT,lw=1,ls=(0,(3,3)),zorder=1)
        ax.set_xticks(pd.date_range("2026-03-01","2026-11-01",freq="MS"))
        ax.set_xticklabels([d.strftime("%b") for d in pd.date_range("2026-03-01","2026-11-01",freq="MS")])
        ax.set_xlim(R.index[0],R.index[-1]+pd.Timedelta(days=4))
    # A
    a.text(0,1.13,"A   Chance of Democratic control",transform=a.transAxes,fontsize=14,weight="bold")
    a.text(0,1.06,"Kalshi market vs. what the 35 race prices imply at ρ = 0 and ρ = 1",transform=a.transAxes,fontsize=12,color=SEC)
    a.fill_between(pre.index,np.minimum(pre.q0,pre.q1)*100,np.maximum(pre.q0,pre.q1)*100,color="#e9e8e3",lw=0,zorder=1); a.plot(pre.index,pre.q0*100,color=G1,lw=1.8); a.plot(pre.index,pre.q1*100,color=G2,lw=1.8); a.plot(pre.index,pre.market*100,color=G3,lw=2.6)
    a.fill_between(post.index,np.minimum(post.q0,post.q1)*100,np.maximum(post.q0,post.q1)*100,color="#dfe7f1",lw=0,zorder=1)
    a.plot(post.index,post.q0*100,color=ORG,lw=2,zorder=3); a.plot(post.index,post.q1*100,color=AQUA,lw=2,zorder=3)
    a.plot(post.index,post.market*100,color=BLUE,lw=3,zorder=4)
    x=R.index[-1]
    for y,c in [(fl.q0,ORG),(fl.q1,AQUA),(fl.market,BLUE)]: a.scatter([x],[y*100],s=55,color=c,edgecolor=SURF,lw=2,zorder=5)
    dx=pd.Timedelta(days=5)
    a.text(x+dx,fl.q0*100,f"{fl.q0*100:.0f}%  ρ = 0",va="center",fontsize=11.5)
    a.text(x+dx,fl.market*100+1.5,f"{fl.market*100:.1f}%  Market",va="center",fontsize=11.5,weight="bold")
    a.text(x+dx,fl.q1*100-2.5,f"{fl.q1*100:.1f}%  ρ = 1",va="center",fontsize=11.5)
    a.set_ylim(0,100); a.set_yticks(range(0,101,25)); a.set_yticklabels([f"{v}%" for v in range(0,101,25)])
    a.text(pd.Timestamp("2026-03-08"),14,"BEFORE MID-SEPTEMBER",fontsize=10,color=MUT,weight="semibold")
    a.text(pd.Timestamp("2026-03-08"),11,"Low national mood:\nlittle co-movement between seats.",fontsize=10.5,color=SEC,va="top")
    a.text(B+pd.Timedelta(days=4),14,"STRONG NATIONAL MOOD",fontsize=10,color=BLUE,weight="semibold")
    a.text(B+pd.Timedelta(days=4),11,"Market sits inside the band:\nseats priced as moving together.",fontsize=10.5,color=SEC,va="top")
    # B
    b.text(0,1.13,"B   Expected Democratic seats",transform=b.transAxes,fontsize=14,weight="bold")
    b.text(0,1.06,"Shaded: expected seats ± 1 std. dev.",transform=b.transAxes,fontsize=12,color=SEC)
    b.plot(pre.index,pre["mean"],color=G3,lw=2.4,zorder=4)
    b.fill_between(post.index,post["mean"]-post.sdimp_s,post["mean"]+post.sdimp_s,color=BLUE,alpha=0.18,lw=0,zorder=2)
    b.fill_between(post.index,post["mean"]-post.sd0,post["mean"]+post.sd0,color=ORG,alpha=0.35,lw=0,zorder=3)
    b.plot(post.index,post["mean"],color=INK,lw=2.6,zorder=4)
    b.axhline(50.5,color=INK,lw=1,zorder=1)
    b.set_ylim(44,60); b.set_yticks(range(44,61,4))
    for y,lab,c,w in [(fl["mean"],f"{fl['mean']:.1f}","#0b0b0b","bold"),
                      (fl["mean"]+fl.sdimp_s,f"{fl['mean']+fl.sdimp_s:.1f}",BLUE,"normal"),(fl["mean"]-fl.sdimp_s,f"{fl['mean']-fl.sdimp_s:.1f}",BLUE,"normal")]:
        b.text(x+dx,y,lab,va="center",fontsize=11.5,weight=w,color=INK)
    b.text(x+dx,50.5,"51-seat line",va="center",fontsize=10.5,color=SEC)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=BLUE,alpha=0.18,label="Market-implied \u03c1"),Patch(color=ORG,alpha=0.35,label="\u03c1 = 0, independent")],
             loc="lower left",bbox_to_anchor=(0.01,0.01),ncol=3,frameon=False,fontsize=10.5,handlelength=1.2,columnspacing=1.2,borderaxespad=0.2)
    b.text(pd.Timestamp("2026-03-08"),58.6,"BEFORE MID-SEPTEMBER",fontsize=10,color=MUT,weight="semibold")
    b.text(pd.Timestamp("2026-03-08"),58.1,"Low national mood:\nlittle co-movement between seats.",fontsize=10.5,color=SEC,va="top")
    b.text(B+pd.Timedelta(days=4),58.6,"NATIONAL MOOD",fontsize=10,color=BLUE,weight="semibold")
    b.text(B+pd.Timedelta(days=4),58.1,"±1.7 seats → ±3.7",fontsize=10.5,color=SEC,va="top")
    fig.text(0.055,0.035,"Source: Kalshi Democratic win prices for 35 Senate races and the Senate-control market, daily close; implied band uses a 5-day average. Model: one-factor Gaussian copula; ρ is the share of each race driven by a\ncommon national factor, and implied ρ is the value that makes the 35 race prices reproduce the control price. Democrats hold 34 seats not up in 2026 and need 51 (a 50-50 tie goes to the Republican VP).",fontsize=9.5,color=MUT)
    fig.savefig(str(FIG / "senate_control_risk_chart.png"),facecolor=SURF)

# ---------------------------------------------------------------------------
# 4. Two portrait (4:5) charts for social media
# ---------------------------------------------------------------------------
def portrait_charts(R):
    
    for f in fm.findSystemFonts("/usr/share/fonts/opentype/inter"): fm.fontManager.addfont(f)
    mpl.rcParams.update({"font.family":"Inter","font.size":15,"axes.edgecolor":"#c9c8c2","axes.linewidth":0.8,
     "xtick.color":"#52514e","ytick.color":"#52514e","xtick.major.size":0,"ytick.major.size":0,"text.color":"#0b0b0b",
     "xtick.labelsize":14,"ytick.labelsize":14})
    INK="#0b0b0b"; SEC="#52514e"; MUT="#8a8984"; GRID="#ebeae6"; SURF="#fcfcfb"
    BLUE="#2a78d6"; ORG="#eb6834"; AQUA="#1baf7a"; G1="#b9b8b2"; G2="#9d9c96"; G3="#7c7b76"
    
    
    R=R[R.index>="2026-03-01"]; R["sdimp_s"]=R.sdimp.rolling(5,min_periods=3).mean()
    B=pd.Timestamp("2026-09-15"); pre=R[R.index<=B]; post=R[R.index>=B]; fl=R.iloc[-1]
    months=pd.date_range("2026-03-01","2026-11-01",freq="MS")
    def frame(title,sub,head,headsub,suby=0.775):
        fig=plt.figure(figsize=(7.2,9),dpi=150,facecolor=SURF)
        fig.text(0.07,0.945,title,fontsize=22,weight="bold",va="top",linespacing=1.15)
        fig.text(0.07,suby,sub,fontsize=14,color=SEC,va="top",linespacing=1.35)
        ax=fig.add_axes([0.10,0.15,0.66,0.43]); ax.set_facecolor(SURF)
        fig.text(0.07,0.635,head,fontsize=16,weight="bold"); fig.text(0.07,0.605,headsub,fontsize=13,color=SEC)
        for s in ["top","right","left"]: ax.spines[s].set_visible(False)
        ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="x",pad=10)
        ax.axvline(B,color=MUT,lw=1,ls=(0,(3,3)),zorder=1)
        ax.set_xticks(months[::2]); ax.set_xticklabels([d.strftime("%b") for d in months[::2]])
        ax.set_xlim(R.index[0],R.index[-1]+pd.Timedelta(days=3))
        return fig,ax
    foot="Source: Kalshi daily close. One-factor Gaussian copula; implied ρ makes the 35 race prices reproduce the control price. Democrats need 51 seats."
    x=R.index[-1]; dx=pd.Timedelta(days=6)
    # 1
    fig,a=frame("Democrats have a 70% chance\nof Senate control, but it is a\ndigital bet with big uncertainty",
     "Since mid-September the Democratic chance\nhas entered a new regime: a national mood\nnow moves every seat together.",
     "Chance of Democratic control","Kalshi market vs. race prices at ρ = 0 and ρ = 1")
    a.fill_between(pre.index,np.minimum(pre.q0,pre.q1)*100,np.maximum(pre.q0,pre.q1)*100,color="#e9e8e3",lw=0)
    a.plot(pre.index,pre.q0*100,color=G1,lw=1.8); a.plot(pre.index,pre.q1*100,color=G2,lw=1.8); a.plot(pre.index,pre.market*100,color=G3,lw=2.6)
    a.fill_between(post.index,np.minimum(post.q0,post.q1)*100,np.maximum(post.q0,post.q1)*100,color="#dfe7f1",lw=0)
    a.plot(post.index,post.q0*100,color=ORG,lw=2.2); a.plot(post.index,post.q1*100,color=AQUA,lw=2.2); a.plot(post.index,post.market*100,color=BLUE,lw=3.2)
    for y,c in [(fl.q0,ORG),(fl.q1,AQUA),(fl.market,BLUE)]: a.scatter([x],[y*100],s=60,color=c,edgecolor=SURF,lw=2,zorder=5)
    a.text(x+dx,fl.q0*100,f"{fl.q0*100:.0f}%\nρ = 0",va="center",fontsize=13,linespacing=1.1)
    a.text(x+dx,fl.market*100+3,f"{fl.market*100:.1f}%  Market",va="center",fontsize=13,weight="bold")
    a.text(x+dx,fl.q1*100-5,f"{fl.q1*100:.1f}%\nρ = 1",va="center",fontsize=13,linespacing=1.1)
    a.set_ylim(0,100); a.set_yticks(range(0,101,25)); a.set_yticklabels([f"{v}%" for v in range(0,101,25)])
    a.text(pd.Timestamp("2026-03-08"),25,"BEFORE MID-SEP",fontsize=11,color=MUT,weight="semibold")
    a.text(pd.Timestamp("2026-03-08"),21.5,"Low national mood:\nlittle co-movement\nbetween seats",fontsize=12,color=SEC,va="top")
    a.text(B+pd.Timedelta(days=4),25,"NATIONAL\nMOOD",fontsize=11,color=BLUE,weight="semibold",va="bottom")
    a.text(B+pd.Timedelta(days=4),21.5,"Seats priced\nas moving\ntogether",fontsize=12,color=SEC,va="top")
    fig.text(0.07,0.04,"\n".join(textwrap.wrap(foot,80)),fontsize=10,color=MUT)
    fig.text(0.93,0.04,"1/2",fontsize=12,color=MUT,ha="right")
    fig.savefig(str(FIG / "linkedin_1_control.png"),facecolor=SURF)
    # 2
    fig,b=frame("Same expected seats,\nmore than twice the spread",
     "Correlation does not move the expected\nseat count, but it widens the range around\nit. A wider spread makes the 70% less stable.",
     "Expected Democratic seats","Shaded: ± 1 standard deviation")
    b.plot(pre.index,pre["mean"],color=G3,lw=2.4)
    b.fill_between(post.index,post["mean"]-post.sdimp_s,post["mean"]+post.sdimp_s,color=BLUE,alpha=0.18,lw=0)
    b.fill_between(post.index,post["mean"]-post.sd0,post["mean"]+post.sd0,color=ORG,alpha=0.35,lw=0)
    b.plot(post.index,post["mean"],color=INK,lw=2.8)
    b.axhline(50.5,color=INK,lw=1)
    b.set_ylim(44,60); b.set_yticks(range(44,61,4))
    b.text(x+dx,fl["mean"],f"{fl['mean']:.1f}",va="center",fontsize=13,weight="bold")
    b.text(x+dx,fl["mean"]+fl.sdimp_s,f"{fl['mean']+fl.sdimp_s:.1f}",va="center",fontsize=13)
    b.text(x+dx,fl["mean"]-fl.sdimp_s,f"{fl['mean']-fl.sdimp_s:.1f}",va="center",fontsize=13)
    b.text(pd.Timestamp("2026-05-15"),50.2,"51-seat line",va="top",fontsize=11.5,color=SEC)
    b.legend(handles=[Patch(color=BLUE,alpha=0.18,label="Market-implied ρ (±3.7)"),Patch(color=ORG,alpha=0.35,label="ρ = 0, independent (±1.7)")],
             loc="lower left",bbox_to_anchor=(0.0,0.0),frameon=False,fontsize=12,handlelength=1.2)
    b.text(pd.Timestamp("2026-03-08"),58.6,"BEFORE MID-SEP",fontsize=11,color=MUT,weight="semibold")
    b.text(pd.Timestamp("2026-03-08"),58.0,"Low national mood:\nlittle co-movement between seats",fontsize=12,color=SEC,va="top")
    fig.text(0.07,0.04,"\n".join(textwrap.wrap(foot+" Implied band: 5-day average.",80)),fontsize=10,color=MUT)
    fig.text(0.93,0.04,"2/2",fontsize=12,color=MUT,ha="right")
    fig.savefig(str(FIG / "linkedin_2_seats.png"),facecolor=SURF)

landscape_chart(R.copy())
portrait_charts(R.copy())
print("Saved results/control_series.csv and charts/*.png")
