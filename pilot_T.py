import numpy as np
from reference_model import run, count_clusters
import time
t0=time.time()
print("=== A) Is T=200 enough? small-world, p=0.5, tau_c0=0.5, b=0, 3 seeds, T=1000 ===")
print("e | tick | variance | clusters | mean change/tick (last tick) ")
for e in (0.1,0.2,0.3):
    res=[run(p=0.5,e=e,T=1000,seed=s) for s in range(3)]
    for t in (50,100,200,400,600,1000):
        var=np.mean([r["checkpoints"].get(t,{}).get("var",np.nan) if t in r["checkpoints"] else np.nan for r in res]) if t%50==0 else np.nan
        cl=[r["checkpoints"][t]["clusters"] for r in res]
        mc=np.mean([r["mean_change"][t-20:t].mean() for r in res])
        print(e,t,round(var,4),cl,f"{mc:.5f}")
print("time",round(time.time()-t0))
