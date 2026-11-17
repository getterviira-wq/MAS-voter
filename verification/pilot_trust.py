import numpy as np
from reference_model import run
print("=== B) Mean computer trust, p=1, b=0 / b=0.15, 3 seeds ===")
print("b  e  tau_c0 | t=1  t=5  t=20  t=200 | share of voters with trust 0 at t=200")
for b in (0,0.15):
  for e in (0.1,0.2,0.3):
    for t0 in (0.25,0.5,0.75):
        rs=[run(p=1,e=e,tau_c0=t0,b=b,T=200,seed=s) for s in range(3)]
        mt=np.mean([r["mean_trust"] for r in rs],axis=0)
        z=np.mean([(r["trust"]==0).mean() for r in rs])
        print(b,e,t0,"|",*(f"{mt[i]:.2f}" for i in (0,4,19,199)),"|",f"{z:.2f}")
