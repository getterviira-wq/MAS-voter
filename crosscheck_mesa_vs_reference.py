import numpy as np, importlib.util
from dataclasses import replace
from reference_model import run
spec=importlib.util.spec_from_file_location("m1","project-assignment-group-20.py"); m1=importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)
print("Cross-check reference vs Mesa model (sigma=0 in reference, 10 seeds, T=200)")
print("setting | quantity | reference | mesa")
for lab,kw in {"p=0.5 e=0.2 b=0":dict(p=0.5,e=0.2,b=0),"p=1 e=0.3 b=0.15":dict(p=1,e=0.3,b=0.15),"p=1 e=0.1 b=0":dict(p=1,e=0.1,b=0)}.items():
    R=[run(sigma=0,T=200,seed=s,**kw) for s in range(10)]
    M=[]
    for s in range(10):
        P=replace(m1.PARAMS,seed=s,**kw); mod=m1.VoterModel(P); m1.run(mod,200); M.append(mod)
    rv=np.mean([r["var"] for r in R]); mv=np.mean([m.opinions().var() for m in M])
    rt=np.mean([r["trust"].mean() for r in R]); mt=np.mean([m.computer_trusts().mean() for m in M])
    rm=np.mean([np.median(r["x"]) for r in R]); mm=np.mean([np.median(m.opinions()) for m in M])
    print(lab,"| variance",round(rv,4),round(mv,4),"| trust",round(rt,3),round(mt,3),"| median",round(rm,3),round(mm,3))
