import numpy as np
from reference_model import run
print("Deffuant check, complete network, p=0, sigma=0, T=200, 5 seeds: clusters per seed")
for e in (0.1,0.2,0.3):
    for two in (True,False):
        for sr,lab in (((0.2,1.0),"s~U(0.2,1) (current)"),((1.0,1.0),"s=1 so s*tau=0.5=mu")):
            c=[run(p=0,sigma=0,e=e,complete=True,two_way=two,seed=s,s_range=sr,T=200)["clusters"] for s in range(5)]
            print(e,"two-way" if two else "receiver-only",lab,c,"mean",np.mean(c))
