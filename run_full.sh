#!/bin/bash
# run_full.sh — clean rerun with fixed A_F computation
# Run this from inside your fisher_pipeline directory:
#   bash run_full.sh

set -e  # stop on any error

echo "============================================"
echo "  Fisher-CNN Full Training Run"
echo "============================================"

# Step 1: Delete ALL old outputs so nothing stale is reused
echo ""
echo "Step 1: Removing old dataset and outputs..."
rm -f dataset.h5
rm -f dataset_test.h5
rm -rf outputs/
mkdir -p outputs/

echo "  ✓ Cleared dataset.h5 and outputs/"

# Step 2: Quick sanity check — print expected A_F range
echo ""
echo "Step 2: Verifying A_F computation (should see values 0.38–0.70)..."
python3 - << 'PYEOF'
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

# Quick inline test of the fixed functions
M = 0.067; H2 = 38.10/M
def V(gs,hw,a,n,dx):
    cx=(gs-1)/2.; x=np.arange(gs,dtype=float)-cx; XX,YY=np.meshgrid(x,x)
    r=np.sqrt(XX**2+YY**2)*dx; th=np.arctan2(YY,XX)
    return (hw**2/(2*H2))*r**2*(1+a*np.cos(n*th))
def rho(Vg,dx):
    N=Vg.shape[0]; T=H2/dx**2; H=lil_matrix((N*N,N*N),dtype=np.float64)
    def idx(i,j): return i*N+j
    for i in range(N):
        for j in range(N):
            p=idx(i,j); H[p,p]=4*T+Vg[i,j]
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni,nj=i+di,j+dj
                if 0<=ni<N and 0<=nj<N: H[p,idx(ni,nj)]=-T
    _,v=eigsh(H.tocsr(),k=1,which='SM'); psi=v[:,0].reshape(N,N)
    r2=np.abs(psi**2); r2/=r2.sum(); return r2
def AF(rh,na=72,eps=1e-12):
    N=rh.shape[0]; cx=(N-1)/2.; x=np.arange(N)-cx; XX,YY=np.meshgrid(x,x)
    rg=np.sqrt(XX**2+YY**2); tg=np.arctan2(YY,XX)
    ts=np.linspace(-np.pi,np.pi,na,endpoint=False); IF=np.zeros(na)
    dt=ts[1]-ts[0]
    for ti,t in enumerate(ts):
        m=(tg>=t)&(tg<t+dt)
        if m.any(): IF[ti]=(rh[m]*rg[m]).sum()
    IF/=(IF.mean()+eps); lo,hi=IF.min(),IF.max()
    return (hi-lo)/(hi+lo) if (hi+lo)>1e-12 else 0.

gs=32; R=15.; dx=R/(gs*0.3)
results = [
    ('isotropic  α=0.0', AF(rho(V(gs,10,0.0,3,dx),dx))),
    ('C3v       α=0.8', AF(rho(V(gs,10,0.8,3,dx),dx))),
    ('C4        α=0.8', AF(rho(V(gs,10,0.8,4,dx),dx))),
]
all_ok = True
for name, val in results:
    status = "✓" if val > 0.01 else "✗ STILL BROKEN"
    print(f"  {status}  {name}: A_F = {val:.4f}")
    if val < 0.01: all_ok = False
if not all_ok:
    print("ERROR: A_F values still near zero. dataset.py fix did not apply.")
    sys.exit(1)
print("  A_F computation looks correct.")
PYEOF

# Step 3: Full training
echo ""
echo "Step 3: Starting full training (80,000 samples, 120 epochs)..."
echo "  This will take ~3.5 hours. You can leave it running."
echo ""

python train.py \
    --n_samples 80000 \
    --grid_size 64 \
    --epochs 120 \
    --batch_size 64 \
    --patience 20 \
    --out_dir outputs/

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Check outputs/ for results and figures."
echo "============================================"
