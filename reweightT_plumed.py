# -------------------------------------------------------------------------------------------------
# This class is used to compute the reweighting array for metadynamics results from PLUMED.
# Example:
# $ python3 reweightT_plumed.py --colvar COLVAR --hills HILLS --stride 20
# -------------------------------------------------------------------------------------------------


import sys
import time
import numpy as np

arglist=sys.argv
for i in range(len(arglist)):
    if arglist[i]=='--stride':
        stride=int(arglist[i+1])
    if arglist[i]=='--colvar':
        COLVAR_dir=arglist[i+1]
    if arglist[i]=='--hills':
        HILLS_dir=arglist[i+1]


T=300.0
print('Temperature T =', T, 'K')

if 'stride' in locals() or 'stride' in globals():
    print('stride =', stride)
else:
    stride=1
    print('Not setting variable "stride" from shell. Using default stride =', stride)

if 'COLVAR_dir' in locals() or 'COLVAR_dir' in globals():
    print('COLVAR_dir: ', COLVAR_dir)
else:
    COLVAR_dir='./COLVAR'
    print('Not setting COLVAR_dir from shell. Using default COLVAR_dir:', COLVAR_dir)

if 'HILLS_dir' in locals() or 'HILLS_dir' in globals():
    print('HILLS_dir: ', HILLS_dir)
else:
    HILLS_dir='./HILLS'
    print('Not setting HILLS_dir from shell. Using default HILLS_dir:', HILLS_dir)


col_colvar=1; col_bias=10 # The column number of collective variable and bias in COLVAR file.
col_rc=1; col_sigma=2; col_height=3; col_biasf=4 # The column of rc, sigma, height, biasf in HILLS file.


def FES(x,n_bias,sigma,height,biasf):
    """
    Calculate the free energy surface.
    Remember that the heights readed from HILLS are already multiplied by (biasf/(biasf-1)).
    """
    return -np.sum(height*np.exp(-0.5*(x-n_bias)**2/sigma**2))

def set_grid(colvar, sigma):
    """
    Grid along colvar for integra
    tion.
    """
    colvar_max=np.max(colvar)+5*sigma
    colvar_min=np.min(colvar)-5*sigma
    bin_num=int( (colvar_max-colvar_min)/sigma )
    grid_pts=np.linspace(colvar_min, colvar_max, bin_num)
    
    return grid_pts


COLVAR=np.loadtxt(COLVAR_dir, usecols=(col_colvar, col_bias), unpack=True)
colvar, bias=COLVAR

HILLS=np.loadtxt(HILLS_dir, usecols=(col_rc, col_sigma, col_height, col_biasf), unpack=True)
rc_bias, sigma, height, biasf=HILLS
print('... finish loading COLVAR and HILLS files.')


start=time.time()

beta=1000/(T*8.28) # kT=(8.28/1000)*T (kJ/mol/K)
sigma_0=sigma[0]
biasf_0=biasf[0]
rc_gridpts=set_grid(rc_bias, sigma_0)

ebetac=[]
weights=[]

for i in range(len(rc_bias)):
    if i%stride==0:
        rc_bias_i=rc_bias[:i+1]
        height_i=height[:i+1]

        exp_arr=np.array([FES(s,rc_bias_i,sigma_0,height_i,biasf_0) for s in rc_gridpts])
        exp_arr=-beta*exp_arr

        num_arr=np.exp(exp_arr)
        den_arr=np.exp(exp_arr/biasf_0)

        num=np.sum(num_arr)
        den=np.sum(den_arr)

        ebetac.append(num/den)
        print('{} ebetac[{}]={}'.format(i, i, num/den))

print('... finish calculating ebetac.\n')

spacing=len(colvar)/len(ebetac)

for i in range(len(colvar)):
    sc=np.exp(beta*bias[i])
    j=int(i/spacing)
    weights.append(sc/ebetac[j])

print('... finish calculating weights.')
end=time.time()

print("Program takes ", end-start, " s")

np.savetxt('weights.txt', weights)

