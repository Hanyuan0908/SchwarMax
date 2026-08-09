"""Build the cache consumed by test_acceptance_ngc4371.py.

Coarsens the published 1442-bin NGC 4371 data to 581 bins and builds the exact-MGE
potential, using the milestone snapshot's own code so the inputs are reproduced
exactly. Writes ngc4371_milestone_acceptance.pkl next to the test.

The milestone snapshot is not redistributed here; set MILESTONE_DIR to your copy.
The two code trees cannot share a process (both define model.py / integrants.py),
which is why this is a separate prep step rather than part of the test.
"""
import sys,os,pickle,numpy as np,warnings; warnings.filterwarnings('ignore')
P=os.environ.get('MILESTONE_DIR')
if not P or not os.path.isdir(P):
    raise SystemExit(
        "Set MILESTONE_DIR to your copy of the NGC 4371 best-fit milestone snapshot\n"
        "(the directory containing run_bestfit.py, data/ and schwarmax_code/).\n"
        "The snapshot is not redistributed with this package.")
sys.path.insert(0,os.path.join(P,'schwarmax_code'))
import jax, jax.numpy as jnp; jax.config.update('jax_platform_name','cpu'); jax.config.update('jax_enable_x64',True)
from scipy.cluster.vq import kmeans2
from get_best_orbital_library_fixed_potential import get_dict_data_bootstrap
from integrants import assign_regular_grid
from scipy.stats import qmc
import mge_model as MM
DATA=os.path.join(P,'data'); OUT=os.path.join(P,'outputs'); NBIN=581
def rebuild_grid(dd,Rmax=4.,zmax=2.,nR=10,nz=6,nphi=6):
    Re=np.linspace(0,Rmax,nR+1);ze=np.linspace(-zmax,zmax,nz+1);pe=np.linspace(-np.pi,np.pi,nphi+1)
    Rm=.5*(Re[:-1]+Re[1:]);zm=.5*(ze[:-1]+ze[1:]);pm=.5*(pe[:-1]+pe[1:])
    RR,ZZ,PP=np.meshgrid(Rm,zm,pm,indexing='ij');Rg,zg,pg=RR.ravel(),ZZ.ravel(),PP.ravel()
    mid=jnp.stack([jnp.array(Rg),jnp.array(zg),jnp.array(pg)],axis=-1)
    idx=assign_regular_grid(mid,grid_min=jnp.array([0.,-zmax,-jnp.pi]),grid_max=jnp.array([Rmax,zmax,jnp.pi]),
                            n_bins=jnp.array([nR,nz,nphi]),strides=jnp.array([1,nR,nR*nz]))
    o=np.argsort(np.asarray(idx))
    dd['R_grid']=Rg[o];dd['z_grid']=zg[o];dd['phi_grid']=pg[o]
    dd['dR']=float(Rm[1]-Rm[0]);dd['dz']=float(zm[1]-zm[0]);dd['dphi']=float(pm[1]-pm[0])
    dd['R_minmax']=[0.,Rmax];dd['z_minmax']=[-zmax,zmax];dd['phi_minmax']=[-np.pi,np.pi]
    dd['Rzphi_n_tot']=nR*nz*nphi;dd['Rzphi_n_grid']=np.array([nR,nz,nphi])
    dd['sample_for_integration']=np.asarray(qmc.Sobol(d=3,scramble=False).random_base2(m=10))
    return dd
src=dict(pickle.load(open(os.path.join(DATA,'schwarmax_ngc4371_mnras_v3_discrete_rebin1442_v1.pkl'),'rb')))
xy=np.asarray(src['voronoi_bins_xy']);npb=np.asarray(src['num_per_bin']).astype(float)
sd=np.asarray(src['surface_density']).astype(float);lum=sd*npb
cent,lab=kmeans2(xy,NBIN,minit='++',seed=0);lab=np.asarray(lab);NB=int(lab.max()+1)
def wavg(v):
    v=np.asarray(v).astype(float);out=np.zeros(NB)
    for c in range(NB):
        m=lab==c;w=lum[m];out[c]=np.sum(v[m]*w)/np.sum(w) if np.sum(w)>0 else 0.
    return out
new_npb=np.array([npb[lab==c].sum() for c in range(NB)]);new_lum=np.array([lum[lab==c].sum() for c in range(NB)])
new_xy=np.array([[np.sum(xy[lab==c,0]*lum[lab==c])/np.sum(lum[lab==c]),
                  np.sum(xy[lab==c,1]*lum[lab==c])/np.sum(lum[lab==c])] for c in range(NB)])
news=wavg(src['s']); coarse=dict(src)
coarse.update(total_bins=np.int32(NB),voronoi_bins_xy=new_xy.astype(np.float32),
  num_per_bin=new_npb.astype(np.int64),surface_density=(new_lum/np.maximum(new_npb,1)).astype(np.float32),
  V_mean=wavg(src['V_mean']).astype(np.float32),V_sigma=wavg(src['V_sigma']).astype(np.float32),
  v0=wavg(src['v0']).astype(np.float32),s=news.astype(np.float32),
  h1=np.zeros(NB,np.float32),h2=np.zeros(NB,np.float32),
  h3=wavg(src['h3']).astype(np.float32),h4=wavg(src['h4']).astype(np.float32))
bm=np.asarray(src['bin_mapping']).copy();good=bm>=0;newbm=bm.copy();newbm[good]=lab[bm[good]]
coarse['bin_mapping']=newbm
for k,val in [('V_mean_err',7.),('V_sigma_err',7.),('h3_err',0.03),('h4_err',0.03)]: coarse[k]=np.full(NB,val,np.float32)
coarse['h1_err']=(7./(news*np.sqrt(2))).astype(np.float32);coarse['h2_err']=(7./(news*np.sqrt(2))).astype(np.float32)
TMP=os.path.join(OUT,'_coarse581_bindict.pkl');pickle.dump(coarse,open(TMP,'wb'))
dd=rebuild_grid(get_dict_data_bootstrap('',TMP))
dd['alpha']=0.0;dd['beta']=59.0;dd['gamma']=78.0
dd['XY_density_data_err']=0.05*np.asarray(dd['XY_density_data'])
dd['relax_3d']=2.5;dd['relax_2d']=1.0;dd['n_realizations']=4
mge=src['meta']['intrinsic_mge']
sx=np.asarray(mge['sigma_x_kpc']);sy=np.asarray(mge['sigma_y_kpc']);sz=np.asarray(mge['sigma_z_kpc']);mass=np.asarray(mge['mass_msun'])
S=jnp.array(sx);Pg=jnp.array(sy/sx);Qg=jnp.array(sz/sx);Mg=jnp.array(mass)
dphi=MM.get_phi_m_exact(Mg,S,Pg,Qg,50,30,1e-3,50.0,1e-3,20.0,6,32)
dphi={k:np.asarray(v) for k,v in dphi.items()}
dphi['mge_S']=np.asarray(S);dphi['mge_P']=np.asarray(Pg);dphi['mge_Q']=np.asarray(Qg)
dphi['mge_rho0']=np.asarray(mass/((2*np.pi)**1.5*sx*sy*sz))
halo=src['meta']['halo']
rng=np.random.default_rng(0);pts=[]
for j in range(len(sx)):
    nj=max(int(200000*mass[j]/mass.sum()),20);pts.append(rng.multivariate_normal([0,0,0],np.diag([sx[j]**2,sy[j]**2,sz[j]**2]),nj))
Pp=np.concatenate(pts);Rr=np.hypot(Pp[:,0],Pp[:,1]);Pp=Pp[(Rr<6.)&(np.abs(Pp[:,2])<5.)&(Rr>0.02)]
dd['w0']=Pp[rng.choice(len(Pp),10000,replace=False)]
dd['nX_nY']=coarse['nX_nY']
out={k:(np.asarray(v) if hasattr(v,'shape') else v) for k,v in dd.items()}
D=os.path.dirname(os.path.abspath(__file__))
pickle.dump({'dd':out,'dphi':dphi,'halo':{'logM':float(halo['log10_nfw_scale_mass']),'Rs':float(halo['rs_kpc'])},
             'Omega':20.0/0.97779}, open(os.path.join(D,'ngc4371_milestone_acceptance.pkl'),'wb'))
print("cached: %d bins, angles=(%.0f,%.0f,%.0f), w0=%s, NFW logM=%.4f Rs=%.1f"%(
  NB,dd['alpha'],dd['beta'],dd['gamma'],np.shape(dd['w0']),float(halo['log10_nfw_scale_mass']),float(halo['rs_kpc'])))
print("MPREP OK")
