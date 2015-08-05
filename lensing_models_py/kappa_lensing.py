import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import fsolve
from scipy import integrate
from math import *
from pylab import *
from mycosmology import *
from alens import *

#--------------------------------------------------
zl = 0.2
zs = 1.0
#--------------------------------------------------
def kappa_sple(x1,x2,a,q,alpha,rc,z1,z2):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr
	sdens = 0.5*(a**(2.0-alpha)/(rc**2.0+x**2.0)**(1.0-alpha/2.0))
	res = sdens/sigma_crit(z1,z2)
	return res
def kappa_pje(x1,x2,a,q,rcut,rc):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr

	sdens = (a/2.0*(1.0/sqrt(rc*rc+x*x)-1.0/sqrt(rcut*rcut+x*x)))
	res = sdens/sigma_crit(z1,z2)
	return res

def kappa_king(x1,x2,a,q,rs):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr

	sdens = (2.12*a/np.sqrt(0.75*rs*rs+x*x)-1.75*a/np.sqrt(2.99*rs*rs+x*x))
	res = sdens/sigma_crit(z1,z2)
	return res

def kappa_dv(x1,x2,a,q,reff,z1,z2):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr

	k = 7.66925001
	sdens = a*np.exp(-k*(x/reff)**(0.25))
	res = sdens/sigma_crit(z1,z2)
	return res

def func_hq(xx):
	x = np.abs(xx)
	x1 = 1.0/np.sqrt(np.abs(1.0-x*x))
	x2 = np.sqrt(np.abs(1.0-x*x))
	s = x*0.0

	idxa = x>0
	idxb = x<1.0
	idx1 = idxa&idxb
	s[idx1] = x1[idx1]*np.arctanh(x2[idx1])	

	idx2 = x == 1.0
	s[idx2]=1.0
	
	idx3 = x >1.0	
	s[idx3] = x1[idx3]*np.arctan(x2[idx3])	

	res = s

	return res

def func_hq_prime(x):
	res = (1.0-x*x*func_hq(x))/(x*(x*x-1.0))
	return res

def kappa_hernquist(x1,x2,rhos,q,rs,z1,z2):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr/rs

	sdens = rhos*rs/(x*x-1.0)**2.0*(-3.0+(2.0+x*x)*func_hq(x))
	res = sdens/sigma_crit(z1,z2)
	return res

def kappa_nfw(x1,x2,rhos,q,rs,z1,z2):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr/rs

	sdens = 2.0*rhos*rs*(1.0-func_hq(x))/(x*x-1.0)
	res = sdens/sigma_crit(z1,z2)

	return res

def kappa_nuker(x1,x2,q,Ib,rb,alpha,beta,gamma,mtl,z1,z2):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr

	sdens = 2.0**(beta-gamma)/alpha*Ib*(x/rb)**(-gamma)*(1.0+(r/rb)**alpha)**((gamma-beta)/alpha)*mtl
	res = sdens/sigma_crit(z1,z2)
	return res


def kappa_expd(x1,x2,q,theta,k0,rd,z1,z2):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr

	sdens = 1.0/np.abs(np.cos(theta))*k0*np.exp(-x/rd)
	res = sdens/sigma_crit(z1,z2)
	return res

def kappa_kuzmind(x1,x2,q,theta,k0,rs,z1,z2):
	r = np.sqrt(x1*x1+x2*x2/q**2.0)
	x = r*Da(z1)/apr

	sdens = 1.0/np.abs(np.cos(theta))*k0*rs**3.0*(rs*rs+x*x)**(-1.5)
	res = sdens/sigma_crit(z1,z2)
	return res
#--------------------------------------------------
def func_fg(z,x,alpha):
	x1 = 1.0/(np.sqrt(x*x+z*z)**(alpha))
	x2 = 1.0/(1.0+np.sqrt(x*x+z*z))**(3.0-alpha)
	res = x1*x2
	return res

def fg(x,alpha):
	res = integrate.quad(func_fg,0.00001,np.Inf,args=(x,alpha))[0]
	#z = np.linspace(0,1000,10000)
	#res = np.sum(func_fg(z,x,alpha))*(z[1]-z[0])
	return res
def func_fl(x,alpha):
	res = fg(x,alpha)*x
	return res

def fl(x,alpha):
	res = integrate.quad(func_fl,0.00001,x,args=(alpha,))[0]
	return res

def rhos_gnfw(c,alpha,z):
	res = dv(z)*rho_crit(z)*c/(3.0*fl(c,beta))
	return res
def mus_gnfw(c,m,alpha,z1,z2):
	rs = r200_m200(m,z1)/c
	res = 4.0*rhos_gnfw(c,alpha,z1)*rs/sigma_crit(z1,z2)
	return res;
#----------------------------------------------------------------------------

def lens_equation_gnfw(x,mus,alpha):
	res = x-mus*fl(x,alpha)/x
	return res
def x_0(mus,alpha):
	res = brentq(lens_equation_gnfw,1e-2,100.0,args=(mus,alpha))
	return res
def dy_dx_gnfw(x,mus,alpha):
	dx = 1e-3*x
	xplus = x+dx
	xminus = x-dx
	dy = lens_equation_gnfw(xplus,mus,alpha)-lens_equation_gnfw(xminus,mus,alpha)

	res = dy/(2.0*dx)
	return res
def x_cr(mus,alpha):
	xlow = 1e-3*x_0(mus,alpha)
	xup = x_0(mus,alpha)
	res = brentq(dy_dx_gnfw,xlow,xup,args=(mus,alpha))
	return res
def y_cr(mus,alpha):
	res = lens_equation_gnfw(x_cr(mus,alpha),mus,alpha)
	return res
def mu_gnfw(x,mus,alpha):
	res = dy_dx_gnfw(x,mus,alpha)*lens_equation_gnfw(x,mus,alpha)/x
	return 1.0/res
#-----------------------------------------------------------------------------------
def gnfw_kr(x,c,m,z1,z2,alpha):
	r200 = r200_m200(m,z1)
	rs = r200/c
	xx = x*Da(z1)/apr/rs
	rhos = rho_crit(z1)*dv(z1)/3.0*c**3.0/(np.log(1.0+c)-c/(1+c))
	kappas = rs*rhos/sigma_crit(z1,z2)
	res = 2.0*kappas*fg(xx,alpha)
	return res
def func_bar(x,c,m,z1,z2,alpha):
	res = 2.0*np.pi*x*gnfw_kr(x,c,m,z1,z2,alpha)
	return res
def gnfw_kr_bar(x,c,m,z1,z2,alpha):
	up = integrate.quad(func_bar,0.0,x,args=(c,m,z1,z2,alpha))[0]
	down = np.pi*x*x
	res = up/down
	return res
def gnfw_shr(rr,c,m,z1,z2,alpha):
	res = gnfw_kr_bar(rr,c,m,z1,z2,alpha)-gnfw_kr(rr,c,m,z1,z2,alpha)
	return res
def gnfw_sh(x,y,c,m,z1,z2,alpha):
	r = np.sqrt(x*x+y*y)
	cosphi = x/r
	sinphi = y/r
	sin2phi = 2.0*sinphi*cosphi
	cos2phi = cosphi**2.0-sinphi**2.0

	gsh = gnfw_shr(r,c,m,z1,z2,alpha)
	res1 = gsh*cos2phi
	res2 = gsh*sin2phi
	return res1,res2
#--------------------------------------------------
def gnfw_fr(rr,c,m,z1,z2,alpha):
	dd = 1e-5
	dr = dd*rr
	r_p = (1.0+dd)*rr
	r_m = (1.0-dd)*rr


	gnk_rp = gnfw_kr(r_p,c,m,z1,z2,alpha)
	gnk_rm = gnfw_kr(r_m,c,m,z1,z2,alpha)
	
	ffr = (gnk_rp-gnk_rm)/(2.0*dr)
	#ffr = np.abs(ffx**2.0+ffy**2.0)
	
	return np.abs(ffr)
def gnfw_ff(x,y,c,m,z1,z2,alpha):
	r = np.sqrt(x*x+y*y)
	ff = gnfw_fr(r,c,m,z1,z2,alpha)
	cosphi = x/r
	sinphi = y/r
	res1 = ff*cosphi
	res2 = ff*sinphi
	return res1,res2
def gnfw_gr(rr,c,m,z1,z2,alpha):
	dd = 1e-3
	dr = dd*rr
	r_p = rr+dr
	r_m = rr-dr

	gnsh_rr = gnfw_shr(rr,c,m,z1,z2,alpha)
	gnsh_rp = gnfw_shr(r_p,c,m,z1,z2,alpha)
	gnsh_rm = gnfw_shr(r_m,c,m,z1,z2,alpha)
	
	gfr_r = (gnsh_rp-gnsh_rm)/(2.0*dr)
	gfr_phi = -2.0*gnsh_rr/rr
	gfr = gfr_r+gfr_phi
	
	return gfr
def gnfw_gf(x,y,c,m,z1,z2,alpha):
	r = np.sqrt(x**2.0+y**2.0)
	cosphi = x/r
	sinphi = y/r
	cos3phi = cosphi**3.0-3.0*cosphi*sinphi**2.0
	sin3phi = 3.0*cosphi**2.0*sinphi-sinphi**3.0

	gf = gnfw_gr(r,c,m,z1,z2,alpha)
	res1 = gf*cos3phi
	res2 = gf*sin3phi
	return res1,res2
#--------------------------------------------------
def gnfw_reduce_fr(r,c,m,z1,z2,alpha):
	if (gnfw_kr(r,c,m,z1,z2,alpha) >= 0.5):
		reduced = 0.5
	else:
		reduced = 1.0-gnfw_kr(r,c,m,z1,z2,alpha)

	res = gnfw_fr(r,c,m,z1,z2,alpha)/reduced
	return res
#--------------------------------------------------
def gnfw_reduce_gr(r,c,m,z1,z2,alpha):
	if (gnfw_kr(r,c,m,z1,z2,alpha) >= 0.5):
		reduced = 0.5
	else:
		reduced = 1.0-gnfw_kr(r,c,m,z1,z2,alpha)

	res = gnfw_gr(r,c,m,z1,z2,alpha)/reduced
	return res
#--------------------------------------------------
X = np.linspace(0.00001,1.0,1000)
Y1 = np.linspace(0.0001,1.0,1000)
Y2 = np.linspace(0.0001,1.0,1000)
Y3 = np.linspace(0.0001,1.0,1000)

for i in range(50):
	#Y2[i] = nfw_gf(X[i],0.1,12.0,1e12,0.2,1.0)[1]
	#Y3[i] = gnfw_gf(X[i],0.1,12.0,1e12,0.2,1.0,1.0)[1]
	#Y1[i] = lens_equation_gnfw(X[i],1.0,1.1)
	#Y2[i] = lens_equation_gnfw(X[i],1.0,1.3)
	#Y3[i] = lens_equation_gnfw(X[i],1.0,1.5)
	Y1[i] = mu_gnfw(X[i],1.0,1.1)
	Y2[i] = mu_gnfw(X[i],1.0,1.3)
	Y3[i] = mu_gnfw(X[i],1.0,1.5)
#--------------------------------------------------
#print lens_equation_gnfw(1e-2,1.0,1.1)
#print lens_equation_gnfw(100,1.0,1.1)
#print x_0(1.0,1.1),x_0(1.0,1.3),x_0(1.0,1.5)
#print x_0(1.0,1.1),x_0(1.0,1.3),x_0(1.0,1.5)
#print x_cr(1.0,1.1),x_cr(1.0,1.3),x_cr(1.0,1.5)
#print y_cr(1.0,1.1),y_cr(1.0,1.3),y_cr(1.0,1.5)
def plot_data():
	#plt.xlabel(r'$\theta_{e}$')
	#plt.ylabel(r'N(>$\theta_{e}$)')
	#plt.title('Cumulative Einstein radius Distribution From 1000 MS Halo')
	#plot(tmp, np.log10(1000-X))

	#xlim([-0.5, 0.5])
	#ylim([-0.5, 0.5])
	plot(X, Y1,'r-',X, Y2, 'g-', X, Y3, 'b-')
	#plot(X, Y2)
	#plot(X, Y3,'ro')
	#hist(theta_e)
	savefig('gnfw_mu.pdf')



plot_data()
#plt.show()
