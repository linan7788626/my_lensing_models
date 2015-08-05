import numpy as np
from mycosmology import *
import alens_arr as aa
#-------------------------------------------------------------

def pw2(x):
	res = x*x
	return res
def re_point(m,z1,z2):
	res = np.sqrt(4.0*G*m/c**2.0*Da(z1,z2)/(Da(z1)*Da(z2)))
	return res
def lens_equation_point(x1,x2,re):
	xq = np.sqrt(x1*x1+x2*x2)
	req = re**2.0
	y1 = x1*(1.0-req/xq**2.0)
	y2 = x2*(1.0-req/xq**2.0)
	mu = (1.0-(re/xq)**4.0)**(-1.0)
	return y1,y2,mu
#-------------------------------------------------------------
def lens_equation_nfw(x,mus):
	#mus = 4.0*ks
	res = x-mus*x*aa.func_bar(x)
	return res

#-------------------------------------------------------------
def lens_equation_sie(x1,x2,lpar):
	xc1 = lpar[0]
	xc2 = lpar[1]
	q   = lpar[2]
	rc  = lpar[3]
	re  = lpar[4]
	pha = lpar[5]

	phirad = np.deg2rad(pha)
	cosa = np.cos(phirad)
	sina = np.sin(phirad)

	xt1 = (x1-xc1)*cosa+(x2-xc2)*sina
	xt2 = (x2-xc2)*cosa-(x1-xc1)*sina

	phi = np.sqrt(pw2(xt2)+pw2(xt1*q)+pw2(rc))
	sq = np.sqrt(1.0-q*q)
	pd1 = phi+rc/q
	pd2 = phi+rc*q
	fx1 = sq*xt1/pd1
	fx2 = sq*xt2/pd2
	qs = np.sqrt(q)

	a1 = qs/sq*np.arctan(fx1)
	a2 = qs/sq*np.arctanh(fx2)
#-----------------------------------------------------------------------
	xt11 = cosa
	xt22 = cosa
	xt12 = sina
	xt21 =-sina

	fx11 = xt11/pd1-xt1*(xt1*q*q*xt11+xt2*xt21)/(phi*pw2(pd1))
	fx22 = xt22/pd2-xt2*(xt1*q*q*xt12+xt2*xt22)/(phi*pw2(pd2))
	fx12 = xt12/pd1-xt1*(xt1*q*q*xt12+xt2*xt22)/(phi*pw2(pd1))
	fx21 = xt21/pd2-xt2*(xt1*q*q*xt11+xt2*xt21)/(phi*pw2(pd2))

	a11 = qs/(1.0+fx1*fx1)*fx11
	a22 = qs/(1.0-fx2*fx2)*fx22
	a12 = qs/(1.0+fx1*fx1)*fx12
	a21 = qs/(1.0-fx2*fx2)*fx21

	rea11 = (a11*cosa-a21*sina)*re
	rea22 = (a22*cosa+a12*sina)*re
	rea12 = (a12*cosa-a22*sina)*re
	rea21 = (a21*cosa+a11*sina)*re

	y11 = 1.0-rea11
	y22 = 1.0-rea22
	y12 = 0.0-rea12
	y21 = 0.0-rea21

	jacobian = y11*y22-y12*y21
	mu = 1.0/jacobian

	res1 = (a1*cosa-a2*sina)*re
	res2 = (a2*cosa+a1*sina)*re
	return res1,res2,mu
#-------------------------------------------------------------
def xy_rotate(x, y, xcen, ycen, phi):
	phirad = np.deg2rad(phi)
	xnew = (x - xcen) * np.cos(phirad) + (y - ycen) * np.sin(phirad)
	ynew = (y - ycen) * np.cos(phirad) - (x - xcen) * np.sin(phirad)
	return (xnew,ynew)

def gauss_2d(x,y,n,amp,sig,xc,yc,e,phi):
	(xnew,ynew) = xy_rotate(x, y, xc, yc, phi)
	r_ell_sq = ((xnew)*e + (ynew)/e)/np.abs(sig)**2
	res = amp * np.exp(-0.5*r_ell_sq)
	return res

def gauss_2dnn(x,y,n,amp,sig,xc,yc,e,phi):
	(xnew,ynew) = xy_rotate(x, y, xc, yc, phi)
	r_ell_sq = ((xnew**n)*e + (ynew**n)/e)/np.abs(sig)**2
	res = amp * np.exp(-0.5*r_ell_sq)
	return res
