from numpy import *
import PyXFocus.transformations as tran
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

import pdb
from scipy.optimize import root

def solve_for_g(xou_cat_zspace = 218,r_func = 812.278,f = 13469.3877):
    return f - sqrt(r_func**2 + (f - xou_cat_zspace)**2)

def get_rR(f = 13469.3877,g = 169.8,xi = 1.8*pi/180):
    '''
    Inputs:
    f -- the assumed focal length
    g -- the distance between the principal plane of the optics and the gratings
    xi -- the torus angle (in a matched torus, the blaze angle)
    '''
    r = ((f - g)/2)/cos(xi)
    R = (f - g)*cos(xi) - r
    return r, R

def torus_equation(xp,yp,zp,r,R):
    return (xp**2 + yp**2 + zp**2 + R**2 - r**2)**2 - 4*R**2*(xp**2 + yp**2)

def construct_ct_transform(f,g,xi):
    h = (f - g)*sin(xi)
    xd,zd = -h*cos(xi),h*sin(xi)
    displace = array([xd,0,zd])
    trans_mat = array([[sin(xi),0,cos(xi)],
                    [0,1,0],
                    [-cos(xi),0,sin(xi)]])
    return trans_mat,displace

def solve_for_z(x,y,f = 13469.3877,g = 169.8,xi = 1.8*pi/180,opt_guess = None):
    #r_func = 812.278,f
    #g = solve_for_g(xou_cat_zspace,r_func,f)
    r,R = get_rR(f,g,xi)
    tran_mat,displace = construct_ct_transform(f,g,xi)

    def min_func(k,x,y):
        vec = array([x,y,k])
        xp,yp,zp = dot(tran_mat,vec - displace)    
        return torus_equation(xp,yp,zp,r,R)
    
    if opt_guess is None:
        opt_guess = r + R
    output = root(min_func,opt_guess,args = (x,y),tol = 1e-9)
    return output.x

def solve_for_z_livetrace(x,y,f,xou_cat_zspace,xi,opt_guess = None):
    g = solve_for_g(xou_cat_zspace,y,f)
    r,R = get_rR(f,g,xi)
    #print g,r, R
    tran_mat,displace = construct_ct_transform(f,g,xi)

    def min_func(k,x,y):
        vec = array([x,y,k])
        xp,yp,zp = dot(tran_mat,vec - displace)    
        return torus_equation(xp,yp,zp,r,R)
    
    if opt_guess is None:
        opt_guess = r + R
    output = root(min_func,opt_guess,args = (x,y),tol = 1e-9)
    return output.x

#def compute_tangent_vecs_torus(torus_vec,f = 13469.3877,g = 169.8,xi = 1.8*pi/180):
#    def compute_vecs(r,R,theta,phi):
#        dfdtheta = array([-r*sin(theta)*cos(phi),-r*sin(phi)*sin(theta),r*cos(theta)])
#        dfdphi = array([-(R + r*cos(theta))*sin(phi),(R + r*cos(theta))*cos(phi),0])
#        
#        dfdtheta = dfdtheta/linalg.norm(dfdtheta)
#        dfdphi = dfdphi/linalg.norm(dfdphi)
#
#        dfnorm = cross(dfdphi,dfdtheta)
#        return dfdtheta,dfdphi,dfnorm
#    
#    r,R = get_rR(f,g,xi)
#    tran_mat,displace = construct_ct_transform(f,g,xi)    
#
#    xp,yp,zp = dot(tran_mat,torus_vec - displace)
#    
#    theta,phi = arcsin(zp/r),arctan2(yp,xp)
#
#    trans_disp,trans_gbar,trans_norm = compute_vecs(r,R,theta,phi)
#    
#    disp = dot(linalg.inv(tran_mat),trans_disp)# + displace
#    gbar = dot(linalg.inv(tran_mat),trans_gbar)# + displace
#    norm = dot(linalg.inv(tran_mat),trans_norm)# + displace
#
#    return disp,gbar,norm
    
    

##################################################
# Support functions for placing the vectors and their orientations.

# Next, setting all the gratings to be co-planar on the torus defined by the finite source distance and the grating positions.
# This is accomplished first by (1) orienting the gratings to share a normal with the center of the X-ray beam as defined by
# both XOUs, (2) establishing a grating bar direction perpendicular to the dispersion direction, (3) rotating about the grating
# bar direction to set the blaze angle, and (4) changing the coordinate systems of all the facets.

def make_facet_orientation_vectors(grat_loc, alpha = 1.8*pi/180):
    '''
    Calculates a set of vectors aligned to the Rowland torus at the stated blaze angle alpha
    Inputs:
    grat_loc - the grating position relative to the optics focus.
    alpha - the blaze angle desired (should be matched to the torus calculation)
    Outputs:
    disp - the xhat vector for the grating.
    gbar - the yhat vector for the grating.
    rot_norm - the zhat vector for the grating.s
    '''
    start_norm = grat_loc
    start_norm = start_norm/linalg.norm(start_norm)
    
    gbar = array([0,1.,-start_norm[1]/start_norm[2]])
    gbar = gbar/linalg.norm(gbar)
    
    norm = dot(tran.tr.rotation_matrix(alpha,gbar)[:3,:3],start_norm)
    disp = cross(gbar,norm)
    return disp,gbar,norm

def make_rot_matrix(xhat,yhat,zhat):
    return transpose(vstack((xhat,yhat,zhat)))


#def make_rotation_matrix_to_equate_normals(normgrat,zhat = array([0,0,1])):
#    cvec,ctheta = cross(zhat,normgrat),dot(zhat,normgrat)
#    skew_mat = tran.skew(cvec)
#    I = array([[1,0,0],[0,1,0],[0,0,1]])
#    R = I + skew_mat + dot(skew_mat,skew_mat)*1/(1 + ctheta)
#    return R