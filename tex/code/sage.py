import numpy as np
import matplotlib.pyplot as plt
import sys, os, \
    glob, cPickle, \
    argparse, errno, json,\
    copy, re,time,datetime
import numpy as np
import os.path as osp
import scipy.io as sio
from pprint import pprint
def lldistkm(latlon1,latlon2):
    '''
     Distance:
     d1km: distance in km based on Haversine formula
     (Haversine: http://en.wikipedia.org/wiki/Haversine_formula)
    '''
        
    radius=6371
    lat1=latlon1[0]*pi/180
    lat2=latlon2[0]*pi/180
    lon1=latlon1[1]*pi/180
    lon2=latlon2[1]*pi/180
    deltaLat=lat2-lat1
    deltaLon=lon2-lon1
    a=sin((deltaLat)/2)^2 + cos(lat1)*cos(lat2) * sin(deltaLon/2)^2
    c=2*atan2(sqrt(a),sqrt(1-a))
    d1km=radius*c    #Haversine distance
    return d1km
def lldistkm_o(l1,l2):
    l1=vector(l1)
    l2=vector(l2)
    return norm(l2-l1)
root='/home/xlwang/l-opt'
os.chdir(root)
print os.getcwd()
with open('location.pkl','r') as f:
    locations =cPickle.load(f)

for m_limits  in range(3,23):
    old_t = time.time()

    location=locations[:m_limits]    
 
    latlons=[ [latlon[1]['lat'],latlon[1]['lng']] for latlon in location ]
    names=[latlon[0].split(',')[0] for latlon in location]
    n_pts=len(latlons)
    tab=table( \
    rows=[(latlons[i][0],latlons[i][1]) for i in range(n_pts)],\
    header_column=names,\
    frame=True\
    )
    str=latex(tab)
    with open("city_data.tex",'w') as f:
        f.write(str)    
    dist=np.zeros((n_pts,n_pts))
    for i in range(n_pts):
        for j in range(n_pts):
            dist[i,j]=N(lldistkm(latlons[i],latlons[j]),32)
    dist  
    
    # For small  problem:
    #dist=Matrix([[0,1,2],[2,0,1],[1,2,0]])
    dist=Matrix(dist)
    p=MixedIntegerLinearProgram(maximization=False,solver="GLPK") #,solver="PPL"
    x=p.new_variable(nonnegative=True,integer=True)
    t=p.new_variable(nonnegative=True,integer=True)
    n=dist.nrows()
    obj_func=0
    for i in range(n):
        for j in range(n):
            obj_func+=x[i,j]*dist[i,j] if i!=j else 0
    p.set_objective(obj_func)
    for i in range(n):
        p.add_constraint(sum([x[i,j] for j in range(n) if i!=j ])==1)
    for j in range(n):
        p.add_constraint(sum([x[i,j] for i in range(n) if i!=j ])==1)
    for i in range(n):
        for j in range(1,n):
            if i==j :
                continue
            p.add_constraint(t[j]>=t[i]+1-n*(1-x[i,j]))
    for i in range(n):
        p.add_constraint(t[i]<=n-1)
                
    #p.show()
    p.solve()  
    elaspe=time.time() - old_t
    print m_limits, elaspe