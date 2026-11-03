"""Exact external cuboid B/J tensor, evaluating geometric terms only once.

The expressions and safe-octant transformation are adapted from Magpylib
5.0.1 ``magnet_cuboid_Bfield`` (Yang / Engel-Herbert / Camacho / Cichon).
No point-dipole, far-field truncation or material approximation is used here.
Only external source-target interactions call this kernel. Surface/edge or
nonfinite rows are marked NaN for the caller's public-Magpylib fallback.

Copyright (c) 2022, Silicon Austria Labs, Magpylib Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np


def external_cube_tensor(observers, edge):
    """Return (n,3,3) exact B/J tensor; unsafe/non-external rows are NaN.

    Length units are arbitrary but must match. Tensor row is field component,
    column is source-polarization component. The external Hessian is symmetric
    and trace free, so two diagonal terms determine the third exactly.
    """
    points=np.asarray(observers,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(edge) or edge<=0:
        raise ValueError('Finite Nx3 observers and positive cube edge required')
    half=edge/2.
    external=np.all(np.isfinite(points),axis=1)&np.any(np.abs(points)>half*(1.+1e-15),axis=1)
    x,y,z=points.T.copy()
    sx=np.where(x<0.,-1.,1.)
    sy=np.where(y>0.,-1.,1.)
    sz=np.where(z>0.,-1.,1.)
    x*=sx;y*=sy;z*=sz
    xma,xpa=x-half,x+half
    ymb,ypb=y-half,y+half
    zmc,zpc=z-half,z+half
    xma2,xpa2=xma*xma,xpa*xpa
    ymb2,ypb2=ymb*ymb,ypb*ypb
    zmc2,zpc2=zmc*zmc,zpc*zpc
    mmm=np.sqrt(xma2+ymb2+zmc2)
    pmp=np.sqrt(xpa2+ymb2+zpc2)
    pmm=np.sqrt(xpa2+ymb2+zmc2)
    mmp=np.sqrt(xma2+ymb2+zpc2)
    mpm=np.sqrt(xma2+ypb2+zmc2)
    ppp=np.sqrt(xpa2+ypb2+zpc2)
    ppm=np.sqrt(xpa2+ypb2+zmc2)
    mpp=np.sqrt(xma2+ypb2+zpc2)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        ff2x=np.log((xma+mmm)*(xpa+ppm)*(xpa+pmp)*(xma+mpp))-np.log((xpa+pmm)*(xma+mpm)*(xma+mmp)*(xpa+ppp))
        ff2y=np.log((-ymb+mmm)*(-ypb+ppm)*(-ymb+pmp)*(-ypb+mpp))-np.log((-ymb+pmm)*(-ypb+mpm)*(ymb-mmp)*(ypb-ppp))
        ff2z=np.log((-zmc+mmm)*(-zmc+ppm)*(-zpc+pmp)*(-zpc+mpp))-np.log((-zmc+pmm)*(zmc-mpm)*(-zpc+mmp)*(zpc-ppp))
    xx=(np.arctan2(ymb*zmc,xma*mmm)-np.arctan2(ymb*zmc,xpa*pmm)
        -np.arctan2(ypb*zmc,xma*mpm)+np.arctan2(ypb*zmc,xpa*ppm)
        -np.arctan2(ymb*zpc,xma*mmp)+np.arctan2(ymb*zpc,xpa*pmp)
        +np.arctan2(ypb*zpc,xma*mpp)-np.arctan2(ypb*zpc,xpa*ppp))
    yy=(np.arctan2(xma*zmc,ymb*mmm)-np.arctan2(xpa*zmc,ymb*pmm)
        -np.arctan2(xma*zmc,ypb*mpm)+np.arctan2(xpa*zmc,ypb*ppm)
        -np.arctan2(xma*zpc,ymb*mmp)+np.arctan2(xpa*zpc,ymb*pmp)
        +np.arctan2(xma*zpc,ypb*mpp)-np.arctan2(xpa*zpc,ypb*ppp))
    tensor=np.empty((len(points),3,3))
    tensor[:,0,0]=xx;tensor[:,1,1]=yy;tensor[:,2,2]=-xx-yy
    tensor[:,0,1]=tensor[:,1,0]=ff2z*sx*sy
    tensor[:,0,2]=tensor[:,2,0]=ff2y*sx*sz
    tensor[:,1,2]=tensor[:,2,1]=-ff2x*sy*sz
    tensor/=4.*np.pi
    tensor[~external|~np.all(np.isfinite(tensor),axis=(1,2))]=np.nan
    return tensor
