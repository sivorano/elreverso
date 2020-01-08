# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 09:39:34 2020

@author: Anders Samsø Birch
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters

import scipy
from scipy.sparse import coo_matrix,csr_matrix,bmat,lil_matrix
import matplotlib.pyplot as plt
from scipy.optimize import nnls,lsq_linear
from scipy.sparse.linalg import spsolve
import matplotlib.image as mpimg
import math
from math import sqrt
import skimage
from skimage import color
from skimage import io
from os import listdir
import ps_utils_test

dotest = True


#From Fracois ps_utils
def read_data_file(filename):
    """
    Read a matlab PS data file and returns
    - the images as a 3D array of size (m,n,nb_images)
    - the mask as a 2D array of size (m,n) with 
      mask > 0 meaning inside the mask
    - the light matrix S as a (nb_images, 3) matrix
    """
    from scipy.io import loadmat
    
    data = loadmat(filename)
    I = data['I']
    mask = data['mask']
    S = data['S']
    return I, mask, S



#Code from francois
def make_bc_data(mask):
     """
     Create the data structure used to enforce some  null Neumann BC condition on
     some PDEs used in my Photometric Stereo Experiments.
     Argument:
     ---------
     mask: numpy array
         a binary mask of size (m,n).
     Returns:
     --------
         west, north, east, south, inside, n_pixels with
         west[i]  index of point at the "west"  of mask[inside[0][i],inside[1][i]]
         north[i] index of point at the "north" of mask[inside[0][i],inside[1][i]]
         east[i]  index of point at the "east"  of mask[inside[0][i],inside[1][i]]
         south[i] index of point at the "south" of mask[inside[0][i],inside[1][i]]
         inside: linear indices of points inside the mask
         n_pixels: number of inside / in domain pixels
     """
     m,n = mask.shape
     inside = np.where(mask)
     x, y = inside
     n_pixels = len(x)
     m2i = -np.ones(mask.shape)
     # m2i[i,j] = -1 if (i,j) not in domain, index of (i,j) else.
     m2i[(x,y)] = range(n_pixels)
     west  = np.zeros(n_pixels, dtype=int)
     north = np.zeros(n_pixels, dtype=int)
     east  = np.zeros(n_pixels, dtype=int)
     south = np.zeros(n_pixels, dtype=int)


     for i in range(n_pixels):
         xi = x[i]
         yi = y[i]
         wi = x[i] - 1
         ni = y[i] + 1
         ei = x[i] + 1
         si = y[i] - 1

         west[i]  = m2i[wi,yi] if (wi > 0) and (mask[wi, yi] > 0) else i
         north[i] = m2i[xi,ni] if (ni < n) and (mask[xi, ni] > 0) else i
         east[i]  = m2i[ei,yi] if (ei < m) and (mask[ei, yi] > 0) else i
         south[i] = m2i[xi,si] if (si > 0) and (mask[xi, si] > 0) else i

     return west, north, east, south, inside, n_pixels




# Same
def display_surface(z):
    """
    Display the computed depth function as a surface using 
    mayavi mlab.
    """
    from mayavi import mlab
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    
    mlab.mesh(x, y, z, scalars=z, colormap="Greys")
    mlab.show()


def CentralDifference(p,q):
    """
    p is horizontal, q is vertiacal
    see page 7 for defintion
    """

    n,m = p.shape

    left1 = np.append([0],np.arange(0,n-1))
    right1 = np.append(np.arange(1,n),[n-1])

    left2 = np.append([0],np.arange(0,m-1))
    right2 = np.append(np.arange(1,m),[m-1])

    horizontalDiff = p[right1,:] - p[left1,:]
    verticalDiff = q[:,right2] - q[:,left2]

    return (1/2)*(horizontalDiff + verticalDiff)



def CentralDifferenceX(p,q):
    """
    p is horizontal, q is vertiacal
    see page 7 for defintion
    """

    n,m = p.shape

    left1 = np.append([0],np.arange(0,n-1))
    right1 = np.append(np.arange(1,n),[n-1])

    horizontalDiff = p[right1,:] - p[left1,:]

    return (1/2)*(horizontalDiff)



def CentralDifferenceY(q):
    """
    p is horizontal, q is vertiacal
    see page 7 for defintion
    """

    n,m = p.shape

    left2 = np.append([0],np.arange(0,m-1))
    right2 = np.append(np.arange(1,m),[m-1])

    verticalDiff = q[:,right2] - q[:,left2]

    return (1/2)*(verticalDiff)



def imgImporter(path,filename,seperator,threshold = 0.005,lightlim = 4):

    imgs = []
    S = []
    for name in listdir(path):
        # print(name)
        if str.startswith(name,filename):
            imgs.append(skimage.color.rgb2gray(skimage.io.imread(path + name)))
            S.append(eval(str.split(name,seperator)[6]))
    # plt.imshow(imgs[0])
    # plt.show()
    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs,0,2)
    S = np.array(S)

    imgMasks = (imgs > threshold).astype(int)
    imgMask = (np.sum(imgMasks,axis = 2) > lightlim).astype(int)

    return (imgs,imgMask,S)
    
def PhotometricStereoNormalsOld(imgArr,mask,LightDirections, display = False):
    """
    imgList is assumed to be a 3d array, with imgArr[x,y,i]

    """

    l, h = imgArr[:,:,0].shape

    #The indices we actually want to calculate on
    notMasked = np.nonzero(mask)
    print(imgArr[notMasked].shape)
    if len(LightDirections) == 3:
        m  = (np.linalg.inv(LightDirections) @ imgArr[notMasked].T)
    else:
        m =  (np.linalg.pinv(LightDirections) @ imgArr[notMasked].T)
    ρ = np.linalg.norm(m,axis=0)
    
    normals = 1/ρ * m

    n1 = np.zeros((l,h))
    n2 = np.zeros((l,h))
    n3 = np.ones((l,h))

    n1[notMasked] = normals[0]
    n2[notMasked] = normals[1]
    n3[notMasked] = normals[2]

    return(n1,n2,n3)

    
def PhotometricStereoNormals(imgArr,mask,LightDirections, display = False):
    """
    imgList is assumed to be a 3d array, with imgArr[x,y,i]

    """

    m,n = imgArr[:,:,0].shape

    iS = np.linalg.inv(LightDirections)
    notMasked = np.nonzero(mask)
    npix = len(notMasked[0])
    I = np.zeros((3,npix))
    I[0,:] = imgArr[:,:,0][notMasked]
    I[1,:] = imgArr[:,:,1][notMasked]
    I[2,:] = imgArr[:,:,2][notMasked]

    N = np.dot(iS, I)
    Rho = np.sqrt(N[0,:]**2 + N[1,:]**2 + N[2,:]**2)
    rho = np.zeros((m,n))
    rho[notMasked] = Rho


    n1 = np.zeros((m,n))
    n2 = np.zeros((m,n))
    n3 = np.ones((m,n))

    n1[notMasked] = N[0,:]/Rho
    n2[notMasked] = N[1,:]/Rho
    n3[notMasked] = N[2,:]/Rho

    return(n1,n2,n3)


def PoissonFDEequationsWithVonNeumann(mask,p,q,b):
    """
    Creates the linear equation system for solving a 2D poisson problem,
    Assumes that the image is square?

    Uses the stencil 
    0  -1  0
    -1  4 -1
    0  -1  0

    For aproximating the laplacian Δz
    """
    
    west, north, east, south, inside, n_pixels = make_bc_data(mask)
    
    
    # We use central (and a bit of foward) aproximation for finding nabla u
    cdx =  1/2*(p[inside][east] - p[inside][west]) 
    cdy =  1/2*(q[inside][north] - q[inside][south])
        
    if 1 == 1: #For turning things into forward aproximation
        cdx[east==np.arange(0,n_pixels)] = cdx[east==np.arange(0,n_pixels)]*2 
        cdx[west==np.arange(0,n_pixels)] = cdx[west==np.arange(0,n_pixels)]*2 
        cdy[north==np.arange(0,n_pixels)] = cdy[north==np.arange(0,n_pixels)]*2 
        cdy[south==np.arange(0,n_pixels)] = cdy[south==np.arange(0,n_pixels)]*2 
    
    cd = cdx + cdy
    nb = cd
    
    A = lil_matrix((n_pixels, n_pixels))
    
    
    for i in range(0,n_pixels):

        for direction,opdir,u in [(west,east,p), (north,south,q), (east,west,p), (south,north,q)]:
            if direction[i] != i: # if the point in that direction is inside
                A[i,i] += 1
                A[direction[i],i] += -1
            else: #If the point is outside
                
                nb[i] += 2*u[inside][i] #boundary condition
                A[opdir[i],i] += -1
                A[i,i] += 2  #Should be 1, but then the result explodes!


    return (csr_matrix(A),nb)


def PoissonFDEequationsWithVonNeumann2(mask,p,q,b):
    """
    Creates the linear equation system for solving a 2D poisson problem,
    Assumes that the image is square?

    Uses the stencil 
    0  -1  0
    -1  4 -1
    0  -1  0

    For aproximating the laplacian Δz
    """
    
    west, north, east, south, inside, n_pixels = make_bc_data(mask)
    
    
    # We use central (and a bit of foward) aproximation for finding nabla u
    cdx =  1/2*(p[inside][east] - p[inside][west]) 
    cdy =  1/2*(q[inside][north] - q[inside][south])
        
    if 2 == 1:
        cdx[east==np.arange(0,n_pixels)] = cdx[east==np.arange(0,n_pixels)]*2 
        cdx[west==np.arange(0,n_pixels)] = cdx[west==np.arange(0,n_pixels)]*2 
        cdy[north==np.arange(0,n_pixels)] = cdy[north==np.arange(0,n_pixels)]*2 
        cdy[south==np.arange(0,n_pixels)] = cdy[south==np.arange(0,n_pixels)]*2 
    
    cd = cdx + cdy
    nb = cd
    
    A = lil_matrix((n_pixels, n_pixels))
    
    
#    for i in range(0,n_pixels):
#        if i == 19830:
#            print("Hej")
#        for direction,opdir,u in [(west,east,p), (north,south,q), (east,west,p), (south,north,q)]:
#            if direction[i] != i: # if the point in that direction is inside
#                A[i,i] += 1
#                A[direction[i],i] += -1
#            else: #If the point is outside
#                
#                nb[i] += 2*u[inside][i] #boundary condition
#                A[opdir[i],i] -= 1
#                A[i,i] += 1  #Should be 1, but then the result explodes!
#        if A[i,i] != 4:
#            print(A[i,i])
#            print(i)    

    for i in range(0,n_pixels):
        if i == 19830:
            print("Hej")
        #np.zeros((n_pixels,))
        for direction,opdir,u in [(west,east,p), (north,south,q), (east,west,p), (south,north,q)]:
            if direction[i] == i and opdir[i] == 1:
                A[i,i] += 1
            elif direction[i] != i: # if the point in that direction is inside
                A[i,i] += 1
                A[direction[i],i] += -1
            elif opdir[i] != i: #If the point is outside, and the oposite point is inside:
                nb[i] += 2*u[inside][i] #boundary condition
                A[opdir[i],i] -= 1
                A[i,i] += 2  #Should be 1, but then the result explodes!
#        if A[i,i] != 4:
#            print(A[i,i])
#            print(i)
#    for i in range(0,n_pixels):
#        if counts[i] < 3:
#            print(i)
#            A[i,:] = 0
#            A[:,i] = 0
#            A[i,i] = 1
#            nb[i] = 0
#            
    return (csr_matrix(A),nb)


def PoissonFDEequationsWithVonNeumann3(mask,p,q,b):
    """
    Creates the linear equation system for solving a 2D poisson problem,
    Assumes that the image is square?

    Uses the stencil 
    0  -1  0
    -1  4 -1
    0  -1  0

    For aproximating the laplacian Δz
    """
    
    west, north, east, south, inside, n_pixels = make_bc_data(mask)
    
    pinside = (lambda i : np.array([inside[0][i],inside[1][i]]))
    # We use central (and a bit of foward) aproximation for finding nabla u
    cdx =  1/2*(p[inside][east] - p[inside][west]) 
    cdy =  1/2*(q[inside][north] - q[inside][south])

    pinside = p[inside]
    qinside = q[inside] 
    A = lil_matrix((n_pixels, n_pixels))


    hasNorth =  (north != np.arange(0,n_pixels))
    hasSouth =  (south != np.arange(0,n_pixels))
    hasEast  =  (east != np.arange(0,n_pixels))
    hasWest =   (west != np.arange(0,n_pixels))

    directionCounts = (hasSouth)*1 + (hasNorth)*1 + (hasEast)*1 + (hasWest)*1
    hasNS = hasNorth * hasSouth
    hasEW = hasEast * hasWest

    #nb = cdx*hasns + cdy*hasew

    if 2 == 1:
        cdx[hasEast == 0] = cdx[hasEast == 0]*2 
        cdx[hasWest == 0] = cdx[hasWest == 0]*2 
        cdy[hasNorth == 0] = cdy[hasNorth == 0]*2 
        cdy[hasSouth == 0] = cdy[hasSouth == 0]*2 
            

    cd = cdx + cdy
    nb = cd
    
    def displayPoint(i):
        cmask = mask.copy()
        cmask[inside[0][i],inside[1][i]] = 2
        plt.imshow(cmask)
        plt.show()
    
    for i in range(0,n_pixels):
        if directionCounts[i] == 4: # The point is inside, easy case
            A[i,i] = 4 
            A[north[i],i] = -1 #byt rundt på indeks!
            A[south[i],i] = -1
            A[east[i],i] = -1
            A[west[i],i] = -1
        elif directionCounts[i] == 3: # Just one missing coordinate
            A[i,i] = 4
#            A[north[i],i] = -1
#            A[south[i],i] = -1
#            A[east[i],i] = -1
#            A[west[i],i] = -1
#            
            #nu = [(hasNorth == 0)*1 - (hasSouth == 0)*1,(hasEast == 0)*1 - (hasWest == 0)*1]
            if north[i] == i: #North is missing            
                A[south[i],i] = -2
                A[east[i],i] = -1
                A[west[i],i] = -1
                nb[i] += -2*qinside[i]
            elif south[i] == i:
                A[north[i],i] = -2
                A[east[i],i] = -1
                A[west[i],i] = -1
                nb[i] += 2*qinside[i]
            elif east[i] == i: 
                A[south[i],i] = -1
                A[north[i],i] = -1
                A[west[i],i] = -2
                nb[i] += -2*pinside[i]
            else: # west[i] == i: 
                A[south[i],i] = -1
                A[north[i],i] = -1
                A[east[i],i] = -2
                nb[i] += 2*pinside[i]
        elif directionCounts[i] == 2:
            #Find out if corner or what
            if hasNS[i]: #East and west is missing
                A[i,i] = 2
                A[south[i],i] = -1
                A[north[i],i] = -1
           
            elif hasEW[i]:
                A[i,i] = 2
                A[east[i],i] = -1
                A[west[i],i] = -1
            else:
                #FIXME: Something Does not work
                A[i,i] = 4
                nu = 1/2*np.array([(hasEast[i] == 0)*1 - (hasWest[i] == 0)*1,(hasNorth[i] == 0)*1 - (hasSouth[i] == 0)*1])
                if hasNorth[i] + hasWest[i] == 0:
                    A[south[i],i] = -2
                    A[east[i],i]  = -2
                    nb[i] += nu[0]*2*pinside[i] #From North 
                    nb[i] += nu[1]*2*qinside[i] #From West

                elif hasWest[i] + hasSouth[i] == 0:
                    A[north[i],i] = -2
                    A[east[i],i]  = -2
                    nb[i] +=  nu[0]*2*pinside[i] #From South
                    nb[i] +=  nu[1]*2*qinside[i] #From West
                elif hasSouth[i] + hasEast[i] == 0:
                    A[north[i],i] = -2
                    A[west[i],i]  = -2
                    nb[i] +=  nu[0]*2*pinside[i] #From South
                    nb[i] +=  nu[1]*2*qinside[i] #From East
                elif hasEast[i] + hasNorth[i] == 0:
                    A[south[i],i] = -2
                    A[west[i],i]  = -2
                    nb[i] += nu[0]*2*pinside[i] #From North
                    nb[i] += nu[1]*2*qinside[i] #From East
                
                #print("Indeks " + str(i))
                #print(A[:,i])
                #A[i,i] = 1
                #nb[i] = 0
            ()
        elif directionCounts[i] == 1:
            A[i,i] = 1
            nb[i] = 0  
            #Using linear intepolation
            if north[i] != i: #North not missing            
                A[north[i],i] = -1
                nb[i] += qinside[i]
            if south[i] != i: #south not missing            
                A[south[i],i] = -1
                nb[i] += -qinside[i]
            if east[i] != i:  #east not missing            
                A[east[i],i] = -1
                nb[i] += pinside[i]
            if west[i] != i: #west not missing            
                A[west[i],i] = -1
                nb[i] += -pinside[i]
            print("Indeks " + str(i))
            print(A[:,i])
        else:
            A[i,i] = 1
            nb[i] = 0
    A = A.T
    return (csr_matrix(A),np.array([nb]).T)




def PoissonFDEequationsWithVonNeumann4(mask,p,q,b):
    """
    Creates the linear equation system for solving a 2D poisson problem,
    Assumes that the image is square?

    Uses the stencil 
    0  -1  0
    -1  4 -1
    0  -1  0

    For aproximating the laplacian Δz
    """
    
    west, north, east, south, inside, n_pixels = make_bc_data(mask)

    # cp = p.copy()
    # cq = q.copy()
    # p = cq
    # q = cp
    #FIXME: der er noget i vejen med normalfindingen, alt fungere fint med denne med
    p[240:,:] = 0
    
    pointinside = (lambda i : np.array([inside[0][i],inside[1][i]]))
    # We use central (and a bit of foward) aproximation for finding nabla u
    cdx =  1/2*(p[inside][east] - p[inside][west]) 
    cdy =  1/2*(q[inside][north] - q[inside][south])

 

    
    pinside = p[inside]
    qinside = q[inside] 
    A = lil_matrix((n_pixels, n_pixels))


    hasNorth =  (north != np.arange(0,n_pixels))
    hasSouth =  (south != np.arange(0,n_pixels))
    hasEast  =  (east != np.arange(0,n_pixels))
    hasWest =   (west != np.arange(0,n_pixels))

    directionCounts = (hasSouth)*1 + (hasNorth)*1 + (hasEast)*1 + (hasWest)*1
    hasNS = hasNorth * hasSouth
    hasEW = hasEast * hasWest

    #nb = cdx*hasns + cdy*hasew

    if 2 == 1:
        cdx[hasEast == 0] = cdx[hasEast == 0]*2 
        cdx[hasWest == 0] = cdx[hasWest == 0]*2 
        cdy[hasNorth == 0] = cdy[hasNorth == 0]*2 
        cdy[hasSouth == 0] = cdy[hasSouth == 0]*2 
            
   
    cd = cdx + cdy
    nb = cd

    cdxplot = np.zeros(mask.shape)
    cdyplot = np.zeros(mask.shape)
    cdplot  = np.zeros(mask.shape)
    cdplot[inside] = cd
    cdxplot[inside] = cdx
    cdyplot[inside] = cdy
    plt.imshow(cdxplot) ; plt.show()
    plt.imshow(cdyplot) ; plt.show()
    plt.imshow(cdplot) ; plt.show()

    
    def displayPoint(i):
        cmask = mask.copy()
        cmask[inside[0][i],inside[1][i]] = 2
        plt.imshow(cmask)
        plt.show()
    
    for i in range(0,n_pixels):
        if pointinside(i)[0] == 245 and pointinside(i)[1] > 180:
            ()
        if directionCounts[i] == 4: # The point is inside, easy case
            A[i,i] = 4
            A[i,north[i]] = -1
            A[i,south[i]] = -1
            A[i,east[i]] = -1
            A[i,west[i]] = -1
        elif 2 == 1:
            A[i,i] =1
            nb[i] = 0
        elif directionCounts[i] == 3: # Just one missing coordinate
            A[i,i] = 4
#            A[north[i],i] = -1
#            A[south[i],i] = -1
#            A[east[i],i] = -1
#            A[west[i],i] = -1
#            
            #nu = [(hasNorth == 0)*1 - (hasSouth == 0)*1,(hasEast == 0)*1 - (hasWest == 0)*1]
            sign = -1
            if north[i] == i: #North is missing            
                A[i,south[i]] = -2
                A[i,east[i]] = -1
                A[i,west[i]] = -1
                nb[i] += sign*qinside[i]
            elif south[i] == i:
                A[i,north[i]] = -2
                A[i,east[i]] = -1
                A[i,west[i]] = -1
                nb[i] += -sign*qinside[i]
            elif east[i] == i: 
                A[i,south[i]] = -1
                A[i,north[i]] = -1
                A[i,west[i]] = -2
                nb[i] += sign*pinside[i]
            else: # west[i] == i: 
                A[i,south[i]] = -1
                A[i,north[i]] = -1
                A[i,east[i]] = -2
                nb[i] += -sign*pinside[i]
        elif directionCounts[i] == 2:
            #Find out if corner or what
            if hasNS[i]: #East and west is missing
                A[i,i] = 2
                A[i,south[i]] = -1
                A[i,north[i]] = -1
           
            elif hasEW[i]: #North and south
                A[i,i] = 2
                A[i,east[i]] = -1
                A[i,west[i]] = -1
            else:
                #FIXME: Something Does not work
                A[i,i] = 4
                nu = -1/sqrt(2)*np.array([(hasEast[i] == 0)*1 - (hasWest[i] == 0)*1,(hasNorth[i] == 0)*1 - (hasSouth[i] == 0)*1])
                mult = -1/sqrt(2)
                if hasNorth[i] + hasWest[i] == 0:
                    A[i,south[i]] = -1 + mult
                    A[i,east[i]]  = -1 + mult
                    nb[i] += nu[0]*pinside[i] 
                    nb[i] += nu[1]*qinside[i] 

                elif hasWest[i] + hasSouth[i] == 0:
                    A[i,north[i]] = -1 + mult
                    A[i,east[i]]  = -1 + mult
                    nb[i] +=  nu[0]*pinside[i] 
                    nb[i] +=  nu[1]*qinside[i]
                elif hasSouth[i] + hasEast[i] == 0:
                    A[i,north[i]] = -1 + mult
                    A[i,west[i]]  = -1 + mult
                    nb[i] +=  nu[0]*pinside[i]
                    nb[i] +=  nu[1]*qinside[i] 
                elif hasEast[i] + hasNorth[i] == 0:
                    A[i,south[i]] = -1 + mult
                    A[i,west[i]]  = -1 + mult
                    nb[i] += nu[0]*pinside[i] 
                    nb[i] += nu[1]*qinside[i] 
                
                #print("Indeks " + str(i))
                #print(A[:,i])
                # A[i,:] = 0
                # A[i,i] = 1
                # nb[i] = 0
            ()
        elif directionCounts[i] == 1:
            A[i,i] = 1
            nb[i] = 0  
            sign = -1
            #Using linear intepolation
            if north[i] != i: #North not missing            
                A[i,north[i]] = -1
                nb[i] += sign*qinside[i]
            if south[i] != i: #south not missing            
                A[i,south[i]] = -1
                nb[i] += -sign*qinside[i]
            if east[i] != i:  #east not missing            
                A[i,east[i]] = -1
                nb[i] += sign*pinside[i]
            if west[i] != i: #west not missing            
                A[i,west[i]] = -1
                nb[i] += -sign*pinside[i]
            print("Indeks " + str(i))
            print(A[:,i])
        else:
            A[i,i] = 1
            nb[i] = 0
            
    if 2 == 1:
        A = A.T
            
    return (csr_matrix(A),nb)


def PoissonSolverPS(normals,mask):
    """
    Solves the 2D case of the possion problem given unit normals
    """

    p = -normals[0]/normals[2]
    q = -normals[1]/normals[2]

    
    b = CentralDifference(p,q)

    A,b = PoissonFDEequationsWithVonNeumann4(mask,p,q,b)
    # A,b = PoissonFDEequationsOld(mask,p,q,b)

    #plt.plot(b) ; plt.show()
    
    z = np.zeros(mask.shape)
    vals = spsolve(A,b)
    #vals = (vals - np.mean(np.abs(vals)))
    z[np.where(mask > 0)] = vals
    # z = vals.reshape(p.shape).T
    z[mask == 0] = np.nan
    z = -z

    
    return (A,b,z)


def PhotometricStereoSolver(imgArr,mask,LightDirections,display = False,cheat = False):
    """

    """
    normals = PhotometricStereoNormals(imgArr,mask,LightDirections,display)
    #mask = np.zeros_like(mask)
    #mask[10:245,30:230] = 1
    z = PoissonSolverPS(normals,mask)[2]
    #z = ps_utils_test.unbiased_integrate(normals[0],normals[1],normals[2],mask)[0]    
    if display:
        
        p = -normals[0]/normals[2]
        q = -normals[1]/normals[2]

        cd = CentralDifference(p,q)

        # _,(ax1,ax2,ax3) = plt.subplots(1,3)
        # ax1.imshow(p)
        # ax2.imshow(q)
        # ax3.imshow(cd)
        # plt.show()

        plt.imshow(z)
        plt.show()

        display_surface(z)

    return z



if (__name__ == "__main__" or dotest) and True:
    DataPath = "../../Specialprojekt/tilAnders/Beethoven.mat"

    Images, mask, S = read_data_file(DataPath)
    # _,(ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.imshow(Images[:,:,0],cmap='gray')
    # ax2.imshow(Images[:,:,1],cmap='gray')
    # ax3.imshow(Images[:,:,2],cmap='gray')
    # plt.show()
#    
#    
#    
#    plt.imshow(west) ; plt.show()
#    plt.imshow(north) ; plt.show()
#    plt.imshow(inside) ; plt.show()
    z = PhotometricStereoSolver(Images,mask,S,display = True,cheat = False)
    
    
