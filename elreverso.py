
import numpy as np
from scipy import ndimage
import scipy
from scipy.sparse import coo_matrix,csr_matrix,bmat,lil_matrix
import matplotlib.pyplot as plt
from scipy.optimize import nnls,lsq_linear
from scipy.sparse.linalg import spsolve

import ps_utils_test
from ps_utils_test import *


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





def PhotometricStereo(imgArr,mask,LightDirections):
    """
    imgList is assumed to be a 3d array, with imgArr[x,y,i]
    shape (len,height,3)
    """

    l, h = imgArr[:,:,0].shape

    #The indices we actually want to calculate on
    notMasked = np.nonzero(mask)

    m  = (np.linalg.inv(LightDirections) @ imgArr[notMasked].T)
    
    ρ = np.linalg.norm(m,axis=0)
    
    normals = 1/ρ * m

    n1 = np.zeros((l,h))
    n2 = np.zeros((l,h))
    n3 = np.ones((l,h))

    n1[notMasked] = normals[0]
    n2[notMasked] = normals[1]
    n3[notMasked] = normals[2]

    
    return(n1,n2,n3)

def NormalsToDivergence(Normals):

    dx = -Normals[0]/Normals[2]
    dy = -Normals[1]/Normals[2]
    return dx + dy



def Amaker2(p,q,mask = None):
    N = p.shape[0]
    M = p.shape[1]
    λ = 1.

    Omega_padded = np.pad(mask, (1,1), mode='constant', constant_values=0)
    down = mask - Omega_padded[2:,1:-1]*mask
    up = mask - Omega_padded[:-2,1:-1]*mask
    right = mask - Omega_padded[1:-1,2:]*mask
    left = mask - Omega_padded[1:-1,:-2]*mask

    boundary = down + up + left + right
    
    rmask = mask.ravel()
    nmask  = (boundary + 4*mask) >0
    
    A = lil_matrix((M*N, M*N))

    for i in range(0,M*N):
        if mask[np.mod(i,N),int(i/N)] > 0:
            A[i,i] = 4 * mask[np.mod(i,N),int(i/N)]
        else:
            A[i,i] = 1

        if i < M*N - 1:
            A[i + 1,i] = -1. * mask[np.mod(i+1,N),int((i+1)/N)] * (np.mod(i + 1,N) != 0)

        if i > 0:
            A[i - 1,i] = -1. * mask[np.mod(i-1,N),int((i-1)/N)] * (np.mod(i,N) != 0)

        if i + N < M*N:
            A[i + N,i] = -1. * mask[np.mod(i+N,N),int((i+N)/N)]

        if i - N >= 0:
            A[i - N,i] = -1.  * mask[np.mod(i-N,N),int((i - N )/N)]

        # if boundary[np.mod(i,N),int(i/N)] > 0:
        #     A[i,i] = 1


    return A


#FIXME: der sker mærkelige ting foruden transponering! (*)
def PoissonSolver2D(normals,mask):
    """
    Solves in the 2D case, 0 on the boundary 
    """

    p = -normals[0]/normals[2]
    q = -normals[1]/normals[2]
    
    #Find ud af hvilken form X har
    N = p.shape[0]
    M = p.shape[1]
    b = (cdx(p) + cdy(q)).T.ravel()#(p + q).T.ravel()
    # b = ps_utils_test.cdx(p) + ps_utils_test.cdy(q) 
    # b = np.ones_like(p).ravel()
    # b = np.flip(p+ q,axis = 0).T.ravel()
    #b = (p + q)[mask > 0]

    #1 if X is square

    A = Amaker2(p,q,mask)
    # A = csr_matrix(bmat(A))

    #A = csr_matrix(Amaker(X,mask))

    #return (A,b,lsq_linear(A,b,verbose = 2).x)#implementer egen (*)
    # res = lsq_linear(A,b,verbose = 2).x
    # z = res.reshape((N,M))
    z = spsolve(A,b)
    z = -z.reshape((N,M)).T
    return (A,b,z)


testX = np.arange(0,25,1).reshape((5,5))

stencil = np.array([[0,1,0],[1,1,1],[0,1,0]])

ndimage.convolve(testX,stencil,mode = "constant")


#testRes = PoissonSolver2D(testX,testX)
#print(np.linalg.norm(testRes[0].dot(testRes[2]) - testRes[1]))

DataPath = "../Specialprojekt/tilAnders/Beethoven.mat"

Images, mask, S = read_data_file(DataPath)


normals = PhotometricStereo(Images,mask,S)

div = NormalsToDivergence(normals)



res = PoissonSolver2D(normals,mask)

reshaped = res[2]


#display_surface(reshaped)

#print(np.linalg.norm(res[0].dot(res[2])-res[1]))
# Images[:,:,0][np.where(mask)]



_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(normals[0])
ax2.imshow(normals[1])
ax3.imshow(normals[2])
plt.show()

_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(-normals[0]/normals[2])
ax2.imshow(-normals[1]/normals[2])
ax3.imshow(NormalsToDivergence(normals))
plt.show()



plt.imshow(reshaped)
plt.show()
ps_utils_test.display_surface(reshaped)


# z = simchony_integrate(normals[0],normals[1],normals[2],mask)


# plt.imshow(z)
# plt.show()

# plt.imshow(z[0] - res[2])
# plt.show()




"""
testN = np.array([[1,0,0],[0,1,0],[0,0,1]])
testimg = np.ones((5,3,10))
testimg2 = np.array([[[2.0,3.0,4.0],[0.0,1.0,2.0]]])
nmatrix = np.dot(testN,testimg)
for i in range(0,5):
    for j in range(0,10):
        print(nmatrix[:,i,j] == np.dot(testN,testimg[i,:,j]))
        
PhotometricStereo(testimg2,testN)
"""
