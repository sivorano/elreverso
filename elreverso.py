import numpy as np
from scipy import ndimage
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





def PhotometricStereoNormals(imgArr,mask,LightDirections):
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

def PhotometricStereoNormals2(imgArr,mask,LightDirections):
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


def PoissonFDEequations(mask,p,q,b):
    """
    Creates the linear equation system for solving a 2D poisson problem,
    Assumes that the image is square?(*)
    Assumes that the value on the boundary is 0 for ∇z

    Uses the stencil 
    0  -1  0
    -1  4 -1
    0  -1  0

    For aproximating the laplacian Δz
    """
    
    N = mask.shape[0]
    M = mask.shape[1]

    #Our matrix for the linear equations
    A = lil_matrix((M*N, M*N))

    for i in range(0,M*N):

        #We incorparete the stencil into A:
        
        if mask[np.mod(i,N),int(i/N)] > 0:
            #We are inside the mask -> inside Ω. As such, the center is multiplied by 4 
            A[i,i] = 4 * mask[np.mod(i,N),int(i/N)]
                    
            # The right side value: We multiply with the mask to check if we have hit ∂Ω or beyond.
            # We multiply by (np.mod(i + 1,N) != 0) to check if we have hit the right side of the
            # image. In that case, there is no value to the right.
            # The same idea goes for the other lines
            if i < M*N - 1:
                A[i + 1,i] = -1. * mask[np.mod(i+1,N),int((i+1)/N)] * (np.mod(i + 1,N) != 0)

            # Left side
            if i > 0:
                A[i - 1,i] = -1. * mask[np.mod(i-1,N),int((i-1)/N)] * (np.mod(i,N) != 0)

            # Above
            if i + N < M*N:
                A[i + N,i] = -1. * mask[np.mod(i+N,N),int((i+N)/N)]

                # Bellow
            if i - N >= 0:
                A[i - N,i] = -1.  * mask[np.mod(i-N,N),int((i - N )/N)]

        else:
            #We are outside the mask -> on the boundary or beyond. thus b[i] = 0, so we just
            #set A[i,i] = 1, so that the value becomes 0, and the system invertible.
            A[i,i] = 1 



    return (csr_matrix(A),b)




def PoissonFDEequations2(mask,p,q,b):
    """
    Creates the linear equation system for solving a 2D poisson problem,
    Assumes that the image is square?(*)
    Assumes that the value on the boundary is 0 for ∇z

    Uses the stencil 
    0  -1  0
    -1  4 -1
    0  -1  0

    For aproximating the laplacian Δz
    """
    
    N = mask.shape[0]
    M = mask.shape[1]

    #Our matrix for the linear equations
    A = lil_matrix((M*N, M*N))

    edgefinder = np.array([[0,1,0],[1,0,1],[0,1,0]])
    up = mask - (scipy.ndimage.filters.correlate(mask,[[0,1,0],[0,0,0],[0,0,0]])*mask)
    down = mask - (scipy.ndimage.filters.correlate(mask,[[0,0,0],[0,0,0],[0,1,0]])*mask)
    left = mask - (scipy.ndimage.filters.correlate(mask,[[0,0,0],[1,0,0],[0,0,0]])*mask)
    right = mask - (scipy.ndimage.filters.correlate(mask,[[0,0,0],[0,0,1],[0,0,0]])*mask)
    boundary = (up + down + left + right) > 0

    nb = b
    for i in range(0,M*N):

        #We incorparete the stencil into A:

        if boundary[np.mod(i,N),int(i/N)] > 0:
            a = (np.mod(i,N),int(i/N))
            v = up[a] * np.array([1,0]) + down[a] * np.array([-1,0]) + left[a] * np.array([0,-1]) + right[a] * np.array([0,1])
            v = v/np.linalg.norm(a)
            ()

        cor = (np.mod(i,N),int(i/N))
        if mask[cor] > 0:
            #We are inside the mask -> inside Ω. As such, the center is multiplied by 4 
            A[i,i] = 4 * mask[np.mod(i,N),int(i/N)]

            # The right side value: We multiply with the mask to check if we have hit ∂Ω or beyond.
            # We multiply by (np.mod(i + 1,N) != 0) to check if we have hit the right side of the
            # image. In that case, there is no value to the right.
            # The same idea goes for the other lines
            if i < M*N - 1:
                A[i + 1,i] = -1. * mask[np.mod(i+1,N),int((i+1)/N)] * (np.mod(i + 1,N) != 0)

            # Left side
            if i > 0:
                A[i - 1,i] = -1. * mask[np.mod(i-1,N),int((i-1)/N)] * (np.mod(i,N) != 0)

            # Above
            if i + N < M*N:
                A[i + N,i] = -1. * mask[np.mod(i+N,N),int((i+N)/N)]

                # Bellow
            if i - N >= 0:
                A[i - N,i] = -1.  * mask[np.mod(i-N,N),int((i - N )/N)]

            if up[np.mod(i,N),int(i/N)] == 1:
                A[i + 1,i] += 1
                A[i - 1,i]
            
            #FIXME: Also check if near edge
            #FIXME: Korrekt fortegn?
            if up[cor]:
                A[i,i-1] += -1
                b[i] = b[i] - 2*q[cor]
            if down[cor]:
                A[i,i+1] += -1
                b[i] = b[i] + 2*q[cor]
            if left[cor]:
                A[i+1,i] += -1
                b[i] = b[i] - 2*p[cor]
            if right[cor]:
                A[i-1,i] += -1
                b[i] = b[i] + 2*p[cor]

                
        else:
            #We are outside the mask -> on the boundary or beyond. thus b[i] = 0, so we just
            #set A[i,i] = 1, so that the value becomes 0, and the system invertible.
            A[i,i] = 1 



    return (csr_matrix(A),b)


#FIXME: der sker mærkelige ting foruden transponering! (*)
def PoissonSolverPS(normals,mask):
    """
    Solves the 2D case of the possion problem given unit normals
    """

    p = -normals[0]/normals[2]
    q = -normals[1]/normals[2]

    
    b = CentralDifference(p,q).T.ravel()

    A,b = PoissonFDEequations(mask,p,q,b)

    z = spsolve(A,b)
    z = -z.reshape(p.shape).T
    #z[mask == 0] = np.nan
    return (A,b,z)



def PhotometricStereoSolver(imgArr,mask,LightDirections,display = False):
    """

    """
    normals = PhotometricStereoNormals(imgArr,mask,LightDirections)
    z = PoissonSolverPS(normals,mask)[2]

    if display:
        display_surface(z)

    return z

if dotest and True:
    
    S = np.array([[0,0,1],
                  [-1/sqrt(2),0,1/sqrt(2)],
                  [1/sqrt(2),0,1/sqrt(2)],
                  [1,0,0],
                  [-1,0,0]])

    img0 = skimage.color.rgb2gray(skimage.io.imread('./syndata-pic-0.png'))
    img1 = skimage.color.rgb2gray(skimage.io.imread('./syndata-pic-1.png'))
    img2 = skimage.color.rgb2gray(skimage.io.imread('./syndata-pic-2.png'))
    img3 = skimage.color.rgb2gray(skimage.io.imread('./syndata-pic-3.png'))
    img4 = skimage.color.rgb2gray(skimage.io.imread('./syndata-pic-4.png'))

    imgs = np.moveaxis(np.array([img0,img1,img2,img3,img4]),0,2)
    
    # plt.imshow(img1)
    # plt.show()
    
    threshold = 0.005
    img0Mask = (img0 > threshold).astype(int)
    img1Mask = (img1 > threshold).astype(int)
    img2Mask = (img2 > threshold).astype(int)
    img3Mask = (img3 > threshold).astype(int)
    img4Mask = (img4 > threshold).astype(int)

    imgMask = ((img0Mask + img1Mask + img2Mask + img3Mask + img4Mask) > 2).astype(int)
    zimgs = PhotometricStereoSolver(imgs,imgMask,S,display = True)
    ()


if (__name__ == "__main__" or dotest) and False:
    DataPath = "../../Specialprojekt/tilAnders/Beethoven.mat"

    Images, mask, S = read_data_file(DataPath)


    normals = PhotometricStereoNormals(Images,mask,S)


    res = PoissonSolverPS(normals,mask)

    reshaped = res[2]


    p = -normals[0]/normals[2]
    q = -normals[1]/normals[2]

    cd = CentralDifference(p,q)


    _,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(normals[0])
    ax2.imshow(normals[1])
    ax3.imshow(normals[2])
    plt.show()

    _,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(p)
    ax2.imshow(q)
    ax3.imshow(cd)
    plt.show()
    edgefinder = np.array([[0,1,0],[1,0,1],[0,1,0]])
    plt.imshow(mask - (scipy.ndimage.filters.convolve(mask,edgefinder) == 4))

    plt.imshow(reshaped)
    plt.show()
    z = PhotometricStereoSolver(Images,mask,S,display = True)
