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
    
def PhotometricStereoNormals(imgArr,mask,LightDirections, display = False):
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

    if display:
        _,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(n1)
        ax2.imshow(n2)
        ax3.imshow(n3)
        plt.show()
        ρimg = np.zeros((l,h))
        ρimg[notMasked] = ρ
        plt.imshow(ρimg)
        plt.show()

    return(n1,n2,n3)

def PhotometricStereoNormals2(imgArr,masks,LightDirections):
    """
    imgList is assumed to be a 3d array, with imgArr[x,y,i]

    """
    mask = masks[0]

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
                # A[i,i] +=  mask[np.mod(i+1,N),int((i+1)/N)] - 1

            # Left side
            if i > 0:
                A[i - 1,i] = -1. * mask[np.mod(i-1,N),int((i-1)/N)] * (np.mod(i,N) != 0)
                # A[i,i] +=  mask[np.mod(i-1,N),int((i-1)/N)] - 1
                
            # Above
            if i + N < M*N:
                A[i + N,i] = -1. * mask[np.mod(i+N,N),int((i+N)/N)]
                # A[i,i] +=  mask[np.mod(i+N,N),int((i+N)/N)] - 1
                
                # Bellow
            if i - N >= 0:
                A[i - N,i] = -1.  * mask[np.mod(i-N,N),int((i - N )/N)]
                # A[i,i] +=  mask[np.mod(i - N,N),int((i - N)/N)] - 1
        else:
            #We are outside the mask -> on the boundary or beyond. thus b[i] = 0, so we just
            #set A[i,i] = 1, so that the value becomes 0, and the system invertible.
            A[i,i] = 1 



    return (csr_matrix(A),b)




def PoissonFDEequationsOld(mask,p,q,b):
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
            fortegn = 1
            if up[cor]:
                A[i,i-1] += -1
                b[i] = b[i] + fortegn*2*q[cor]
            if down[cor]:
                A[i,i+1] += -1
                b[i] = b[i] - fortegn*2*q[cor]
            if left[cor]:
                A[i+1,i] += -1
                b[i] = b[i] + fortegn*2*p[cor]
            if right[cor]:
                A[i-1,i] += -1
                b[i] = b[i] - fortegn*2*p[cor]

                
        else:
            #We are outside the mask -> on the boundary or beyond. thus b[i] = 0, so we just
            #set A[i,i] = 1, so that the value becomes 0, and the system invertible.
            A[i,i] = 1 



    return (csr_matrix(A),nb)


#FIXME: Ingnores von neumann boundary condition
def PoissonFDEequationsNoVonNeumann(mask,p,q,b):
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
    hasup = (filters.correlate(mask,[[0,1,0],[0,0,0],[0,0,0]],mode = "constant")*mask)
    hasdown = (filters.correlate(mask,[[0,0,0],[0,0,0],[0,1,0]],mode = "constant")*mask)
    hasleft = (filters.correlate(mask,[[0,0,0],[1,0,0],[0,0,0]],mode = "constant")*mask)
    hasright =  (filters.correlate(mask,[[0,0,0],[0,0,1],[0,0,0]],mode = "constant")*mask)



    maskList = mask.T.ravel()
    notMasked = np.nonzero(mask)
    count = len(notMasked[0])
    indexConverter = -np.ones(mask.shape,dtype = int)
    indexConverter[notMasked] = np.arange(0,count)

    upConverter = filters.correlate(indexConverter,[[0,1,0],[0,0,0],[0,0,0]],mode = "constant")*mask
    downConverter = filters.correlate(indexConverter,[[0,0,0],[0,0,0],[0,1,0]],mode = "constant")*mask
    leftConverter = filters.correlate(indexConverter,[[0,0,0],[1,0,0],[0,0,0]],mode = "constant")*mask
    rightConverter = filters.correlate(indexConverter,[[0,0,0],[0,0,1],[0,0,0]],mode = "constant")*mask

    newb = np.zeros(p.shape)
    newb += -1/2*filters.correlate(p,[[0,1,0],[0,1,0],[0,0,0]],mode = "constant")* hasup
    newb += 1/2*filters.correlate(p,[[0,0,0],[0,1,0],[0,1,0]],mode = "constant")* hasdown
    newb += 1/2*filters.correlate(q,[[0,0,0],[0,1,1],[0,0,0]],mode = "constant")* hasright
    newb += -1/2*filters.correlate(q,[[0,0,0],[1,1,0],[0,0,0]],mode = "constant")* hasleft
    
    
    
    A = lil_matrix((count, count))

    X,Y = notMasked
    
    
    def checker(hasdir,dirConverter):
        indexHasDir = hasdir[(X,Y)]
        wherehasdir = np.nonzero(indexHasDir)[0]
        indexDir = dirConverter[(X,Y)][wherehasdir]
        A[(indexDir,wherehasdir)] = A[(indexDir,wherehasdir)].toarray() - 1
        A[(wherehasdir,wherehasdir)] = A[(wherehasdir,wherehasdir)].toarray() + 1


        
    checker(hasup,upConverter)
    checker(hasdown,downConverter)
    checker(hasleft,leftConverter)
    checker(hasright,rightConverter)



    nb = newb[mask != 0] #barr #(b)[mask != 0]

    
    return (csr_matrix(A),nb)


def PoissonFDEequationsWithVonNeumann(mask,p,q,b):
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
    hasup = (filters.correlate(mask,[[0,1,0],[0,0,0],[0,0,0]],mode = "constant")*mask)
    hasdown = (filters.correlate(mask,[[0,0,0],[0,0,0],[0,1,0]],mode = "constant")*mask)
    hasleft = (filters.correlate(mask,[[0,0,0],[1,0,0],[0,0,0]],mode = "constant")*mask)
    hasright =  (filters.correlate(mask,[[0,0,0],[0,0,1],[0,0,0]],mode = "constant")*mask)



    maskList = mask.T.ravel()
    notMasked = np.nonzero(mask)
    count = len(notMasked[0])
    indexConverter = -np.ones(mask.shape,dtype = int)
    indexConverter[notMasked] = np.arange(0,count)



    upConverter = filters.correlate(indexConverter,[[0,1,0],[0,0,0],[0,0,0]],mode = "constant")*mask
    downConverter = filters.correlate(indexConverter,[[0,0,0],[0,0,0],[0,1,0]],mode = "constant")*mask
    leftConverter = filters.correlate(indexConverter,[[0,0,0],[1,0,0],[0,0,0]],mode = "constant")*mask
    rightConverter = filters.correlate(indexConverter,[[0,0,0],[0,0,1],[0,0,0]],mode = "constant")*mask

    newb = np.zeros(p.shape)
    newb += -1/2*filters.correlate(p,[[0,1,0],[0,1,0],[0,0,0]],mode = "constant") * hasup
    newb += 1/2*filters.correlate(p,[[0,0,0],[0,1,0],[0,1,0]],mode = "constant") * hasdown
    newb += 1/2*filters.correlate(q,[[0,0,0],[0,1,1],[0,0,0]],mode = "constant") * hasright
    newb += -1/2*filters.correlate(q,[[0,0,0],[1,1,0],[0,0,0]],mode = "constant") * hasleft
    
    # newb += 2*p*(hasup == 0)*mask
    # newb += -2*p*(hasdown == 0)*mask
    # newb += 2*q*(hasright == 0)*mask
    # newb += -2*q*(hasleft == 0)*mask

    
    A = lil_matrix((count+1, count+1))
    A = lil_matrix((count, count))

    # Force the first element to be 0
    #A[0,count] = 1

    X,Y = notMasked


    def checker(hasdir,dirConverter,hasopdir,opositeConverter):
        indexHasDir = hasdir[(X,Y)]
        wherehasdir = np.nonzero(indexHasDir)[0]
        indexDir = dirConverter[(X,Y)][wherehasdir]
        A[(indexDir,wherehasdir)] = A[(indexDir,wherehasdir)].toarray() - 1
        A[(wherehasdir,wherehasdir)] = A[(wherehasdir,wherehasdir)].toarray() + 1
        
        wherenothasdir = np.nonzero((indexHasDir == 0) * hasopdir[(X,Y)])[0]
        indexOpDir = opositeConverter[(X,Y)][wherenothasdir]
        A[(indexOpDir,wherenothasdir)] = A[(indexOpDir,wherenothasdir)].toarray() - 1
        A[(wherenothasdir,wherenothasdir)] = A[(wherenothasdir,wherenothasdir)].toarray() + 1
        


        
    checker(hasup,upConverter,hasdown,downConverter)
    checker(hasdown,downConverter,hasup,upConverter)
    checker(hasleft,leftConverter,hasright,rightConverter)
    checker(hasright,rightConverter,hasleft,leftConverter)



    nb = newb[mask != 0] #barr #(b)[mask != 0]
    
    
    return (csr_matrix(A),nb)


def PoissonSolverPS(normals,mask):
    """
    Solves the 2D case of the possion problem given unit normals
    """

    p = -normals[0]/normals[2]
    q = -normals[1]/normals[2]

    
    b = CentralDifference(p,q)

    A,b = PoissonFDEequationsWithVonNeumann(mask,p,q,b)

    
    z = np.zeros(mask.shape)
    vals = spsolve(A,b)
    #vals = (vals - np.mean(np.abs(vals)))
    z[np.where(mask > 0)] = vals
    z[mask == 0] = np.nan
    z = -z

    
    return (A,b,z)



def PhotometricStereoSolver(imgArr,mask,LightDirections,display = False,cheat = False):
    """

    """
    normals = PhotometricStereoNormals(imgArr,mask,LightDirections,display)
    if cheat:
        z = ps_utils_test.unbiased_integrate(normals[0],normals[1],normals[2],mask)[0]
        
    else:
        z = PoissonSolverPS(normals,mask)[2]
        
    if display:
        
        p = -normals[0]/normals[2]
        q = -normals[1]/normals[2]

        cd = CentralDifference(p,q)
    


        _,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(p)
        ax2.imshow(q)
        ax3.imshow(cd)
        plt.show()

        plt.imshow(z)
        plt.show()

        display_surface(z)

    return z

if dotest and True:

    # imgs, imgMask, S = imgImporter("./testbrick/","testbrick","Δ",0.005)
    # imgs, imgMask, S = imgImporter("./sphere/","testsphere","Δ",0.005,lightlim = 4)
    # imgs, imgMask, S = imgImporter("./testmonkey/","testmonkey","Δ",0.005,lightlim = 4)
    # zimgs = PhotometricStereoSolver(imgs,imgMask,S,display = True,cheat = False)
    # 
    ()
    
if (__name__ == "__main__" or dotest) and True:
    DataPath = "../../Specialprojekt/tilAnders/Beethoven.mat"

    Images, mask, S = read_data_file(DataPath)
    _,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(Images[:,:,0],cmap='gray')
    ax2.imshow(Images[:,:,1],cmap='gray')
    ax3.imshow(Images[:,:,2],cmap='gray')
    plt.show()
    z = PhotometricStereoSolver(Images,mask,S,display = True,cheat = False)

    # normals = PhotometricStereoNormals(Images,mask,S)


    # res = PoissonSolverPS(normals,mask)

    # reshaped = res[2]


    # p = -normals[0]/normals[2]
    # q = -normals[1]/normals[2]

    # cd = CentralDifference(p,q)


    # _,(ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.imshow(normals[0])
    # ax2.imshow(normals[1])
    # ax3.imshow(normals[2])
    # plt.show()

    # _,(ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.imshow(p)
    # ax2.imshow(q)
    # ax3.imshow(cd)
    # plt.show()
    # edgefinder = np.array([[0,1,0],[1,0,1],[0,1,0]])
    # plt.imshow(mask - (scipy.ndimage.filters.convolve(mask,edgefinder) == 4))

    # plt.imshow(reshaped)
    # plt.show()
    # z = PhotometricStereoSolver(Images,mask,S,display = True)
