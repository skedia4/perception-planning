import numpy as np
from sklearn.neighbors import KDTree

def fit_rigid(src, tgt):
    
    m = src.shape[1]

    centroid_src = np.mean(src, axis=0)
    centroid_tgt = np.mean(tgt, axis=0)
    
    tf_src = src - centroid_src
    tf_tgt = tgt - centroid_tgt

    H = np.dot(tf_src.T, tf_tgt)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    diag=np.diag([1,1,np.linalg.det(R)])
    R = np.matmul(Vt.T,np.dot(diag,U.T))


    t = centroid_tgt.T - np.dot(R,centroid_src.T)

    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def icp(A, B, init_pose=None, max_iter=20, tolerance=0.001):

    
    m = A.shape[1]

    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    tree = KDTree(dst[:m,:].T, leaf_size=2)

    for i in range(max_iter):

        distances, indices =tree.query(src[:m,:].T, k=1)  
        T,_,_ = fit_rigid(src[:m,:].T, dst[:m,indices.ravel()].T)

        src = np.dot(T, src)

        mean_error = np.mean(distances.ravel())
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    T,_,_ = fit_rigid(A, src[:m,:].T)

    return T, distances, i

