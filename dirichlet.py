import numpy as np

def samples_per_class_dirichlet(y,c_clients,alpha,n=None,debug=False):
    """
    Returns the number of samples the cth client must sample from each kth class
    according to dirichlet sampling with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    To guarantee that every 0 < alpha <= 1 can be met the number of samples each
    client/segment can sample is set to the number of labels with the minimum
    frequency in y ('n_max_all_alphas' in the code).
    
    maximize_n indicates whether or not to try to use all available labels of y
    (respecting n_max_all_alphas).

    Parameters
    ----------
    y : numpy array
        The numpy array of labels to be partitioned, assumed to be of integers 0 to
        # of classes -1.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the proportion of elements of y that the i-th client must sample.
        Therefore if must be true that 0 <= n[i] <= 1 and sum(n) = 1.

    debug : boolean, optional
        Whether to perform extra checks (which can be slow) 
    
    Returns
    -------
    A numpy array of shape(c,k) matrix where A[i,j] denotes
    the amount of instances of class j the client i must draw.

    """
    
    _,counts_y=np.unique(y,return_counts=True)

    # Number of classes (i.e. # of unique elements in y)
    k=counts_y.size

    # Max n such that all alphas can be guaranteed
    n_max_all_alphas=counts_y.min()*k

    # Sample from de Dirichlet distribution
    proportions=np.random.dirichlet(alpha*np.ones(k),c_clients)

    if n is None:
        # Every client gets the same proportion of samples
        n=n_max_all_alphas

        # Multiply proportions
        out=(proportions*(n/c_clients))
    
    else:
        # Sanity checks on n
        assert isinstance(n,np.ndarray) or isinstance(n,list)
        assert sum(n)==1 and len(n)==c_clients
        
        # Each client gets their proportion of n_max_all_alphas
        out=proportions
        for c in range(c_clients):
            out[c,:]*=(n[c]*n_max_all_alphas)

    # Cast output to integer - truncation produces errors
    out=out.astype('int')

    # Correct errors produced by truncation
    missing_by_class=counts_y.min()-out.sum(axis=0)
    while (missing_by_class!=0).any():
        
        # Correct missing/surplus
        for c,missed_c in enumerate(missing_by_class):
            p=out[:,c]/out[:,c].sum() if out[:,c].sum()>0 else None
            where_to_add=np.random.choice(c_clients,size=abs(missed_c),p=p)
            np.add.at(out[:,c],where_to_add,1 if missed_c>0 else -1)

        out=np.abs(out)
        missing_by_class=counts_y.min()-out.sum(axis=0)

    if debug:
        # Perform expensive checks
        assert (out >= 0).all()
        assert out.sum()==n_max_all_alphas
        assert all(out.sum(axis=0)==counts_y.min())
        
    return out
    
def dirichlet_partition(y,c_clients,alpha,n=None,debug=False):
    """
    Randomly partitions an array of labels y into a # c_clients of clients
    according to Dirichelet sampling with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    To guarantee that every 0 < alpha <= 1 can be met the number of samples each
    client/segment can sample is set to the number of labels with the minimum
    frequency in y ('n_max_all_alphas' in the code).

    Calls samples_per_class_dirichlet to determine how y will be distributed.

    alpha --> 0 implies very uneven sampling while alpha --> inf approaches uniform sampling.  

    Parameters
    ----------
    y : numpy array
        The numpy array of labels to be partitioned, assumed to be of integers 0 to
        # of classes -1.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the number of elements of y that the i-th client must sample.

    debug : boolean, optional
        Whether to perform extra checks (which can be slow) 
        
    Returns
    -------
    The partition as a dictionary: client id (int) -> array of indices (np.array).

    """

    # The number of classes if y is assumed to be pandas' categorical codes.
    k_classes=y.max()+1

    # Given how many examples each client must sample from each class
    how_many=samples_per_class_dirichlet(y,c_clients,alpha,n,debug)

    # Find indices for each class and shuffle them
    wheres={}
    for l in range(k_classes):
        w=np.where(y==l)[0]
        np.random.shuffle(w)
        wheres[l]=list(w)

    # Client -> list of indices
    partition={c:[] for c in range(c_clients)}

    # For every class
    for k in range(k_classes):
        # We distribute the corresponding indices to the clients
        prev=0
        for i,ni in enumerate(how_many[:,k]):
            partition[i].extend(wheres[k][prev:prev+ni])
            prev+=ni
    
    if debug:
        # Check if partition matches with how_many
        actual=np.zeros(shape=how_many.shape,dtype='int')
        for c,ixs in partition.items():
            temp=np.zeros(shape=how_many.shape[1])
            classes,counts=np.unique(y[ixs],return_counts=True)
            temp[classes]+=counts
            actual[c,:]=temp

        assert np.array_equal(actual,how_many)


    return partition


if __name__=='__main__':
    # Test
    while True:
        size=np.random.randint(low=100,high=10000)
        k_classes=np.random.randint(10)
        alpha=np.random.random()
        alpha_p=np.random.random()
        p=np.random.dirichlet(alpha_p*np.ones(k_classes))
        y=np.random.choice(k_classes,size=size,p=p)
        c_clients=np.random.randint(15)
        #print(y)
        partition=dirichlet_partition(y,c_clients=c_clients,alpha=alpha,debug=True)