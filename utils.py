from functools import partial, lru_cache
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln as LG
jax.config.update("jax_enable_x64", True)
import pandas as pd
import scipy
import numpy as np

from gmpy2 import mpq
import gmpy2
gmpy2.get_context().precision = 1000

@partial(jax.jit)
def logBB(n, k, beta, gamma):
    '''
    Computes log pdf of beta-binomial.

    Parameters
    ----------
    n : ndarray
        size of the parent node.
    k : ndarray
        size of the left (or right) child node.
    beta, gamma : ndarray
        imbalance parameter.    

    Returns
    -------
    float
        log pdf of beta-binomial.

    '''
    T1 = LG(n+1) - LG(k+1) - LG(n-k+1)
    T2 = LG(k+beta) + LG(n-k+gamma) - LG(n+beta+gamma)
    T3 = LG(beta+gamma) - LG(beta) - LG(gamma)
    return T1 + T2 + T3

@partial(jax.jit)
def BB(n, k, beta, gamma):
    '''
    Computes pdf of beta-binomial.

    Parameters
    ----------
    n : ndarray
        size of the parent node.
    k : ndarray
        size of the left (or right) child node.
    beta, gamma : ndarray
        imbalance parameter.    

    Returns
    -------
    float
        pdf of beta-binomial.

    '''
    return jnp.exp(logBB(n, k, beta, gamma))

@partial(jax.jit)
def logfr(n, k, beta):
    '''
    reflected beta-binomial
    log of the fr in the paper    
    
    Parameters
    ----------
    n : ndarray or int
        number of sample the node subtends to.
    k : ndarray or int (same size with n)
        number of sample the right (or left) child subtends to.
    beta : float (0, \infnty)
        beta-splitting parameter.

    Returns
    -------
    ndarray or float (same size with n)

    '''
    l = n//2+1
    g = 1

    normalizer = jnp.log(2)+jnp.log1p(-BB(l,0,beta,g)-BB(l,l,beta,g))
    normalizer = normalizer - (1-jnp.mod(n,2))*BB(l,l-1,beta,g)
    return jnp.where(k>=l, logBB(l,n-k,beta,g), logBB(l,k,beta,g))-normalizer

@partial(jax.jit)
def logfs(n, k, beta):
    '''
    log of fs in the paper, eq 3    
    
    Parameters
    ----------
    n : ndarray or int
        number of sample the node subtends to.
    k : ndarray or int (same size with n)
        number of sample the right (or left) child subtends to.
    beta : float (0, \infnty)
        beta-splitting parameter.
        
    Returns
    -------
    ndarray or float (same size with n)
    '''
    normalizer = jnp.log1p(-2*BB(n,0,beta,beta))
    return logBB(n, k, beta, beta) - normalizer

@partial(jax.jit)
def fs(n, k, beta):
    '''
    fs in the paper, eq 3    
    
    Parameters
    ----------
    n : ndarray or int
        number of sample the node subtends to.
    k : ndarray or int (same size with n)
        number of sample the right (or left) child subtends to.
    beta : float (0, \infnty)
        beta-splitting parameter.
        
    Returns
    -------
    ndarray or float (same size with n)
    '''
    normalizer = 1-2*BB(n,0,beta,beta)
    return BB(n, k, beta, beta)/normalizer

@partial(jax.jit)
def fr(n, k, beta):
    '''
    reflected beta-binomial
    log of the fr in the paper    
    
    Parameters
    ----------
    n : ndarray or int
        number of sample the node subtends to.
    k : ndarray or int (same size with n)
        number of sample the right (or left) child subtends to.
    beta : float (0, \infnty)
        beta-splitting parameter.

    Returns
    -------
    ndarray or float (same size with n)

    '''
    l = n//2+1
    g = 1.

    normalizer = 2*(1-BB(l,0,beta,g)-BB(l,l,beta,g))
    normalizer = normalizer - (1-jnp.mod(n,2))*BB(l,l-1,beta,g)
    return jnp.where(k>=l, BB(l,n-k,beta,g), BB(l,k,beta,g))/normalizer

def H(N):
    '''
    \sum_{i=1}^N \frac{1}{i}

    Parameters
    ----------
    N : int
        sample size.

    Returns
    -------
    float
        harmonic number.

    '''
    i = np.arange(1,N+1)
    return np.sum(1/i)

def H2(N):
    '''
    \sum_{i=1}^N \frac{1}{i^2}

    Parameters
    ----------
    N : int
        sample size.

    Returns
    -------
    float
        harmonic number.

    '''
    i = np.arange(1,N+1)
    return np.sum(1/i**2)

class Neutrality_Tests:   
    '''
    Population genetics statistics with the form of difference between two measures of genetic diversity.

    Parameters
    ----------
    N : int
        sample size.
    '''
    def Norm1(self):
        # Normalizer for TajD and FulD
        N = self.N
        an = self.an
        bn = self.bn
        
        ln = (N+1)/(3*(N-1)*an)-1/an**2

        T1 = 1/(an**2+bn)
        T2 = (2*(N**2+N+3))/(9*N*(N-1))
        T3 = (N-2)/(N*an)
        T4 = bn/an**2
        kn = T1*(T2-T3+T4)
        
        return ln, kn
            
    def Norm2(self):
        # Normalizer for FayH
        N = self.N
        an = self.an
        bn = self.bn
        bn1 = self.bn1
        
        ln = (N-2)/(6*(N-1)*an)

        T1 = (18*N**2)*(3*N+2)*bn1
        T2 = 88*N**3+9*N**2-13*N+6
        T3 = (9*N*(N-1)**2)*(an**2+bn)
        kn = (T1+T2)/T3  
        
        return ln, kn
            
    def Norm3(self):
        # Normalizer for ZngE
        N = self.N
        an = self.an
        bn = self.bn

        ln = N/(2*(N-1)*an)-1/an**2

        T1 = 1/(an**2+bn)
        T2 = bn/an**2
        T3 = 2*bn*(N/(N-1))**2
        T4 = 2*(N*bn-N+1)/((N-1)*an)
        T5 = (3*N+1)/(N-1)
        kn = T1*(T2+T3-T4-T5)   
        
        return ln, kn
            
    def Norm4(self):
        # Normalizer for FerL
        N = self.N
        an = self.an
        bn = self.bn
        bn1 = self.bn1

        ln = (1/an)*(1-1/an)

        T1 = 1/(an**2+bn)
        T2 = bn/an**2
        T3 = 2*(36*N**2*(2*N+1)*bn1-116*N**3+9*N**2+2*N-3)/(9*N*(N-1)**2)
        T4 = 4/(N*(N-1)*an)
        T5 = N**2*bn-((5*N+2)*(N-1))/4
        kn = T1*(T2+T3-T4*T5)
        
        return ln, kn
        
    def __init__(self, N):
        
        self.N = N
        
        self.an = H(N)
        self.bn = H2(N)
        self.bn1 = H2(N+1)
        
        i = np.arange(1,N)         
        self.w_pi = 2*i*(N-i)/(N*(N-1))
        self.w_W = np.ones(N-1)/self.an
        self.w_L = i/(N-1)
        self.w_H = 2*i**2/(N*(N-1))
        self.w_S = np.r_[1, np.zeros(N-2)]     
        
        self.Norms = [f() for f in [self.Norm1, self.Norm2, self.Norm3, self.Norm4]]
        
    def TajD(self, sfs, normalize = True):
        # sfs : ndarray sfs[i] = (i+1) copies in the sample
        # len(sfs) == N-1
        # Tajima (1989)
        t1, t2 = self.w_pi, self.w_W
        ret = (sfs*(t1-t2)).sum()
        
        if normalize:
            S = sfs.sum()
            ln, kn = self.Norms[0]
            ret /= np.sqrt(ln*S+kn*S*(S-1))
            
        return ret
    
    def FayH(self, sfs, normalize = True):
        # sfs : ndarray sfs[i] = (i+1) copies in the sample
        # len(sfs) == N-1
        # Fay and Wu (2000)
        t1, t2 = self.w_pi, self.w_H
        ret = (sfs*(t1-t2)).sum()
        
        if normalize:
            S = sfs.sum()
            ln, kn = self.Norms[1]
            ret /= np.sqrt(ln*S+kn*S*(S-1))
            
        return ret
    
    def ZngE(self, sfs, normalize = True):
        # sfs : ndarray sfs[i] = (i+1) copies in the sample
        # len(sfs) == N-1
        # Zeng et al. (2006)
        t1, t2 = self.w_L, self.w_W
        ret = (sfs*(t1-t2)).sum()
        
        if normalize:
            S = sfs.sum()
            ln, kn = self.Norms[2]
            ret /= np.sqrt(ln*S+kn*S*(S-1))
            
        return ret
    
    def FerL(self, sfs, normalize = True):
        # sfs : ndarray sfs[i] = (i+1) copies in the sample
        # len(sfs) == N-1
        # Feretti et al. (2017)
        t1, t2 = self.w_W, self.w_H 
        ret = (sfs*(t1-t2)).sum()
        
        if normalize:
            S = sfs.sum()
            ln, kn = self.Norms[3]
            ret /= np.sqrt(ln*S+kn*S*(S-1))
            
        return ret
    
    def FulD(self, sfs, normalize = True):
        # sfs : ndarray sfs[i] = (i+1) copies in the sample
        # len(sfs) == N-1
        # Fu and Li (1993)
        t1, t2 = self.w_S, self.w_W
        ret = (sfs*(t1-t2)).sum()
        
        if normalize:
            S = sfs.sum()
            ln, kn = self.Norms[0]
            ret /= np.sqrt(ln*S+kn*S*(S-1))
            
        return ret

def Colless(n, k):
    '''
    Compute colless statistic

    Parameters
    ----------
    n : ndarray
        number of sample the node subtends to.
    k : ndarray
        number of sample the right (or left) child subtends to.

    Returns
    -------
    float
        colless statistic

    '''
    N = len(n) + 1
    return 2*np.abs(n-2*k).sum()/(N-1)/(N-2)

def Omega(n, k, top = 3): 
    '''
    Compute Omega statistic (Li & Wiele 2013)

    Parameters
    ----------
    n : ndarray
        number of sample the node subtends to.
    k : ndarray
        number of sample the right (or left) child subtends to.
    top : int
          number of nodes from the top of the tree to include in the statistic
    
    Returns
    -------
    float
        Returns T_{top} in equation 14 in the paper

    '''
    n, k = n[-top:], k[-top:]
    omega = np.min(np.array((k, n-k)), 0)
    omega = 2*omega/n - 0.5
    omega = np.sqrt(12/len(n))*omega.sum()
    return omega
   
@lru_cache(None)
def _W(N):
    '''
    Parameters
    ----------
    N : int
        sample size.

    Returns
    -------
    W : ndarray
        W matrix as calculated as eq 13:15 @ Polanski 2013.

    '''
    W = np.zeros(
        [N - 1, N - 1], dtype=object
    )  # indices are [b, j] offset by 1 and 2 respectively
    W[:, 2 - 2] = mpq(6, N + 1)
    b = list(range(1, N))
    W[:, 3 - 2] = np.array([mpq(30 * (N - 2 * bb), (N + 1) * (N + 2)) for bb in b])
    for j in range(2, N - 1):
        A = mpq(-(1 + j) * (3 + 2 * j) * (N - j), j * (2 * j - 1) * (N + j + 1))
        B = np.array([mpq((3 + 2 * j) * (N - 2 * bb), j * (N + j + 1)) for bb in b])
        W[:, j + 2 - 2] = A * W[:, j - 2] + B * W[:, j + 1 - 2]
    return W

class InferEta:
    '''
    Object for modeling populaiton size histories as a piecewise constant function 
    '''

    def __init__(self, N, t, a1 = 0, a2 = 0, ar = 0):
        import jax 
        global jax
        import jax.numpy as jnp
        self.jnp = jnp
        self.N = N
        t = jnp.array(t)
        self.t = t
        self.a1 = a1
        self.a2 = a2
        self.ar = ar
        self.h = H(N-1)   
        
        m = len(t)
        self.m = m
        D = (jnp.eye(m, k=0) - jnp.eye(m, k=-1))
        A = jnp.eye(m)
        A = A.at[0, 0].set(0)
        self.D1 = A @ D  # 1st difference matrix        
        
        yref = jnp.ones(self.m)
        self.yref = yref
        
        if self.a1>0:
            l1 = lambda logy: jnp.abs(self.D1 @ logy).sum()
        else:
            l1 = lambda logy: 0

        if self.a2>0:
            l2 = lambda logy: ((self.D1 @ logy) ** 2).sum()
        else:
            l2 = lambda logy: 0

        if self.ar>0:
            logyref = jnp.log(yref)
            Gamma = jnp.eye(self.m)
            def lr(logy):
                X = logy.flatten()-logyref.flatten()
                return X.T @ Gamma @ X
        else:
            lr = lambda logy: 0
    
        get_esfs = self.get_esfs
        
        @partial(jax.jit)
        def f(A, SFS):
            model = get_esfs(A)
            model = model/model.sum()
            loss1v = -jnp.mean(SFS*jnp.log(model))

            logy = jnp.log(A).reshape(m, 1)
            loss = loss1v + a1*l1(logy) + .5*a2*l2(logy) + .5*ar*lr(logy)
            return loss.flatten()[0]
        
        self.jvgf = jax.jit(jax.value_and_grad(f))
        
    
    #@partial(jax.jit, static_argnums=(0, ))
    def exp_integral(self, a, x, j):
        jnp = self.jnp
        t = self.t
        c = j * (j - 1) // 2
        a = a * c
        dt = jnp.diff(t)
        integrals = a[:-1] * dt
        z = jnp.array([a[0] * 0.])
        I = jnp.concatenate((z, jnp.cumsum(integrals)))
        exp_integrals = jnp.concatenate([
            jnp.exp(-I[:-1]) * -jnp.expm1(-a[:-1] * dt) / a[:-1],
            jnp.exp(-I[-1:]) / a[-1:]
        ])
        return (0., exp_integrals.sum())

    #@partial(jax.jit, static_argnums=(0, ))
    def get_esfs(self, A):
        jnp = self.jnp
        N = self.N
        from jax import lax
        
        W = _W(N)
        a = 1/jnp.array(A)

        f = lambda x, j: self.exp_integral(a, x, j)

        x = jnp.array(0.)
        _, ETmm = lax.scan(f, x, jnp.arange(2, N+1))

        return W.astype('float')@ETmm

    def predict(self, SFS, yref = None, maxiter = 200):
        '''
        Infer piecewise constant population history

        Parameters
        ----------
        SFS : ndarray
            SFS[i] = (i+1) copies in the sample. len(SFS) == N-1

        Returns
        -------
        scipy optimization out
        '''

        out = scipy.optimize.minimize(self.jvgf, x0 = self.yref, 
                                      bounds = self.m*[[1, 1e10]],
                                      jac=True, method='SLSQP',
                                      args = SFS,
                                      options={'maxiter': maxiter, 
                                               'ftol': 1e-20})
        
        return out

def slidingSFS(ts, wsz = None, tsz = None, ssz = None):
    '''
    Get site frequency spectrum by sliding windows. If user provides wsz but not
    wsz, then ssz = wsz

    Parameters
    ----------
    ts : tskit.TreeSequence
        A tree seqeunce.
    wsz : int, optional
        Window size. The default is None.
    tsz : int, optional
        Tree-sequence size. The default is None.
    ssz : TYPE, optional
        Stride size. The default is None.

    Returns
    -------
    SFS : list of ndarray
        Site frequency spectrum array.
    df : DataFrame
        Data frame of start end end points of sfs
    starts : list of ndarray
        starts[i][j] is the starting genomic position for SFS[i][j].
    ends : list of ndarray
        ends[i][j] is the ending genomic position for SFS[i][j].

    '''    

    L = ts.get_sequence_length()
    trees = ts.trees()
    
    nones = sum([i is None for i in [wsz, tsz]])
    
    if  nones == 0:
        raise ValueError('you should specift either window size (wsz) or tree-seq size (tsz), not both')
    elif nones == 2:    
        sfs = ts.allele_frequency_spectrum(windows=[0, L], 
                                           span_normalise=False, 
                                           polarised=True)[:, 1:-1].astype('int')
        df = pd.DataFrame({'start':[0], 'end':[L]})
        starts = [np.array([0])]
        ends = [np.array([L])]
        
    else:
        if tsz:
            bps = []
            for Tree in trees:
                start, end = Tree.interval
                bps.append(start)

            bps.append(end)
            bps = np.array(bps)

            wsz = tsz

        elif wsz:
            bps = np.arange(0, int(L)+1)

        if not ssz:
            ssz = wsz

        step = wsz//ssz
        if step != wsz/ssz:
            raise ValueError('<window size>/<stride size> should be an integer')

        win = np.arange(0, len(bps), ssz)
        win[-1] = len(bps) - 1
        win = bps[win]
        sfsp = ts.allele_frequency_spectrum(windows=win, 
                                            span_normalise=False, 
                                            polarised=True)[:, 1:-1].astype('int')


        sfs = np.zeros((sfsp.shape[0]-step+1, sfsp.shape[1]), dtype = 'int')

        starts = []
        ends = []
        for i in range(sfs.shape[0]):
            f = i
            t = i+step
            sfs[i,:] = sfsp[f:t, :].sum(0)

            starts.append(win[f])
            ends.append(win[t])

        df = pd.DataFrame({'start':starts, 'end':ends})

        starts = [np.array(starts)[np.arange(i, len(starts), step, dtype = 'int')] for i in range(step)]
        ends = [np.array(ends)[np.arange(i, len(ends), step, dtype = 'int')] for i in range(step)]

    return sfs, df, starts, ends
    

def intersect_with_weights(bs, be, cps, cpe, y):
    '''
    Returns intersection of these two lines 
    (bs[i], be[i]) and (cps[k], cpe[k]). 
    (bs[i], be[i]) \cap (cps[k], cpe[k]) is the weights 
    and y[i] is the value for the observation

    Parameters
    ----------
    bs : ndarray
        bs[i] start of the y[i].
    be : ndarray
        be[i] end of the y[i].
    cps : ndarray
        cps[i] start of the region.
    cpe : ndarray
        cpe[i] end of the region.
    y : ndarray
        scan statistic.

    Returns
    -------
    pandas.DataFrame

    '''

    cplen = cpe-cps

    n = len(bs)
    m = len(cps)
    i = 0
    k = 0
    dx_new = []
    while((i<n)&(k<m)):
        if (bs[i]>=cps[k]) and (be[i]<=cpe[k]): #all tree inside the region
            w = be[i] - bs[i]
            dx_new.append({'start': cps[k], 'end': cpe[k], 'val': y[i], 'w': w/cplen[k], 
                           'val_start': bs[i], 'val_end': be[i]})
            i += 1
        elif (bs[i]<cps[k]) and (be[i]<=cpe[k]) and (be[i]>cps[k]): #left intersect with the region
            w = be[i] - cps[k]
            dx_new.append({'start': cps[k], 'end': cpe[k], 'val': y[i], 'w': w/cplen[k], 
                           'val_start': bs[i], 'val_end': be[i]})
            i += 1        
        elif (bs[i]>=cps[k]) and (be[i]>cpe[k]) and (bs[i]<=cpe[k]): #right intersect with the region
            w = cpe[k] - bs[i]
            dx_new.append({'start': cps[k], 'end': cpe[k], 'val': y[i], 'w': w/cplen[k], 
                           'val_start': bs[i], 'val_end': be[i]})
            k += 1
        elif (bs[i]>=cpe[k]): # right side of the region
            k += 1
        elif (be[i]<=cps[k]): # left side of the region
            i = i + 1
        elif (bs[i]<=cps[k]) and (be[i]>cpe[k]): # region is inside the tree
            w = cplen[k]
            dx_new.append({'start': cps[k], 'end': cpe[k], 'val': y[i], 'w': w/cplen[k], 
                           'val_start': bs[i], 'val_end': be[i]})
            k += 1
        else: 
            print({'start': cps[k], 'end': cpe[k], 'val': y[i], 'w': w/cplen[k], 
                   'val_start': bs[i], 'val_end': be[i]})
            raise Exception('IDK')
    
    return dx_new  

@lru_cache(None)
def get_ps(pm, abeta):
    beta = abeta/2+1
    vals = np.arange(1, pm)
    ps = fs(pm, vals, beta)
    return ps

def rsplit(masses, abeta, polsplits):
    '''
    masses is the list of number of samples subtend from each children
    abeta is the aldous' beta for the split (default is 0)
    returns mass of the left and right child
    '''
    
    nchildren = len(masses)
    
    if nchildren == 1:
        return None
    else:
        ps = get_ps(nchildren, abeta)
        c = np.random.choice(range(1, nchildren), p = ps)
        lc, rc = masses[:c], masses[c:]
        polsplits['n'].append(sum(masses))
        polsplits['k'].append(sum(lc))
        return rsplit(lc, abeta, polsplits), rsplit(rc, abeta, polsplits)
    
def split_polytomy(masses, abeta = 0):
    '''
    it splits the mass in the polytomy by aldous's beta
    '''
    polsplits = {'n':[], 'k':[]}
    rsplit(masses, abeta, polsplits)
    return polsplits

def tree_to_splits(Tree, abeta = 0):
    '''
    Calculate size of the parent node and size of the left child for a given tree

    Parameters
    ----------
    Tree : tskit.trees.Tree
    abeta: Aldous' beta (-2, infinity) default is 0

    Returns
    -------
    dict
        splits.

    '''
    
    N = Tree.num_samples()
    n = np.zeros(N-1, dtype = 'int16')
    k = np.zeros(N-1, dtype = 'int16') 

    np.random.seed(1)
    np.random.seed(int(Tree.get_index()*Tree.get_length())%(2**32-1))

    mass = {}
    ind = 0
    for i in Tree.nodes(order = 'timeasc'):
        children = list(Tree.children(i))
        lenc = len(children)

        if lenc == 0: # leaf
            mass[i] = 1 
        elif lenc == 2: #bifurcating
            rm = mass[children[0]]
            lm = mass[children[1]]                
            pm = rm+lm

            mass[i] = pm

            k[ind] = rm
            n[ind] = pm
            ind += 1  
        else: #polytomy
            masses = [mass[i] for i in children] # sample nodes of each children
            pm = sum(masses) # sample nodes of the parent
            mass[i] = pm

            sp = split_polytomy(masses, abeta) # it returns the random split parent-chilren masses 

            start = ind
            end = ind+lenc-1
            k[start:end] = sp['k']
            n[start:end] = sp['n']

            ind = end

    return {'splits': (n, k)}
    
def segsites_to_trees(GM, positions = None, seqlen = None):
    '''
    Creates a tskit.trees object from genotype matrix. Refer to tsinfer
    documentation for a more complete spesification

    Parameters
    ----------
    GM : ndarray
        L x N, L is number of sites and N is the sample size
    positions : ndarray, optional
        Positions for the sites. Should have length L. The default is None.

    Returns
    -------
    None.

    '''
    import tsinfer
    
    
    L = GM.shape[0]
    if positions is None:
        seqlen = L
        positions = np.arange(L)

    with tsinfer.SampleData(sequence_length=seqlen) as sample_data:
        for i in range(L):
            sample_data.add_site(positions[i], GM[i]) 
    
    return tsinfer.infer(sample_data).simplify()