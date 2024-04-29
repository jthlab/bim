import jax.numpy as jnp
from jax import lax
import jax
from jax.scipy.special import gammaln as LG
jax.config.update("jax_enable_x64", True)

from gmpy2 import mpq
import gmpy2
gmpy2.get_context().precision = 1000

from functools import partial, lru_cache
from dataclasses import dataclass
import scipy
import tskit
import re
import numpy as np

from bim.utils import tree_to_splits

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

@lru_cache(None)
def _nc2(N):
    return N * (N - 1) // 2

@lru_cache(None)
def _A(N):
    '''
    returns A_jk calculated as eq6 @ Polanski

    Parameters
    ----------
    N : int
        sample size.

    Returns
    -------
    A : ndarray

    '''
    A = np.zeros([N - 1, N - 1], dtype=object)
    for k in range(2, N + 1):
        for j in range(k, N + 1):
            A[k - 2, j - 2] = np.prod([mpq(_nc2(ell), _nc2(ell) - _nc2(j)) for ell in range(k, N + 1) if ell != j])
    A[N - 2, N - 2] = 1
    return A

@partial(jax.jit, static_argnums=(0,))
def Pnkb(N, beta):
    '''
    (Computation of mutant type size spectrum in each level) defn 12 in paper 

    Parameters
    ----------
    N : int
        sample size.
    beta : float (-\infnty, \infnty)
        beta-splitting parameter.

    Returns
    -------
    ndarray
        pnkb matrix.

    '''
    
    beta = jnp.exp(beta)
    n = jnp.arange(1, N+1)[:, None]
    k = jnp.arange(1, N+1)[None, :]
    P = jnp.where(k < n, 2*fs(n, k, beta), 0)
    P = P - jnp.identity(N)
    
    State = jnp.zeros(N)
    State = jax.ops.index_update(State, jax.ops.index[N-1], 1)
    # AttributeError: module 'jax.ops' has no attribute 'index_update'
    # Instead of ops.index_update(x, idx, vals) you should use x.at[idx].set(vals).
    # https://github.com/google/jax/issues/11706

    # State = State.at[N-1].set(1)
    lins = jnp.arange(N)
    State = State.reshape(1, N)
       
    mean_ling = jnp.concatenate([jnp.zeros(N - 1), jnp.ones(1)])
    def loop_body(mean_ling_im1, i):
        State = i * mean_ling_im1
        probs = lins * State # weight the probs with size (Durett forward split)
        probs /= jnp.sum(probs)
        State += jnp.matmul(probs, P)
        State = jax.ops.index_update(State, jax.ops.index[-i], 0)
        # State = State.at[-i].set(0)      
        return (State / (i + 1),) * 2

    _, mean_ling = lax.scan(loop_body, mean_ling, jnp.arange(1, N))
    
    return mean_ling[:,:-1]

@dataclass
class PConst:
    """Piecewise constant rate function. Computation requires gmpy2.
    This represents the function
        eta(x) = a_i, t_i <= t < t_{i+1}
    Args:
        a, b: arrays of shape [T] corresponding to a_i, b_i in the formula
            shown above. a must be positive, while be can be any number.
        t: array of shape [T] corresponding to t_i in the formula shown
            above; t[T] is implicitly taken to equal +infinity.
    We require that t[0] < t[1] < ... < t[T - 1], and that t[0] == b[T - 1] = 0.
    """
    a: np.ndarray
    t: np.ndarray   
    
    def __post_init__(self):
        self.a = np.array(self.a)
        self.t = np.array(self.t)
        if any([len(x.shape) != 1 for x in (self.a, self.t)]):
            raise ValueError("a and t should be one-dimensional arrays.")
        T = self.a.shape[0]
        if self.t.shape[0] != T:
            raise ValueError("a and t should have length T")
        if np.any(self.a <= 0):
            raise ValueError("a must be strictly positive.")
        if self.t[0] != 0.:
            raise ValueError("t[0] should equal 0.")
        if np.any(self.t[:-1] >= self.t[1:]):
            raise ValueError("t should be monotonically increasing.")
    @property
    def _ctx(self):
        import contextlib
        return contextlib.nullcontext()
    def __call__(self, u: np.ndarray):
        r"Evaluate eta(u)."
        a = self.a
        t = self.t
        j = np.searchsorted(t, u) - 1
        return a[j]
    def R(self, u: np.ndarray):
        r"Evaluate R(u) = \int_0^u eta(s) ds"
        a = self.a
        t = self.t
        dt = np.append(np.minimum(t[1:], u), u) - np.minimum(u, t)
        integrals = a * dt
        return integrals.sum()
    def exp_integral(self, c: float = 1.):
        r"""Compute the integral $\int_0^inf exp[-c * R(t)] dt$ for $R(t) = \int_0^s eta(s) ds$.
        Args:
            c: The constant multiplier of R(t) in the integral.
        Returns:
            The value of the integral.
        """
        # ET = \int_0^inf exp(-R(t))
        #    = \sum_{i=0}^{T - 1} \int_{t_i}^{t_{i + 1}} exp(-R(t))
        #    = \sum_{i=0}^{T - 1} \int_{t_i}^{t_{i + 1}} exp(-I_i) exp[-a_i (t - t_i)] / a
        #    = \sum_{i=0}^{T - 1} exp(-I_i) (1 - exp(-a_i * dt_i) / a_i
        #
        a = np.array([gmpy2.mpfr(float(aa)) for aa in self.a]) * gmpy2.mpz(c)
        t = np.array([gmpy2.mpfr(float(tt)) for tt in self.t])
        dt = np.diff(t)
        # to prevent nan's from infecting the gradients, we handle the last epoch separately.
        integrals = a[:-1] * dt
        z = a[0] * 0.
        I = np.concatenate([[z], np.cumsum(integrals)])
        ndt = len(dt)
        exp_integrals = gmpy2.exp(-I[-1]) / a[-1]
        exp_integrals += gmpy2.fsum([
            gmpy2.exp(-I[i]) * -gmpy2.expm1(-a[i] * dt[i]) / a[i]
            for i in range(ndt)
        ])
        return exp_integrals


class bSFS:
    '''
    Method to infer beta-splitting using site frequency spectrum

    Parameters
    ----------
    N : int
        sample size.    
    eta: utils.PConst
        piecewise constatn populaiton size object
    rho1 : float
        ridge coefficient.
    rho2 : float
        lasso coefficient.
    '''
    def __init__(self, N, eta = PConst(a = [1.], t = [0.]), rho1 = 0., rho2 = 0.):    
        
        self.N = N
        self.rho1 = rho1
        self.rho2 = rho2
        self.eta = eta        
        self.BETmm = self.get_BETmm() # BETMM
        
        @partial(jax.jit)
        def L(beta, SFS):
            pen1 = jnp.abs(beta)
            pen2 = jnp.power(beta,2)
            model = self.Ebl(beta)
            return jnp.reshape(-jnp.mean(SFS*jnp.log(model))+ pen1*rho1 + pen2*rho2, ())
        
        self.obj = jax.jit(jax.value_and_grad(L))  
        
        test = self.predict(2/np.arange(2, N+1))
        if not test.success:
            raise ValueError('A problem occured!')
        
    def get_BETmm(self):
        eta = self.eta
        N = self.N
        j = np.arange(2, N + 1)
        k = np.arange(2, N + 1)
        A = _A(N)
        B = (A * (j * (j - 1))[None, :] / (k - 1)[:, None])
        ETmm = [eta.exp_integral(j * (j - 1) // 2) for j in range(2, N + 1)]
        BETmm = B @ ETmm        
        return BETmm.astype('float')
    
    @partial(jax.jit, static_argnums=(0,))
    def Ebl(self, beta):
        N = self.N
        BETmm = self.BETmm
        pnkb = Pnkb(N, beta)
        model = jnp.matmul(pnkb.T, BETmm)
        return model/model.sum()
    
    def segsites_to_SFS(self, GM):
        '''
        Get SFS from Segregeting Sites

        Parameters
        ----------
        GM : ndarray
            L x N, L is number of sites and N is the sample size

        Returns
        -------
        ndarray
            SFS.

        '''
        return np.bincount(GM.sum(1), minlength = GM.shape[1])[1:]
        
    def predict(self, sfs):
        np.random.seed(int(sum(sfs)))
        x0 = np.random.randn(1)/100000
        #h0 = self.obj(x0, SFS)[0]
        out = scipy.optimize.minimize(self.obj, x0 = x0, 
                                      jac=True, method='SLSQP', 
                                      bounds = [[-15,10]],
                                      options = {'maxiter':50},
                                      args=sfs)

        #out = {'betahat':out.x[0],
        #       'h0':float(h0), 'h1':out.fun}
        #out.update(Theta(SFS))
        
        return out
    
class bTree:
    '''
    Method to infer beta-splitting using bifurcating trees

    Parameters
    ----------
    N : int
        sample size.    
    rho1 : float
        ridge coefficient.
    rho2 : float
        lasso coefficient.
    frac: fraction of the splits compared to overall splits 
        e.g frac = (0, 10) will only consider bottom 10 percent
        e.g frac = (90, 100) will only consider top 10 percent
    log_pdf: function
        It is the splitting distribution, takes n,k,beta.
        n: node size, k: children size, beta is the splitting parameter
    abeta: float
        aldous beta-splitting to solve polytomies default is 0
    '''
    def __init__(self, N, rho1 = 0., rho2 = 0., frac = None, log_pdf = logfs, abeta = 0.):    
        
        if frac is not None:
            N1 = N-1
            frac = np.round((N1*frac[0]/100, N1*frac[1]/100)).astype('int')
            self.start = frac[0]
            self.end = frac[1]
        else:
            self.start = 0
            self.end = N
        
        self.w = np.ones(N-1)

        def obj_bb(beta, n, k, w):
            abeta = jnp.abs(beta)
            beta = jnp.exp(beta)
            loglik = jnp.average(log_pdf(n, k, beta), weights = w)
            return jnp.reshape(-loglik + rho1*abeta + rho2*jnp.power(abeta, 2)/2, ())
        
        self.obj_bb = jax.jit(jax.value_and_grad(obj_bb))
        self.N = N
        self.rho1 = rho1
        self.rho2 = rho2
        self.cfinder = re.compile('\(\d+\,\d+\)') # find the merges from the newick tree
        self.success = None
        self.abeta = abeta
        
        test = self.predict(np.arange(2,N+1), np.arange(1, N))
        if not test.success:
            raise ValueError('A problem occured!')
        
    def tree_to_splits(self, Tree):
        '''
        Calculate size of the parent node and size of the left child for a given tree

        Parameters
        ----------
        Tree : tskit.trees.Tree

        Returns
        -------
        dict
            splits.

        '''
        return tree_to_splits(Tree, self.abeta)
        
    def newick_to_splits(self, newick):
        '''
        Calculate size of the parent node and size of the left child for a given tree

        Parameters
        ----------
        Tree : str
            A string that encodes a newick tree

        Returns
        -------
        dict
            splits.

        '''
        N = self.N
        newick = re.sub(':\d\.\d*', '', newick)
        mass = {str(i):1 for i in range(1, N+1)}
        n = np.zeros(N-1, dtype = 'int16')
        k = np.zeros(N-1, dtype = 'int16') 
        i = N + 1

        sisters = self.cfinder.finditer(newick)
        cnt = True
        while(cnt):
            j = 0
            while(1):

                try:
                    cout = next(sisters)
                except:
                    break

                start, end = cout.span()
                start -= j
                end -= j

                # record splits
                cnode = str(i)
                lc, rc = cout.group().split(',')
                lc, rc = lc[1:], rc[:-1]
                lm, rm = mass[lc], mass[rc]
                mass[cnode] = lm + rm
                cind = i-N-1
                n[cind] = mass[cnode]
                k[cind] = lm

                # update string
                newick = newick[:start]+cnode+newick[end:]

                i = i + 1
                j += end - start - len(cnode)

            if cind == (N-2):
                break

            sisters = self.cfinder.finditer(newick)

        return {'splits': (n, k)}
    
    def predict(self, n, k, w = None, x0 = 0.):
        '''
        Betahat for an inferred tree

        Parameters
        ----------
        n : ndarray
            number of sample the node subtends to.
        k : ndarray
            number of sample the right (or left) child subtends to.
        w : size N-1 arraay
            weights for weigted log-likelihood 
        Returns
        -------
        scipy 
            estimates.

        '''
        
        if w is None:
            w = self.w
        
        fr, to = self.start, self.end
        
        # h0 = self.obj_bb(0., n[fr:to], k[fr:to], w[fr:to])[0] # the null
        out = scipy.optimize.minimize(self.obj_bb, x0 = x0,
                                      jac=True, method='SLSQP',
                                      args=(n[fr:to], k[fr:to], w[fr:to]))
        
        self.lastout = out        
        
        return out
        
        # return {'betahat': out.x[0], 
        #         'h0': float(h0), 'h1': out.fun,
        #         'colless': Colless(n, k)}
    
    def split_predict(self, Tree, w = None, x0 = 0.):
        '''
        Splits and then predicts

        Parameters
        ----------
        Tree : tskit.trees.Tree or str (Newick representation)

        Returns
        -------
        scipy optimization out
        '''
        
        if isinstance(Tree, tskit.trees.Tree):
            ret = self.tree_to_splits(Tree)
        elif isinstance(Tree, str):
            ret = self.newick_to_splits(Tree)
        else:
            raise ValueError('Not an accepted type. Newick should be a `str` and Tree should be a `tskit.trees.Tree`. Not ', type(Tree))
                
        return self.predict(*ret['splits'], w = None, x0 = x0)
