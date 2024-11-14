#%%
import pymc as pm
from numpy.typing import ArrayLike
from itertools import product 
from pytensor import tensor as pt
from functools import partial

class MarginalizedNegativeBinomial(pm.CustomDist):

    def __init__(
        self,
        name: str,
        n_strains: int,
        coverages: pt.TensorVariable,
        dispersion: pt.TensorVariable,
        p: pt.TensorVariable,
        **kwargs
    ):
        super().__init__(
            name = name,
            dist_params = [coverages, dispersion, p],
            logp = partial(self.logp, n_strains=n_strains),
            ndims_params = [1, 0, 0],
            classname = "MarginalizedNegativeBinomial",
            **kwargs
        )

    def logp(
        self, 
        value: pt.TensorVariable,
        n_strains: int,
        coverages: pt.TensorVariable,
        dispersion: pt.TensorVariable,
        p: pt.TensorVariable
    ) -> pt.TensorVariable:
        
        # Get every possible vector of strain presence/absence
        q = pt.as_tensor([x for x in product([0,1], repeat=n_strains)])
        # Multiply q with coverages and sum to get the expected number of reads. Add a pseudocount
        n = pm.math.sum(q * coverages, axis=1) + 1

        # Define a negative binomial and get logp
        nb = pm.logp(pm.NegativeBinomial.dist(n=n, p=dispersion), value=value)
        # Multiply with prior for p (p is the same for all strains and qi are iid)
        p_prior = pm.math.log(pt.pow(p, pm.math.sum(q, axis=1)) * pt.pow((1-p), pm.math.sum(1-q, axis=1)))

        # Sum over qs
        ind_logp = nb+p_prior
        return pm.math.log(pm.math.sum(pm.math.exp(ind_logp)))
