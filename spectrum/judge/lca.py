import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints


def lca_model(claims):
    """Build a Latent Credibility Analysis (LCA).
   
    A LCA model represents a joint distribution p(Y, H, X), where
    Y represents hidden truth rvs, H represents data source honesty, and X
    represents observation.

    Concretely, let's assume that we have M objects, S data sources, and our
    observation will be the mask matrix W, `mask`, (see build_mask()), and
    observation matrix `observation` (see build_observation()).

    With this context, we have:
        p(Y, H, X) = product_{m=1,..,M}[p(y_m, H, X)], where
        p(y_m, H, X) = p(y_m)product_{s in S_m}[p(b_sm|y_m,s)p(s)],
        where S_m are the set of sources that make claims about an object m.

    Parameters
    ----------
    claims: pd.DataFrame
        a data frame that has columns [source_id, object_id, value]

    """
    problem_sizes = claims.nunique()
    n_sources = problem_sizes['source_id']
    n_objects = problem_sizes['object_id']
    domain_size = claims.groupby('object_id').max()['value'] + 1
    # create honest rv, H_s, for each sources
    honest = []
    for s in pyro.plate(name='sources', size=n_sources):
        honest.append(
            pyro.sample(
                f's_{s}',
                dist.Bernoulli(
                    probs=pyro.param(f'theta_s_{s}',
                                     init_tensor=_draw_probs(),
                                     constraint=constraints.simplex))))

    # creat hidden truth rv for each object m
    hidden_truth = []
    for m in pyro.plate(name='objects', size=n_objects):
        hidden_truth.append(
            pyro.sample(
                f'y_{m}',
                dist.Categorical(
                    probs=pyro.param(f'theta_m_{m}',
                                     init_tensor=torch.ones((
                                         domain_size[m], )),
                                     constraint=constraints.simplex))))

    for c in pyro.plate(name='claims', size=len(claims.index)):
        m = claims.iloc[c]['object_id']
        s = claims.iloc[c]['source_id']
        y_m = hidden_truth[m]
        probs = _build_obj_probs_from_src_honest(pyro.param(f'theta_s_{s}'),
                                                 domain_size[m], y_m)
        pyro.sample(f'b_{s}_{m}', dist.Categorical(probs=probs))


def _build_obj_probs_from_src_honest(src_prob, obj_domain_size, truth):
    obj_probs = torch.ones((obj_domain_size, ))
    obj_probs *= (1 - src_prob) / (obj_domain_size - 1)
    obj_probs[truth] = src_prob
    return obj_probs


# def _build_obj_logits_from_src_honest(*args):
#     return torch.log(_build_obj_probs_from_src_honest(*args))


def lca_guide(claims):
    """Build a guide for lca_model.

    A guide is an approximation of the real posterior distribution p(z|D), 
    where z represents hidden variables and D is a training dataset.
    A guide is needed to perform variational inference.
    
    Parameters
    ----------
    claims: pd.DataFrame
        a data frame that has columns [source_id, object_id, value]
    """
    max_ids = claims.max()
    n_sources = max_ids['source_id'] + 1
    n_objects = max_ids['object_id'] + 1
    domain_size = claims.groupby('object_id').max()['value'] + 1
    for s in pyro.plate('sources', size=n_sources):
        # honest source rv
        pyro.sample(
            f's_{s}',
            dist.Bernoulli(probs=pyro.param(f'beta_s_{s}',
                                            init_tensor=_draw_probs(),
                                            constraint=constraints.simplex)))

    for m in pyro.plate('objects', size=n_objects):
        # hidden truth
        m_dist = 1 / domain_size[m] * torch.ones((domain_size[m], ))
        probs_m = pyro.param(f'beta_m_{m}',
                             init_tensor=m_dist,
                             constraint=constraints.simplex)
        pyro.sample(f'y_{m}', dist.Categorical(probs=probs_m))


def make_observation_mapper(claims):
    """Make a dictionary of observation.

    Parameters
    ----------
    claims: pd.DataFrame
       
    Returns
    -------
    observation_mapper: dict
        an dictionary that map rv to their observed value
    """
    observation_mapper = dict()

    for _, c in claims.iterrows():
        m = c['object_id']
        s = c['source_id']
        observation_mapper[f'b_{s}_{m}'] = torch.tensor(c['value'])
    return observation_mapper


def bvi(model, guide, claims, learning_rate=1e-5, num_samples=1):
    """perform blackbox mean field variational inference on simpleLCA.
    
    This methods take a simpleLCA model as input and perform blackbox variational
    inference, and returns a list of posterior distributions of hidden truth and source
    reliability. 
    
    Concretely, if s is a source then posterior(s) is the probability of s being honest.
    And if o is an object, or more correctly, is a random variable that has the support as
    the domain of an object, then posterior(o) is the distribution over these support.
    The underlying truth value of an object could be computed as the mode of this
    distribution.
    """
    data = make_observation_mapper(claims)
    conditioned_lca = pyro.condition(lca_model, data=data)
    pyro.clear_param_store()  # is it needed?
    svi = pyro.infer.SVI(model=conditioned_lca,
                         guide=lca_guide,
                         optim=pyro.optim.Adam({
                             "lr": learning_rate,
                             "betas": (0.90, 0.999)
                         }),
                         loss=pyro.infer.TraceGraph_ELBO(),
                         num_samples=num_samples)
    return svi


def fit(svi, claims, epochs=10):
    losses = []
    for t in range(epochs):
        cur_loss = svi.step(claims)
        losses.append(cur_loss)
        print(f'current loss - {cur_loss}')
    return losses


# def _draw_logits():
#     return torch.log(_draw_probs())


def _draw_probs():
    return torch.distributions.Dirichlet(torch.tensor([0.5, 0.5])).sample()[0]


def discover_trusted_source(posteriors, reliability_threshold=0.8):
    """Compute a list of trusted sources given a threshold of their relability

    Parameters
    ----------
    posteriors: dict
        a dictionary rv_name->posterior dist

    reliability_threshold: float
        if a source has reliability > reliability_threshold then it will be included
        in the result

    Returns
    -------
    trusted_sources: list
        a list of trusted sources id
    """
    result = [
        int(k.split('_')[2]) for k, v in posteriors.items()
        if k.startswith('beta_s') and torch.exp(v) > reliability_threshold
    ]
    return result


def discover_truths(posteriors):
    object_id = []
    value = []
    for k, v in posteriors.items():
        if k.startswith('beta_m'):
            object_id.append(int(k.split('_')[2]))
            value.append(int(torch.argmax(torch.exp(v)).numpy()))
    return pd.DataFrame(data={'object_id': object_id, 'value': value})


def main():
    """We use the following example, taken from LCA paper, to test out
    all functions in this examples:

    2 objects: m1='obama's birth place', m2='obama's birthdate'.
    1 sources:  s1='John'.
    s1 asserts about m1: c1 = 'Obama was born in Haiwai'
    
    Suppose we have other assertions made by different sources about
    m1: 
        c2="Obama was born in Indonesia",
        c3="Obama was born in Kenya".

    Let's denote dom(m1) as m1's domain.

    We encode datasets as follows.

    sources = {0, 1, ... S-1}, S is the total number of sources
    object = {0, 1, ..., M-1}, M is the total number of objects
    for each object o=0,..,M-1, we encode dom(o) as {0,..., |dom(o)|-1}
    
    For "john", we encode him as s=0 and for object o, we have vector
    [1, 0, 0] to represent the fact that he asserts 'Obama was born
    in Hawai'.

    What about other objects that John does not make any assertions about?
    We can apply a mask W = [w_so] where s indices sources and o indices
    objects. w_so =1 if s makes assertion about o, otherwise 0.

    """
    pass


if __name__ == "__main__":
    main()
