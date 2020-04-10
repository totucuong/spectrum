import pytest
import pandas as pd

from spectrum.judge.lca import simpleLCA_EM as LCA_EM
from spectrum.judge.lca import simpleLCA_VI as LCA_VI


@pytest.fixture
def claims():
    claims = dict()
    claims['source_id'] = [0, 0, 1, 1, 2]
    claims['object_id'] = [0, 1, 1, 0, 2]
    claims['value'] = [0, 1, 3, 1, 1]
    return pd.DataFrame(data=claims)


@pytest.fixture
def claims2():
    claims = dict()
    claims['source_id'] = [0, 0, 1, 1, 2]
    claims['object_id'] = [0, 1, 1, 0, 2]
    claims['value'] = [0, 1, 0, 1, 1]
    return pd.DataFrame(data=claims)


def test_build_observation_matrix(claims):
    lca = LCA_EM(claims)
    assert lca.observation.shape == (3, 5)
    assert lca.observation[1][2] == 3
    assert lca.observation[1][0] == -1


def test_build_weight_matrix(claims):
    lca = LCA_EM(claims)
    assert lca.weight.shape == (3, 3)
    assert lca.weight[2][0] == 0
    assert lca.weight[2][2] == 1


def test_get_value(claims):
    lca = LCA_EM(claims)
    assert lca.get_value(1, 0) == 1
    assert lca.get_value(1, 1) == 3


def test_compute_responsibility(claims2):
    lca = LCA_EM(claims2)
    assert lca.compute_responsibility(1, 1) == 0.125


def test_compute_joint(claims2):
    lca = LCA_EM(claims2)
    joint_prob = lca.compute_joint(1)
    assert joint_prob.shape == (2, )
    assert joint_prob[1] == 0.0625


def test_e_step(claims2):
    lca = LCA_EM(claims2)
    lca.e_step()
    assert lca.posterior[1][1] == 0.5


def test_m_step(claims2):
    lca = LCA_EM(claims2)
    lca.e_step()
    lca.m_step()
    assert lca.theta_new[1] == 0.5


# Test of simpleLCA blax box bbvi
@pytest.fixture
def claims3():
    claims = dict()
    claims['source_id'] = [0, 0, 1, 1, 2, 4, 3, 5, 6, 7]
    claims['object_id'] = [0, 1, 1, 0, 2, 2, 2, 3, 3, 3]
    claims['value'] = [0, 1, 0, 1, 1, 0, 2, 0, 1, 2]
    claims = pd.DataFrame(data=claims)
    return claims.sample(claims.shape[0], replace=False)


def test_build_domsize_to_objects(claims3):
    lca = LCA_VI(claims3)
    assert lca.domsize_to_objects[3][1] == 3


def test_build_object_to_sources(claims3):
    lca = LCA_VI(claims3)
    object_to_sources = lca.build_object_to_sources()
    assert object_to_sources[2][1] == 3


@pytest.fixture
def claims4():
    claims = dict()
    claims['source_id'] = [0, 1, 0, 1, 2, 2, 1, 0]
    claims['object_id'] = [0, 0, 1, 1, 1, 2, 2, 2]
    claims['value'] = [0, 1, 2, 1, 0, 2, 1, 0]
    claims = pd.DataFrame(data=claims)
    return claims.sample(claims.shape[0], replace=False)


def test_compute_batch_idx(claims4):
    lca = LCA_VI(claims4)
    assert lca.to_batch_idx[2] == 1


def test_compute_observed_probs_VI(claims4):
    lca = LCA_VI(claims4)
    probs = lca.compute_observed_probs(2, 1)
    assert probs.shape == (3, 3)
    probs = probs.numpy()

    assert probs[0][1] == pytest.approx(0.33333334)
    assert probs[0][0] == pytest.approx(0.33333334)
    assert probs[0][2] == pytest.approx(0.33333334)


def test_build_claim_obs(claims4):
    lca = LCA_VI(claims4)
    probs = lca._build_claim_probs(0.5, 3, 1)
    probs = probs.numpy()
    assert probs[1] == 0.5
    assert probs[0] == 0.25
    assert probs[2] == 0.25