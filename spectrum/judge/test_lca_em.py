import pytest
import pandas as pd

from spectrum.judge.lca_em import LCA_EM


@pytest.fixture
def claims():
    claims = dict()
    claims['source_id'] = [0, 0, 1, 1]
    claims['object_id'] = [0, 1, 1, 0]
    claims['value'] = [0, 1, 3, 1]
    return pd.DataFrame(data=claims)


def test_build_observation_matrix(claims):
    lca = LCA_EM(claims)
    assert lca.observation[1][2] == 3
