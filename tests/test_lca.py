from spectrum.discovers.lca import build_mask

import pandas as pd

claims = dict()
claims['source_id'] = [0, 0, 1]
claims['object_id'] = [0, 1, 1]
claims['value'] = [0, 1, 0]
claims = pd.DataFrame(data=claims)


def test_build_mask():
    W = build_mask(claims)
    assert W.shape == (2, 2)
    assert W[0, 0] == 1
    assert W[1, 0] == 0
