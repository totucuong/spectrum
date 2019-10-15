from spectrum.judge.lca import build_mask, build_observation

import pandas as pd

claims = dict()
claims['source_id'] = [0, 0, 1]
claims['object_id'] = [0, 1, 1]
claims['value'] = [0, 1, 0]
claims = pd.DataFrame(data=claims)

# def test_build_mask():
#     W = build_mask(claims)
#     assert W.shape == (2, 2)
#     assert W[0, 0] == 1
#     assert W[1, 0] == 0

# def test_build_observation_shape():
#     # claims_enc = claims.rename({'value': 'value_id'}, axis=1)
#     observation = build_observation(claims)
#     assert observation[0].shape == (2, 1)  # 2 sources 1 value for object 0
#     assert observation[1].shape == (2, 2)  # 2 sources 2 value for object 1

# def test_build_observation_content():
#     claims = dict()
#     claims['source_id'] = [0, 0, 1, 1]
#     claims['object_id'] = [0, 1, 1, 0]
#     claims['value'] = [0, 1, 0, 1]
#     claims = pd.DataFrame(data=claims)
#     observation = build_observation(claims)
#     assert observation[0][0, 0] == 1
#     assert observation[1][1, 0] == 1
