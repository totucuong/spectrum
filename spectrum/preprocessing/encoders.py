from sklearn.preprocessing import LabelEncoder


def transform(claims):
    """label encode value of claims
    
    Parameters
    ----------
    claims: panda.DataFrame
        a data frame [source_id, object_id, value] where source_id, and object_id is already label encoded.
    
    Returns
    -------
    claims_value_enc: pandas.DataFrame
        a data frame [source_id, object_id, value_id], everything is label encoded.
        
    le_dict: dict
        a dictionary of (object_id -> LabelEncoder)
    """
    df = claims.copy()
    group_by_source = df.groupby('object_id')
    le_dict = group_by_source.apply(lambda x: LabelEncoder().fit(x['value']))
    for g, index in group_by_source.groups.items():
        df.loc[index, 'value'] = le_dict[g].transform(df.loc[index, 'value'])
    return df, le_dict


def inverse_transform(claims_enc, le_dict):
    """inverse transform value of claims
    
    Parameters
    ----------
    claims_enc: panda.DataFrame
        a data frame [source_id, object_id, value] where value is already label encoded.
    
    Returns
    -------
    claims: pandas.DataFrame
        a data frame [source_id, object_id, value]
    """
    df = claims_enc.copy()
    group_by_source = df.groupby('object_id')
    for g, index in group_by_source.groups.items():
        df.loc[index, 'value'] = le_dict[g].inverse_transform(
            df.loc[index, 'value'])
    return df