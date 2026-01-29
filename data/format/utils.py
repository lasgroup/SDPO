from datasets import Value


def cast_large_strings(ds, columns):
    new_feats = ds.features.copy()
    for col in columns:
        if col in ds.column_names:
            new_feats[col] = Value("large_string")
    ds = ds.cast(new_feats)
    return ds
