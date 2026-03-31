for dftemp in [traindf, testdf]:
    if "Gender" in dftemp.columns:
        dftemp["Gender"] = dftemp["Gender"].replace("Fe Male", "Female")

    cols_to_drop = ["Unnamed: 0", "CustomerID", "__index_level_0__"]
    existing_cols = [col for col in cols_to_drop if col in dftemp.columns]

    if existing_cols:
        dftemp.drop(columns=existing_cols, inplace=True)
