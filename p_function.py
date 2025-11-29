import pandas as pd

DATA_FILE = "./Swissprot_Train_Validation_dataset.csv"


def parse_data(in_file):
    cellular_locations = [
        "Membrane",
        "Cytoplasm",
        "Nucleus",
        "Extracellular",
        "Cell membrane",
        "Mitochondrion",
        "Plastid",
        "Endoplasmic reticulum",
        "Lysosome/Vacuole",
        "Golgi apparatus",
        "Peroxisome",
    ]

    df = pd.read_csv(
        in_file,
        usecols=[
            "ACC",
            *cellular_locations,
            "Sequence",
        ],
    )

    binary_cols = [c for c in df.columns if c not in ("ACC", "Sequence")]
    df["location_count"] = df[binary_cols].sum(axis=1)

    # See how many have a unique location
    # num_one = (df["location_count"] == 1).sum()
    # print("Total entries", len(df))
    # print("Unique cellular location", num_one)

    df["location"] = df[cellular_locations].idxmax(axis=1)
    df.loc[df["location_count"] != 1, "location"] = None

    # filter the df
    df = df[df["location"].notna()]
    df = df[["ACC", "Sequence", "location"]]
    df = df.reset_index(drop=True)

    # See how many in each location
    # counts = df["location"].value_counts()
    # print(counts)

    df.to_csv("functional_annotations.csv", index=False)

    return df


if __name__ == "__main__":
    df = parse_data(DATA_FILE)
    print(df.head(100))
