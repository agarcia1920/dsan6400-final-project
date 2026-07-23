"""
Loads studio data from a CSV file.
"""

from pathlib import Path
from soulcycle_network.studio import Studio
import pandas as pd

# required columns for the studio data
REQUIRED_COLUMNS = {
    "studio_id",
    "studio_name",
    "official_region",
    "network_market",
    "market_tier",
    "local_ridership_cluster",
    "rides_per_wk_a",
    "bikes_per_ride_a",
    "rides_per_wk_b",
    "bikes_per_ride_b"
}

def load_studios(file_path: str | Path) -> dict[str, Studio]:
    """
    Load studios from a CSV file.
    """
    # check that the file path is a valid file
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix != ".csv":
        raise ValueError(f"File {file_path} is not a CSV file.")

    # read the studio data into a pandas dataframe
    studio_df = pd.read_csv(file_path)
    missing_cols = REQUIRED_COLUMNS - set(studio_df.columns)

    # check that the required columns are present
    if missing_cols:
        raise ValueError(
            f"File {file_path} is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    # initialize the studios dictionary
    studios: dict[str, Studio] = {}

    # iterate over the rows of the studio dataframe
    for idx, row in studio_df.iterrows():
        # try to parse the studio data
        try:
            weekly_class_count = int(row["rides_per_wk_a"]) + int(row["rides_per_wk_b"])
            max_class_capacity = int(row["bikes_per_ride_a"])
        
            # create a Studio object
            studio=Studio(
                studio_id=row["studio_id"],
                studio_name=row["studio_name"],
                official_region=row["official_region"],
                network_market=row["network_market"],
                market_tier=row["market_tier"],
                local_ridership_cluster=row["local_ridership_cluster"],
                weekly_class_count=weekly_class_count,
                class_capacity=max_class_capacity,
            )
        # if the studio data is invalid, raise an error
        except (ValueError, TypeError, KeyError) as e:
            # get the line number of the row
            line_num = idx + 2
            raise ValueError(f"Error parsing studio data for row {line_num}: {e}") from e
        
        # add the Studio object to the studios dictionary
        studios[row["studio_id"]] = studio

    return studios