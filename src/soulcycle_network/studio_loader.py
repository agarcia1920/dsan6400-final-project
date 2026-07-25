# Loads studio data from a CSV file.
# The studios.csv file has one row per studio in the SoulCycle network.

from pathlib import Path
from soulcycle_network.studio import Studio
import pandas as pd

#required columns for the studio data
REQUIRED_COLUMNS = {"studio_id", "studio_name", "official_region", "network_market", "market_tier", "local_ridership_cluster", "rides_per_wk_a", "bikes_per_ride_a", "rides_per_wk_b", "bikes_per_ride_b"}

def load_studios(file_path: str | Path) -> dict[str, Studio]:
    #load studios from a CSV file and return them as a dictionary keyed by studio_id
    file_path = Path(file_path)

    #make sure the file actually exists
    if not file_path.is_file():
        raise FileNotFoundError("File not found: " + str(file_path))
    if file_path.suffix != ".csv":
        raise ValueError("File " + str(file_path) + " is not a CSV file.")

    studio_df = pd.read_csv(file_path)

    #check that we have all the columns we need
    missing_cols = REQUIRED_COLUMNS - set(studio_df.columns)
    if missing_cols:
        raise ValueError("File " + str(file_path) + " is missing required columns: " + str(sorted(missing_cols)))

    studios: dict[str, Studio] = {}

    #go row by row and build a Studio object for each one
    for idx, row in studio_df.iterrows():
        try:
            #some studios have two ride blocks in the data, so we add them together
            weekly_class_count = int(row["rides_per_wk_a"]) + int(row["rides_per_wk_b"])
            max_class_capacity = int(row["bikes_per_ride_a"]) #we use the first block's bike count as capacity
            studio = Studio(studio_id=row["studio_id"], studio_name=row["studio_name"], official_region=row["official_region"], network_market=row["network_market"], market_tier=row["market_tier"], local_ridership_cluster=row["local_ridership_cluster"], weekly_class_count=weekly_class_count, class_capacity=max_class_capacity)
        except (ValueError, TypeError, KeyError) as e:
            line_num = idx + 2 #add 2 because pandas is 0-indexed and the csv has a header row
            raise ValueError("Error parsing studio data for row " + str(line_num) + ": " + str(e)) from e

        studios[row["studio_id"]] = studio

    return studios
