# Load studios from CSV.

from pathlib import Path
from soulcycle_network.studio import Studio
import pandas as pd

#required columns for the studio data
REQUIRED_COLUMNS = {"studio_id", "studio_name", "official_region", "network_market", "market_tier", "local_ridership_cluster", "rides_per_wk_a", "bikes_per_ride_a", "rides_per_wk_b", "bikes_per_ride_b"}

def load_studios(file_path: str | Path) -> dict[str, Studio]:
    #load studios from a CSV file and return them as a dictionary keyed by studio_id
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError("File not found: " + str(file_path))
    if file_path.suffix != ".csv":
        raise ValueError("File " + str(file_path) + " is not a CSV file.")

    studio_df = pd.read_csv(file_path)
    missing_cols = REQUIRED_COLUMNS - set(studio_df.columns)
    if missing_cols:
        raise ValueError("File " + str(file_path) + " is missing required columns: " + str(sorted(missing_cols)))

    studios: dict[str, Studio] = {}

    for idx, row in studio_df.iterrows():
        try:
            rides_a = int(row["rides_per_wk_a"])
            rides_b = int(row["rides_per_wk_b"])
            bikes_a = int(row["bikes_per_ride_a"])
            bikes_b = int(row["bikes_per_ride_b"])

            room_class_counts = {"A": rides_a, "B": rides_b}
            room_capacities = {"A": bikes_a, "B": bikes_b}
            weekly_class_count = rides_a + rides_b

            studio = Studio(
                studio_id=row["studio_id"],
                studio_name=row["studio_name"],
                official_region=row["official_region"],
                network_market=row["network_market"],
                market_tier=row["market_tier"],
                local_ridership_cluster=row["local_ridership_cluster"],
                weekly_class_count=weekly_class_count,
                room_class_counts=room_class_counts,
                room_capacities=room_capacities,
            )
        except (ValueError, TypeError, KeyError) as e:
            line_num = idx + 2
            raise ValueError("Error parsing studio data for row " + str(line_num) + ": " + str(e)) from e

        studios[row["studio_id"]] = studio

    return studios
