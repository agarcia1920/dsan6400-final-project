from soulcycle_network.studio import Studio
from soulcycle_network.studio_loader import load_studios

#build one studio by hand first
georgetown = Studio(studio_id="GTWN", studio_name="Georgetown", official_region="DC", network_market="DMV", market_tier="Large", local_ridership_cluster="DC-Arlington", weekly_class_count=36, class_capacity=59)

print(georgetown)

#then load the full network from the csv
studios = load_studios("data/studios.csv")

print(len(studios))
print(studios["GTWN"])
