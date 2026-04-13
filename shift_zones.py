import json
import copy
import shutil

INPUT  = r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete.geojson"
OUTPUT = r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete(trial).geojson"

# Adjust these values to move the zones
# 0.00001 degrees is roughly 1 meter
DX = 0.00106359    # Absolute shift to align NW corner with Target Lon
DY = -0.00165374   # Absolute shift to align NW corner with Target Lat

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

def shift_coord(coord):
    lon, lat = coord[0], coord[1]
    z = coord[2] if len(coord) > 2 else None
    
    new_lon = lon + DX
    new_lat = lat + DY
    
    if z is not None:
        return [round(new_lon, 8), round(new_lat, 8), z]
    else:
        return [round(new_lon, 8), round(new_lat, 8)]

def shift_coords(coords):
    if isinstance(coords[0], (int, float)):
        return shift_coord(coords)
    else:
        return [shift_coords(c) for c in coords]

shifted = copy.deepcopy(data)
for feat in shifted["features"]:
    geom = feat["geometry"]
    geom["coordinates"] = shift_coords(geom["coordinates"])

# Save to trial
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(shifted, f, indent=2, ensure_ascii=False)

# Copy to final destinations
destinations = [
    r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete.geojson",
    r"E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\public\buildsight_zones_complete.geojson",
    r"E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\dist\buildsight_zones_complete.geojson"
]

for dest in destinations:
    shutil.copy2(OUTPUT, dest)

print(f"Successfully shifted zones by DX={DX}, DY={DY}")
print("Updates copied to public and dist folders.")
