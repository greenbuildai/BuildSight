import json
import shutil

INPUT = r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete.geojson"
OUTPUT = r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete(trial).geojson"

TARGET_LAT = 10.816634
TARGET_LON = 78.668821  # Final Micro-shift left again by ~20 inches for 100% flush

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

# Find building_footprint centroid
footprint_coords = []
for f in data["features"]:
    if f["properties"].get("zone") == "site_boundary":
        # using site_boundary or building footprint? The prompt says "coordinates of the building inside the box"
        pass
    if f["properties"].get("zone") == "building_footprint":
        footprint_coords = f["geometry"]["coordinates"][0]
        break

if not footprint_coords:
    # fallback to site_boundary if footprint missing
    for f in data["features"]:
        if f["properties"].get("zone") == "site_boundary":
            footprint_coords = f["geometry"]["coordinates"][0]
            break

lats = [pt[1] for pt in footprint_coords]
lons = [pt[0] for pt in footprint_coords]
current_lat = sum(lats) / len(lats)
current_lon = sum(lons) / len(lons)

dlat = TARGET_LAT - current_lat
dlon = TARGET_LON - current_lon

print(f"Current Centroid: {current_lat}, {current_lon}")
print(f"Target Centroid: {TARGET_LAT}, {TARGET_LON}")
print(f"Applying Delta: dLat={dlat}, dLon={dlon}")

def shift_coords(coords):
    if isinstance(coords[0], (int, float)):
        lon, lat = coords[0], coords[1]
        z = coords[2] if len(coords) > 2 else None
        new_lon = lon + dlon
        new_lat = lat + dlat
        if z is not None:
            return [round(new_lon, 8), round(new_lat, 8), z]
        return [round(new_lon, 8), round(new_lat, 8)]
    return [shift_coords(c) for c in coords]

for feat in data["features"]:
    feat["geometry"]["coordinates"] = shift_coords(feat["geometry"]["coordinates"])

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

destinations = [
    r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete.geojson",
    r"E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\public\buildsight_zones_complete.geojson",
    r"E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\dist\buildsight_zones_complete.geojson"
]

for dest in destinations:
    shutil.copy2(OUTPUT, dest)

print("Alignment complete and files synchronized.")
