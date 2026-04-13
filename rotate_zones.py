"""
Rotate all GeoJSON zone coordinates by 85° counter-clockwise
around the site centroid, accounting for lat/lon aspect ratio.
"""
import json, math, copy

INPUT  = r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete(trial).geojson"
OUTPUT = r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete(trial).geojson"

ANGLE_DEG = 85.0  # left tilt, counter-clockwise
angle_rad = math.radians(ANGLE_DEG)
cos_a = math.cos(angle_rad)
sin_a = math.sin(angle_rad)

# ─── Load ─────────────────────────────────────
with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

# ─── Find centroid of site_boundary ───────────
site_boundary = None
for feat in data["features"]:
    if feat["properties"]["zone"] == "site_boundary":
        site_boundary = feat["geometry"]["coordinates"][0]  # outer ring
        break

cx = sum(c[0] for c in site_boundary[:-1]) / (len(site_boundary) - 1)
cy = sum(c[1] for c in site_boundary[:-1]) / (len(site_boundary) - 1)
print(f"Rotation center: ({cx:.8f}, {cy:.8f})")

# Aspect ratio correction: 1° lon is shorter than 1° lat at this latitude
aspect = math.cos(math.radians(cy))
print(f"Aspect ratio correction (cos {cy:.4f}°): {aspect:.6f}")

# ─── Rotate a single coordinate ───────────────
def rotate_coord(coord):
    lon, lat = coord[0], coord[1]
    z = coord[2] if len(coord) > 2 else None

    # Shift to center, correct aspect ratio
    dx = (lon - cx) * aspect
    dy = lat - cy

    # Apply rotation
    dx2 = dx * cos_a - dy * sin_a
    dy2 = dx * sin_a + dy * cos_a

    # Undo aspect correction, shift back
    new_lon = dx2 / aspect + cx
    new_lat = dy2 + cy

    if z is not None:
        return [round(new_lon, 8), round(new_lat, 8), z]
    else:
        return [round(new_lon, 8), round(new_lat, 8)]

# ─── Walk and rotate all coordinates ──────────
def rotate_coords(coords):
    """Recursively rotate coordinate arrays."""
    if isinstance(coords[0], (int, float)):
        # This is a single coordinate [lon, lat] or [lon, lat, z]
        return rotate_coord(coords)
    else:
        return [rotate_coords(c) for c in coords]

rotated = copy.deepcopy(data)

# Remove camera features
rotated["features"] = [
    f for f in rotated["features"] 
    if f["properties"].get("zone") not in ["camera_fov_CAM01", "camera_CAM01_position"]
]
for feat in rotated["features"]:
    geom = feat["geometry"]
    geom["coordinates"] = rotate_coords(geom["coordinates"])

# Update metadata
rotated["metadata"]["rotation_deg"] = abs(ANGLE_DEG)
rotated["metadata"]["rotation_note"] = f"All coordinates rotated {abs(ANGLE_DEG)}° counter-clockwise from axis-aligned to match real-world building orientation"

# ─── Save ──────────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(rotated, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Rotated all features by {ANGLE_DEG} deg around centroid")
print(f"   Saved to: {OUTPUT}")

# Print before/after for site boundary corners
print("\n── Site Boundary (before → after) ──")
for i, (old, new) in enumerate(zip(site_boundary[:-1], rotated["features"][0]["geometry"]["coordinates"][0][:-1])):
    corners = ["SW", "SE", "NE", "NW"]
    print(f"   {corners[i]}: ({old[0]:.8f}, {old[1]:.8f}) → ({new[0]:.8f}, {new[1]:.8f})")
