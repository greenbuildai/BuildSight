import json
import matplotlib.pyplot as plt

filepath = r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete(trial).geojson"

with open(filepath, 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(10, 10))

for feat in data['features']:
    geom = feat['geometry']
    props = feat['properties']
    if geom['type'] == 'Polygon':
        coords = geom['coordinates'][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        if props['zone'].startswith('camera_fov'):
            ax.plot(xs, ys, label=props['zone'], color='cyan', alpha=0.5)
            ax.fill(xs, ys, color='cyan', alpha=0.2)
        elif props['zone'] == 'site_boundary':
            ax.plot(xs, ys, label='site_boundary', color='black', linewidth=2)
        else:
            ax.plot(xs, ys, label=props.get('zone', 'zone'), linestyle='--')
    elif geom['type'] == 'Point':
        c = geom['coordinates']
        if props['type'] == 'camera':
            ax.plot(c[0], c[1], 'bo', markersize=10, label='Camera')

ax.set_aspect('equal', 'datalim')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Zones Verification")
plt.tight_layout()
plt.savefig("zones_verification.png")
