import json
import glob

# The original unrotated GeoJSONs
files = [
    r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete.geojson",
    r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete(trial).geojson",
    r"E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\public\buildsight_zones_complete.geojson",
    r"E:\Company\Green Build AI\Prototypes\BuildSight\dashboard\dist\buildsight_zones_complete.geojson"
]

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # If it was rotated, let's load from the original anyway to reset it
        if "(trial)" in filepath:
            with open(r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_zones_complete.geojson", 'r', encoding='utf-8') as f:
                data = json.load(f)

        for feat in data['features']:
            props = feat['properties']
            if props['zone'] == 'camera_fov_CAM01':
                props['host_building'] = "Neighbour building (north)"
                props['direction'] = "Looking south into construction site"
                # New Apex: [78.66894568, 10.81674386, 7.62], NW, SW, SE, NE, Apex
                feat['geometry']['coordinates'] = [[
                    [78.66894568, 10.81674386, 7.62],
                    [78.66881504, 10.81668959, 0.0],
                    [78.66881504, 10.81658333, 0.0],
                    [78.66902447, 10.81658333, 0.0],
                    [78.66902447, 10.81668959, 0.0],
                    [78.66894568, 10.81674386, 7.62]
                ]]
            elif props['zone'] == 'camera_CAM01_position':
                props['host_building'] = "Neighbour building (north)"
                props['coverage'] = "Full site top-down angled view looking SOUTH"
                feat['geometry']['coordinates'] = [78.66894568, 10.81674386, 7.62]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated {filepath}")
    except Exception as e:
        print(f"Failed {filepath}: {e}")
