const fs = require('fs');
const geojson = JSON.parse(fs.readFileSync('buildsight_zones_dashboard.geojson', 'utf8'));

const html = `
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        #map { width: 1280px; height: 900px; background: #0b1118; }
        body { margin: 0; padding: 0; background: #0b1118; font-family: sans-serif; }
        .zone-label {
            background: transparent;
            border: none;
            box-shadow: none;
            color: #fff;
            font-weight: bold;
            font-size: 11px;
            text-shadow: 1px 1px 2px #000, -1px -1px 2px #000, 1px -1px 2px #000, -1px 1px 2px #000;
            white-space: nowrap;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        const geojsonData = ${JSON.stringify(geojson)};
        
        // Use satellite tiles
        const map = L.map('map', { zoomControl: false });
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri',
            maxZoom: 20
        }).addTo(map);

        const geojsonLayer = L.geoJSON(geojsonData, {
            style: function (feature) {
                return feature.properties.style || {
                    color: '#00ffff',
                    weight: 2,
                    fillColor: '#00ffff',
                    fillOpacity: 0.2
                };
            },
            onEachFeature: function(feature, layer) {
                if (feature.properties.show_label && feature.properties.display_name) {
                    layer.bindTooltip(feature.properties.display_name, {
                        permanent: true,
                        direction: 'center',
                        className: 'zone-label'
                    });
                }
            }
        }).addTo(map);

        map.fitBounds(geojsonLayer.getBounds(), { padding: [50, 50] });
    </script>
</body>
</html>
`;
fs.writeFileSync('render_geojson.html', html);
