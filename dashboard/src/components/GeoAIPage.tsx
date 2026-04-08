import { motion } from 'framer-motion'

/* ── GeoAI Placeholder Module ─────────────────────────────────────────────── */
/* Provides visual placeholders for:                                          */
/*   1. Heatmap Panel                                                        */
/*   2. Zone Overlay                                                         */
/*   3. Geofencing                                                           */
/*   4. Risk Zone Visualization                                               */
/*   5. Worker Movement Paths                                                 */
/*   6. Incident Hotspot                                                      */
/*                                                                             */
/* These will connect to the backend /api/geoai/* endpoints once QGIS         */
/* integration is complete.                                                    */
/* ──────────────────────────────────────────────────────────────────────────── */

const cardVariants = {
  hidden: { opacity: 0, y: 16, scale: 0.98 },
  visible: (i: number) => ({
    opacity: 1, y: 0, scale: 1,
    transition: { 
      delay: i * 0.05, 
      type: 'spring' as const,
      stiffness: 260,
      damping: 24
    },
  }),
}

const MODULES = [
  {
    id: 'heatmap',
    icon: '🌡️',
    title: 'Heatmap Panel',
    subtitle: 'Worker Density & Activity Heat',
    description: 'Real-time heatmap overlay showing worker density patterns across site zones. Connects to /api/geoai/heatmap for live spatial data.',
    status: 'Awaiting QGIS Data',
    endpoint: 'GET /api/geoai/heatmap',
  },
  {
    id: 'zones',
    icon: '📐',
    title: 'Zone Overlay',
    subtitle: 'Restricted & Active Zone Mapping',
    description: 'Polygon overlays from GeoJSON — restricted areas, crane swing radii, material staging. Upload via /api/geoai/upload.',
    status: 'Awaiting GeoJSON',
    endpoint: 'GET /api/geoai/zones',
  },
  {
    id: 'geofencing',
    icon: '🚧',
    title: 'Geofencing Engine',
    subtitle: 'Virtual Perimeter Enforcement',
    description: 'Geofence triggers when workers enter exclusion zones. Integrates with AlertLog for real-time breach notifications.',
    status: 'Pending Integration',
    endpoint: 'POST /api/geoai/geofence/check',
  },
  {
    id: 'risk-zones',
    icon: '⚠️',
    title: 'Risk Zone Visualization',
    subtitle: 'Dynamic Hazard Classification',
    description: 'Color-coded risk zones (Low / Medium / High / Critical) based on incident history and proximity violations.',
    status: 'Pending Analytics',
    endpoint: 'GET /api/geoai/risk-zones',
  },
  {
    id: 'movement',
    icon: '🔄',
    title: 'Worker Movement Paths',
    subtitle: 'Temporal Trajectory Tracking',
    description: 'Worker path reconstruction from sequential detection coordinates. Visualizes flow patterns and dwell-time anomalies.',
    status: 'Awaiting Tracker',
    endpoint: 'GET /api/geoai/worker-tracks',
  },
  {
    id: 'hotspots',
    icon: '🔴',
    title: 'Incident Hotspots',
    subtitle: 'Historical Incident Clustering',
    description: 'Cluster analysis of PPE violations mapped to spatial coordinates. Identifies repeat-offender zones for targeted inspections.',
    status: 'Awaiting Data',
    endpoint: 'GET /api/geoai/violations/map',
  },
]

export function GeoAIPage() {
  return (
    <div className="geoai-page">
      <div className="geoai-page__header">
        <div>
          <p className="section-label">Spatial Intelligence</p>
          <h2>GeoAI Integration</h2>
        </div>
        <div className="geoai-page__status">
          <span className="geoai-status-pill">◌ QGIS STANDBY</span>
          <button
            className="geoai-upload-btn"
            onClick={async () => {
              const input = document.createElement('input')
              input.type = 'file'
              input.accept = '.geojson,.json'
              input.onchange = async () => {
                const file = input.files?.[0]
                if (!file) return
                const formData = new FormData()
                formData.append('file', file)
                try {
                  const res = await fetch('http://localhost:8000/api/geoai/upload', {
                    method: 'POST',
                    body: formData,
                  })
                  if (res.ok) {
                    alert('GeoJSON uploaded successfully!')
                  } else {
                    alert('Upload failed. Check backend logs.')
                  }
                } catch {
                  alert('Backend not available.')
                }
              }
              input.click()
            }}
          >
            ↑ Upload GeoJSON
          </button>
        </div>
      </div>

      {/* ── Map Canvas Placeholder ────────────────────────────────────────── */}
      <motion.div
        className="panel geoai-map-canvas"
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
      >
        <div className="geoai-map-placeholder">
          <div className="geoai-map-grid">
            {/* Simulated grid dots */}
            {Array.from({ length: 120 }, (_, i) => (
              <motion.div
                key={i}
                className="geoai-grid-dot"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.15 + Math.random() * 0.25 }}
                transition={{ delay: i * 0.008, duration: 0.5 }}
              />
            ))}
          </div>
          <div className="geoai-map-overlay-text">
            <span className="geoai-map-icon">🗺️</span>
            <h4>QGIS Integration Ready</h4>
            <p>Upload GeoJSON to activate spatial overlays</p>
            <p className="geoai-map-hint">Supports: Polygons · Points · LineStrings · MultiPolygons</p>
          </div>
        </div>
      </motion.div>

      {/* ── Module Cards ─────────────────────────────────────────────────── */}
      <div className="geoai-modules-grid">
        {MODULES.map((mod, i) => (
          <motion.div
            className="panel geoai-module-card"
            key={mod.id}
            custom={i}
            initial="hidden"
            animate="visible"
            variants={cardVariants}
          >
            <div className="geoai-module-card__header">
              <span className="geoai-module-icon">{mod.icon}</span>
              <div>
                <h4>{mod.title}</h4>
                <span className="geoai-module-subtitle">{mod.subtitle}</span>
              </div>
            </div>
            <p className="geoai-module-desc">{mod.description}</p>
            <div className="geoai-module-footer">
              <code className="geoai-endpoint">{mod.endpoint}</code>
              <span className="geoai-module-status">{mod.status}</span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
