import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import type { HeatmapUpdatePayload, ZoneCollection, ZoneFeature } from '../types/geoai'

const SITE_CENTER: [number, number] = [10.81662, 78.66891]
const INITIAL_ZOOM = 19
const TARGET_OVERLAY_ROTATION_DEG = 85.0
const OVERLAY_OFFSET_METERS: [number, number] = [10.5, -2.5]
const OVERLAY_SCALE = 0.80
const CAMERA_OFFSET_METERS: [number, number] = [1.2, 0]
const CAMERA_FOV_SCALE = 1.16
const LABEL_OFFSET_METERS: Record<string, [number, number]> = {
  high_risk_scaffolding: [0, 2.4],
  high_risk_staircase: [-1.1, 0.85],
  moderate_risk_interior: [0, 0],
  low_risk_parking: [1.1, 0],
}

const RISK_COLORS: Record<string, string> = {
  CRITICAL: '#ff3b3b',
  HIGH: '#ff7b00',
  MODERATE: '#ffd600',
  LOW: '#00e676',
  none: '#00b4d8',
}

const RISK_FILL_OPACITY: Record<string, number> = {
  CRITICAL: 0.35,
  HIGH: 0.25,
  MODERATE: 0.18,
  LOW: 0.12,
  none: 0.08,
}

interface GeoAIMapProps {
  data: HeatmapUpdatePayload | null
  showZones: boolean
  showLabels: boolean
  showCameraFOV: boolean
  showHeatmap: boolean
  showWorkers: boolean
  heatmapOpacity?: number
  viewMode: string
}

interface OverlayMetadata {
  rotation_deg?: number
}

type LatLngTuple = [number, number]
type LngLatTuple = [number, number]

function getMetersPerDegreeLon(lat: number) {
  return 111320 * Math.cos((lat * Math.PI) / 180)
}

function rotateLatLng(point: LatLngTuple, center: LatLngTuple, angleDeg: number): LatLngTuple {
  if (Math.abs(angleDeg) < 0.0001) return point

  const theta = (angleDeg * Math.PI) / 180
  const cosTheta = Math.cos(theta)
  const sinTheta = Math.sin(theta)
  const metersPerDegLat = 110574
  const metersPerDegLon = getMetersPerDegreeLon(center[0])

  const dx = (point[1] - center[1]) * metersPerDegLon
  const dy = (point[0] - center[0]) * metersPerDegLat

  const rx = dx * cosTheta - dy * sinTheta
  const ry = dx * sinTheta + dy * cosTheta

  return [
    center[0] + ry / metersPerDegLat,
    center[1] + rx / metersPerDegLon,
  ]
}

function scaleLatLng(point: LatLngTuple, center: LatLngTuple, scale: number): LatLngTuple {
  if (Math.abs(scale - 1) < 0.0001) return point

  const metersPerDegLat = 110574
  const metersPerDegLon = getMetersPerDegreeLon(center[0])
  const dx = (point[1] - center[1]) * metersPerDegLon
  const dy = (point[0] - center[0]) * metersPerDegLat

  return [
    center[0] + (dy * scale) / metersPerDegLat,
    center[1] + (dx * scale) / metersPerDegLon,
  ]
}

function offsetLatLng(point: LatLngTuple, offsetMeters: [number, number]): LatLngTuple {
  const [dxMeters, dyMeters] = offsetMeters
  const metersPerDegLat = 110574
  const metersPerDegLon = getMetersPerDegreeLon(point[0])
  return [
    point[0] + dyMeters / metersPerDegLat,
    point[1] + dxMeters / metersPerDegLon,
  ]
}

function scaleFromAnchor(point: LatLngTuple, anchor: LatLngTuple, scale: number): LatLngTuple {
  if (Math.abs(scale - 1) < 0.0001) return point

  const metersPerDegLat = 110574
  const metersPerDegLon = getMetersPerDegreeLon(anchor[0])
  const dx = (point[1] - anchor[1]) * metersPerDegLon
  const dy = (point[0] - anchor[0]) * metersPerDegLat

  return [
    anchor[0] + (dy * scale) / metersPerDegLat,
    anchor[1] + (dx * scale) / metersPerDegLon,
  ]
}

function computeOverlayCenter(features: ZoneFeature[]): LatLngTuple {
  const cornerPoints = features
    .filter((feature) => feature.geometry.type === 'Point' && feature.properties.type === 'building_corner')
    .map((feature) => {
      const [lng, lat] = feature.geometry.coordinates as number[]
      return [lat, lng] as LatLngTuple
    })

  if (cornerPoints.length) {
    const lat = cornerPoints.reduce((sum, point) => sum + point[0], 0) / cornerPoints.length
    const lng = cornerPoints.reduce((sum, point) => sum + point[1], 0) / cornerPoints.length
    return [lat, lng]
  }

  const boundary = features.find((feature) => feature.properties.zone === 'site_boundary' && feature.geometry.type === 'Polygon')
  if (boundary) {
    const ring = (boundary.geometry.coordinates as number[][][])[0]
    const lats = ring.map((coord) => coord[1])
    const lngs = ring.map((coord) => coord[0])
    return [
      (Math.min(...lats) + Math.max(...lats)) / 2,
      (Math.min(...lngs) + Math.max(...lngs)) / 2,
    ]
  }

  return SITE_CENTER
}

export function GeoAIMap({
  data,
  showZones,
  showLabels,
  showCameraFOV,
  showHeatmap,
  showWorkers,
  heatmapOpacity = 0.65,
  viewMode
}: GeoAIMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<any>(null)
  const tacticalTileRef = useRef<any>(null)
  const satelliteTileRef = useRef<any>(null)
  const heatLayerRef = useRef<any>(null)
  const zonesLayerRef = useRef<any>(null)
  const labelsLayerRef = useRef<any>(null)
  const fovLayerRef = useRef<any>(null)
  const workersLayerRef = useRef<any>(null)
  const [leafletReady, setLeafletReady] = useState(false)
  const [geoData, setGeoData] = useState<ZoneCollection | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)

  const TACTICAL_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
  const SATELLITE_URL = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'

  const overlayCenter = useMemo(
    () => (geoData ? computeOverlayCenter(geoData.features) : SITE_CENTER),
    [geoData]
  )

  const overlayRotationDelta = useMemo(() => {
    const metadata = (geoData as ZoneCollection & { metadata?: OverlayMetadata } | null)?.metadata
    const sourceRotation = metadata?.rotation_deg ?? 0
    return TARGET_OVERLAY_ROTATION_DEG - sourceRotation
  }, [geoData])

  const normalizeLatLng = useCallback(
    (point: LatLngTuple): LatLngTuple =>
      offsetLatLng(
        scaleLatLng(
          rotateLatLng(point, overlayCenter, overlayRotationDelta),
          overlayCenter,
          OVERLAY_SCALE
        ),
        OVERLAY_OFFSET_METERS
      ),
    [overlayCenter, overlayRotationDelta]
  )

  const normalizeLngLat = useCallback(
    (point: LngLatTuple): LatLngTuple => normalizeLatLng([point[1], point[0]]),
    [normalizeLatLng]
  )

  useEffect(() => {
    if (typeof window !== 'undefined' && window.L) {
      setLeafletReady(true)
      return
    }
    const check = setInterval(() => {
      if (window.L) {
        setLeafletReady(true)
        clearInterval(check)
      }
    }, 200)
    return () => clearInterval(check)
  }, [])

  useEffect(() => {
    fetch('/buildsight_zones_complete.geojson')
      .then(r => {
        if (!r.ok) throw new Error('GeoJSON fetch failed')
        return r.json()
      })
      .then((d: ZoneCollection) => {
        setGeoData(d)
        setFetchError(null)
      })
      .catch((err) => {
        console.warn('GeoAI: Could not load buildsight_zones_complete.geojson', err)
        setFetchError('Failed to parse hazard zones')
      })
  }, [])

  useEffect(() => {
    if (!leafletReady || !containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      center: SITE_CENTER,
      zoom: INITIAL_ZOOM,
      zoomControl: false,
      attributionControl: false,
    })

    const isSat = viewMode === 'satellite'

    satelliteTileRef.current = L.tileLayer(SATELLITE_URL, {
      maxZoom: 22,
      subdomains: 'abcd',
      opacity: isSat ? 1 : 0
    }).addTo(map)

    tacticalTileRef.current = L.tileLayer(TACTICAL_URL, {
      maxZoom: 22,
      subdomains: 'abcd',
      opacity: isSat ? 0 : 1
    }).addTo(map)

    L.control.zoom({ position: 'bottomright' }).addTo(map)

    if (window.HeatmapOverlay) {
      const heatLayer = new window.HeatmapOverlay({
        radius: 25,
        maxOpacity: heatmapOpacity,
        minOpacity: 0.08,
        blur: 0.85,
        scaleRadius: false,
        useLocalExtrema: false,
        latField: 'lat',
        lngField: 'lng',
        valueField: 'value',
        gradient: {
          '0.0': '#000000',
          '0.2': '#00e5ff',
          '0.4': '#00ff88',
          '0.6': '#ffd600',
          '0.8': '#ff8c00',
          '1.0': '#ff2a2a',
        },
      })
      map.addLayer(heatLayer as any)
      heatLayerRef.current = heatLayer as unknown as typeof heatLayerRef.current
    }

    zonesLayerRef.current = L.layerGroup().addTo(map)
    labelsLayerRef.current = L.layerGroup().addTo(map)
    fovLayerRef.current = L.layerGroup().addTo(map)
    workersLayerRef.current = L.layerGroup().addTo(map)

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [leafletReady])

  useEffect(() => {
    if (!mapRef.current || !tacticalTileRef.current || !satelliteTileRef.current) return

    const isSat = viewMode === 'satellite'
    const targetSatOpacity = isSat ? 1 : 0
    const targetTacOpacity = isSat ? 0 : 1

    let progress = 0
    const startSatOpacity = satelliteTileRef.current.options.opacity
    const startTacOpacity = tacticalTileRef.current.options.opacity

    const fadeInterval = setInterval(() => {
      progress += 0.08
      if (progress >= 1) progress = 1

      satelliteTileRef.current.setOpacity(startSatOpacity + (targetSatOpacity - startSatOpacity) * progress)
      tacticalTileRef.current.setOpacity(startTacOpacity + (targetTacOpacity - startTacOpacity) * progress)

      if (progress === 1) {
        clearInterval(fadeInterval)
      }
    }, 20)
  }, [viewMode])

  useEffect(() => {
    if (!mapRef.current || !geoData || !zonesLayerRef.current || !labelsLayerRef.current || !fovLayerRef.current) return

    let renderTimeout = setTimeout(() => {
      zonesLayerRef.current.clearLayers()
      labelsLayerRef.current.clearLayers()
      fovLayerRef.current.clearLayers()

      geoData.features.forEach((feature: ZoneFeature) => {
        const props = feature.properties
        const geom = feature.geometry

        if (geom.type === 'Point') {
          // Render camera markers
          if (props.type === 'camera') {
            const coord = offsetLatLng(
              normalizeLngLat(geom.coordinates as LngLatTuple),
              CAMERA_OFFSET_METERS
            )
            const camIcon = L.divIcon({
              className: 'geoai-camera-marker',
              html: `<div style="
                background: rgba(0, 180, 216, 0.9);
                border: 2px solid #fff;
                border-radius: 50%;
                width: 22px; height: 22px;
                display: flex; align-items: center; justify-content: center;
                font-size: 12px; box-shadow: 0 0 12px rgba(0,229,255,0.6);
              ">📷</div>`,
              iconSize: [22, 22],
              iconAnchor: [11, 11],
            })
            const camMarker = L.marker(coord, { icon: camIcon })
            camMarker.bindPopup(`
              <div class="geoai-popup">
                <strong>${props.camera_id ?? 'CAM'}</strong><br/>
                Height: ${props.height_ft ?? '?'}ft<br/>
                Direction: ${props.direction ?? '?'}<br/>
                Placement: Neighbour building (North)
              </div>
            `, { className: 'geoai-popup-container' })
            fovLayerRef.current!.addLayer(camMarker)
          }
          return
        }

        const risk = (props.risk || 'none').toUpperCase()
        const isFOV = props.type === 'camera_coverage' || props.zone?.includes('camera_fov')
        const isBoundary = props.zone === 'site_boundary'

        if (isFOV) {
          const baseCoords = (geom.coordinates as number[][][])[0].map(
            (c: number[]) => normalizeLngLat(c as LngLatTuple)
          )
          const apex = offsetLatLng(baseCoords[0], CAMERA_OFFSET_METERS)
          const coords = baseCoords.map((coord, index, allCoords) => {
            if (index === 0 || index === allCoords.length - 1) return apex
            return scaleFromAnchor(
              offsetLatLng(coord, CAMERA_OFFSET_METERS),
              apex,
              CAMERA_FOV_SCALE
            )
          })
          const fovPoly = L.polygon(coords, {
            color: '#00e5ff',
            weight: 1,
            dashArray: '6 4',
            fillColor: '#00e5ff',
            fillOpacity: 0.08,
          })
          fovLayerRef.current!.addLayer(fovPoly)
          return
        }

        if (isBoundary) {
          const coords = (geom.coordinates as number[][][])[0].map(
            (c: number[]) => normalizeLngLat(c as LngLatTuple)
          )
          const boundaryPoly = L.polygon(coords, {
            color: '#c0cdd9',
            weight: 2,
            dashArray: '8 6',
            fill: false,
          })
          zonesLayerRef.current!.addLayer(boundaryPoly)
          return
        }

        const zoneColor = RISK_COLORS[risk] || RISK_COLORS.none
        const fillOpacity = RISK_FILL_OPACITY[risk] || 0.1

        if (geom.type === 'Polygon') {
          const rings = (geom.coordinates as number[][][]).map(ring =>
            ring.map((c: number[]) => normalizeLngLat(c as LngLatTuple))
          )
          const zonePoly = L.polygon(rings, {
            color: zoneColor,
            weight: 1.5,
            fillColor: zoneColor,
            fillOpacity,
          })
          zonesLayerRef.current!.addLayer(zonePoly)

          const bounds = zonePoly.getBounds()
          const center = bounds.getCenter()
          const labelPosition = offsetLatLng(
            [center.lat, center.lng],
            LABEL_OFFSET_METERS[props.zone ?? ''] ?? [0, 0]
          )
          const labelText = (props.zone || '').replace(/_/g, ' ').toUpperCase()
          const label = L.marker(labelPosition, {
            icon: L.divIcon({
              className: 'geoai-zone-label',
              html: `<span style="color:${zoneColor}; text-shadow: 0 0 10px rgba(0,0,0,0.8)">${labelText}</span>`,
              iconSize: [120, 16],
              iconAnchor: [60, 8],
            }),
          })
          labelsLayerRef.current!.addLayer(label)
        }
      })
    }, 150) // Debounce

    return () => clearTimeout(renderTimeout)
  }, [geoData, normalizeLngLat])

  useEffect(() => {
    if (!mapRef.current || !zonesLayerRef.current) return
    if (showZones) {
      mapRef.current.addLayer(zonesLayerRef.current)
    } else {
      mapRef.current.removeLayer(zonesLayerRef.current)
    }
  }, [showZones])

  useEffect(() => {
    if (!mapRef.current || !labelsLayerRef.current) return
    if (showLabels) {
      mapRef.current.addLayer(labelsLayerRef.current)
    } else {
      mapRef.current.removeLayer(labelsLayerRef.current)
    }
  }, [showLabels])

  useEffect(() => {
    if (!mapRef.current || !fovLayerRef.current) return
    if (showCameraFOV) {
      mapRef.current.addLayer(fovLayerRef.current)
    } else {
      mapRef.current.removeLayer(fovLayerRef.current)
    }
  }, [showCameraFOV])

  useEffect(() => {
    if (!heatLayerRef.current || !data?.cells) return
    if (!showHeatmap) {
      heatLayerRef.current.setData({ min: 0, max: 1, data: [] })
      return
    }
    const hmData = {
      min: 0,
      max: 1,
      data: data.cells
        .filter(c => c.risk_score > 0.05)
        .map(c => {
          const [lat, lng] = normalizeLatLng([c.lat, c.lon])
          return { lat, lng, value: c.risk_score }
        }),
    }
    heatLayerRef.current.setData(hmData)
  }, [data, showHeatmap, normalizeLatLng])

  useEffect(() => {
    if (!mapRef.current || !workersLayerRef.current || !data?.workers) return
    workersLayerRef.current.clearLayers()

    if (!showWorkers) return

    data.workers.forEach(w => {
      const riskColor = RISK_COLORS[w.risk] || '#00e676'
      const ppeClass = w.ppe_ok ? 'ppe-ok' : 'ppe-violation'
      const helmetIcon = w.has_helmet ? '⛑️' : '⚠'
      const vestIcon = w.has_vest ? '🦺' : '⚠'

      const isAtRisk = w.status === 'AT_RISK' || w.status === 'CRITICAL'
      const statusPulse = isAtRisk ? '<div class="geoai-worker-pulse"></div>' : ''
      const dwellBadge = w.dwell_time_s && w.dwell_time_s > 30
        ? `<span class="geoai-worker-dwell">${Math.floor(w.dwell_time_s)}s</span>`
        : ''

      const icon = L.divIcon({
        className: `geoai-worker-marker ${ppeClass}`,
        html: `
          ${statusPulse}
          <div class="geoai-worker-dot" style="background:${riskColor};box-shadow:0 0 12px ${riskColor}">
            <span class="geoai-worker-ppe">${helmetIcon}${vestIcon}</span>
          </div>
          ${dwellBadge}
        `,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      })

      const [lat, lon] = normalizeLatLng([w.lat, w.lon])
      const marker = L.marker([lat, lon], { icon })
      marker.bindPopup(`
        <div class="geoai-popup">
          <strong>Worker:</strong> W${w.worker_id ?? '?'} &nbsp; <strong>Camera:</strong> ${w.camera_id ?? 'CAM-01'}<br/>
          <strong>Zone:</strong> ${w.zone.replace(/_/g, ' ')}<br/>
          <strong>Risk:</strong> <span style="color:${riskColor}">${w.risk}</span>
          &nbsp; <strong>Status:</strong> <span style="color:${riskColor}">${w.status ?? '—'}</span><br/>
          <strong>Height:</strong> ${w.height_m}m<br/>
          <strong>Helmet:</strong> ${w.has_helmet ? '✅' : '❌'} &nbsp;
          <strong>Vest:</strong> ${w.has_vest ? '✅' : '❌'}<br/>
          ${w.dwell_time_s && w.dwell_time_s > 0 ? `<strong>Dwell:</strong> ${w.dwell_time_s}s (${w.dwell_severity ?? ''})` : ''}
        </div>
      `, { className: 'geoai-popup-container' })

      workersLayerRef.current!.addLayer(marker)
    })
  }, [data, showWorkers, normalizeLatLng])

  return (
    <div className="geoai-map-wrapper" style={{ height: '100%' }}>
      <div ref={containerRef} className="geoai-map-container" style={{ height: '100%' }} />

      {fetchError && (
        <div className="geoai-toast" style={{ position: 'absolute', top: 20, right: 20, zIndex: 1000 }}>
          <span>{fetchError}</span>
        </div>
      )}

      {!leafletReady && (
        <div className="geoai-map-loading">
          <div className="geoai-map-loading__spinner" />
          <span>Initializing Spatial Engine…</span>
        </div>
      )}
    </div>
  )
}
