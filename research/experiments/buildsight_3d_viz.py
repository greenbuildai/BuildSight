"""
BuildSight — Interactive 3D Risk Zone Visualization
Maran Constructions, Thanjavur | Green Build AI
Run: python buildsight_3d_viz.py
Opens automatically in your default browser.
"""

import plotly.graph_objects as go
import numpy as np

# ─── Site dimensions (meters, origin = SW corner of building) ───
BW   = 18.90   # East-West width
BD   =  9.75   # North-South depth
CAR  =  2.74   # Car parking width (east strip)

# ─── Building heights (meters from road level) ───
H_TOTAL    = 6.096   # 20ft — full building height
H_GROUND   = 4.877   # 16ft — ground floor slab
H_BASEMENT = 1.524   # 5ft  — basement
H_CAMERA   = 7.620   # 25ft — camera in neighbour building

# ─── Neighbour building position (south of site) ───
CAM_X = BW / 2       # centered east-west
CAM_Y = -8.0         # 8m south of SW corner
CAM_Z = H_CAMERA


def make_box(x0, y0, z0, x1, y1, z1, color, name, opacity=0.45):
    """
    Create a solid 3D box (6 faces) as a Mesh3d object.
    Coordinates: X=East, Y=North, Z=Up (meters from SW corner at road level)
    """
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]

    # 12 triangles forming 6 faces
    i = [0,0,0,0,4,4,2,2,1,1,3,3]
    j = [1,2,3,5,5,6,6,7,5,6,7,6]
    k = [2,3,7,4,6,7,7,3,2,2,6,2]

    return go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Width: {x1-x0:.2f}m × {y1-y0:.2f}m × {z1-z0:.2f}m<br>"
            f"Base: {z0:.2f}m | Top: {z1:.2f}m<extra></extra>"
        ),
        showscale=False,
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.3),
        lightposition=dict(x=100, y=200, z=300)
    )


def make_box_edges(x0, y0, z0, x1, y1, z1, color, name):
    """Wireframe edges for a box."""
    lines_x, lines_y, lines_z = [], [], []
    edges = [
        # Bottom face
        [(x0,y0,z0),(x1,y0,z0)], [(x1,y0,z0),(x1,y1,z0)],
        [(x1,y1,z0),(x0,y1,z0)], [(x0,y1,z0),(x0,y0,z0)],
        # Top face
        [(x0,y0,z1),(x1,y0,z1)], [(x1,y0,z1),(x1,y1,z1)],
        [(x1,y1,z1),(x0,y1,z1)], [(x0,y1,z1),(x0,y0,z1)],
        # Verticals
        [(x0,y0,z0),(x0,y0,z1)], [(x1,y0,z0),(x1,y0,z1)],
        [(x1,y1,z0),(x1,y1,z1)], [(x0,y1,z0),(x0,y1,z1)],
    ]
    for (p1, p2) in edges:
        lines_x += [p1[0], p2[0], None]
        lines_y += [p1[1], p2[1], None]
        lines_z += [p1[2], p2[2], None]
    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color=color, width=2),
        name=name + " (outline)",
        hoverinfo='skip',
        showlegend=False
    )


def make_camera_fov(cx, cy, cz, site_corners):
    """Camera FOV cone as transparent mesh."""
    # Apex = camera, base = site corners at ground
    vx = [cx] + [c[0] for c in site_corners]
    vy = [cy] + [c[1] for c in site_corners]
    vz = [cz] + [0.0 for _ in site_corners]
    n  = len(site_corners)
    # Triangles from apex (index 0) to each edge of base
    i_idx, j_idx, k_idx = [], [], []
    for idx in range(n):
        i_idx.append(0)
        j_idx.append(idx + 1)
        k_idx.append((idx + 1) % n + 1)
    return go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=i_idx, j=j_idx, k=k_idx,
        color='#1D9E75',
        opacity=0.10,
        name='Camera FOV',
        hovertemplate='<b>Camera FOV</b><br>CAM-01 coverage cone<extra></extra>',
        showscale=False,
        flatshading=True
    )


# ════════════════════════════════════════════════════════
# BUILD ALL TRACES
# ════════════════════════════════════════════════════════
traces = []

# 1 ── SITE BOUNDARY (transparent outline only)
traces.append(make_box_edges(-3, -3, 0, BW+3, BD+3, 0.5,
                              '#888888', 'Site boundary'))

# 2 ── HIGH RISK: Scaffolding perimeter band
#      Outer shell — then inner zone will overlay as moderate
traces.append(make_box(-2, -2, 0, BW+2, BD+2, H_TOTAL,
                        '#C0392B', 'HIGH — scaffolding perimeter', opacity=0.30))
traces.append(make_box_edges(-2, -2, 0, BW+2, BD+2, H_TOTAL,
                              '#E74C3C', 'HIGH scaffolding'))

# 3 ── HIGH RISK: Staircase zone (NW corner)
traces.append(make_box(0, BD-2.3, 0, 4.88, BD, H_TOTAL,
                        '#8B0000', 'HIGH — staircase (NW)', opacity=0.60))
traces.append(make_box_edges(0, BD-2.3, 0, 4.88, BD, H_TOTAL,
                              '#FF0000', 'HIGH staircase'))

# 4 ── MODERATE RISK: Interior work zone
traces.append(make_box(1, 1, 0, BW-CAR-0.5, BD-1, H_GROUND,
                        '#E67E22', 'MODERATE — interior work', opacity=0.40))
traces.append(make_box_edges(1, 1, 0, BW-CAR-0.5, BD-1, H_GROUND,
                              '#F39C12', 'MODERATE interior'))

# 5 ── LOW RISK: Car parking strip
traces.append(make_box(BW-CAR, 0, 0, BW, BD, H_TOTAL,
                        '#27AE60', 'LOW — car parking', opacity=0.40))
traces.append(make_box_edges(BW-CAR, 0, 0, BW, BD, H_TOTAL,
                              '#2ECC71', 'LOW parking'))

# 6 ── BUILDING SHELL (outer walls, transparent)
traces.append(make_box(0, 0, 0, BW, BD, H_TOTAL,
                        '#BDC3C7', 'Building shell', opacity=0.08))
traces.append(make_box_edges(0, 0, 0, BW, BD, H_TOTAL,
                              '#2C3E50', 'Building outline'))

# 7 ── CAMERA FOV cone
site_corners_fov = [(-2, 0), (BW+2, 0), (BW+2, BD+2), (-2, BD+2)]
traces.append(make_camera_fov(CAM_X, CAM_Y, CAM_Z, site_corners_fov))

# 8 ── CAMERA position marker
traces.append(go.Scatter3d(
    x=[CAM_X], y=[CAM_Y], z=[CAM_Z],
    mode='markers+text',
    marker=dict(size=12, color='#2980B9', symbol='diamond',
                line=dict(color='white', width=2)),
    text=['📷 CAM-01'],
    textposition='top center',
    name='CAM-01 (Neighbour Bldg, 25ft)',
    hovertemplate=(
        '<b>CAM-01</b><br>'
        'Position: neighbour building<br>'
        f'Height: {H_CAMERA:.2f}m (25ft)<br>'
        f'Elevation above construction roof: {H_CAMERA - H_TOTAL:.2f}m'
        '<extra></extra>'
    )
))

# 9 ── CAMERA elevation line (vertical drop to ground)
traces.append(go.Scatter3d(
    x=[CAM_X, CAM_X], y=[CAM_Y, CAM_Y], z=[0, CAM_Z],
    mode='lines',
    line=dict(color='#2980B9', width=2, dash='dash'),
    name='Camera height reference',
    hoverinfo='skip', showlegend=False
))

# 10 ── HEIGHT REFERENCE LINES (building levels)
for z_level, label, color in [
    (H_BASEMENT, 'Basement top (1.52m)', '#95A5A6'),
    (H_GROUND,   'Ground floor slab (4.88m)', '#7F8C8D'),
    (H_TOTAL,    'Roof / parapet (6.10m)', '#2C3E50'),
    (H_CAMERA,   'Camera height (7.62m)', '#2980B9'),
]:
    traces.append(go.Scatter3d(
        x=[-3, BW+3], y=[BD+3, BD+3], z=[z_level, z_level],
        mode='lines+text',
        line=dict(color=color, width=1, dash='dot'),
        text=['', label],
        textposition='middle right',
        hoverinfo='skip',
        showlegend=False
    ))

# 11 ── GROUND PLANE
x_g = np.array([-3, BW+3, BW+3, -3, -3])
y_g = np.array([-10, -10, BD+3, BD+3, -10])
traces.append(go.Scatter3d(
    x=x_g, y=y_g, z=np.zeros(5),
    mode='lines',
    line=dict(color='#95A5A6', width=1),
    name='Ground (road level)',
    hoverinfo='skip', showlegend=False
))

# ════════════════════════════════════════════════════════
# LEGEND ANNOTATIONS
# ════════════════════════════════════════════════════════
annotations = [
    dict(x=BW/2, y=BD+5, z=H_TOTAL+1.5,
         text='<b>BuildSight — Maran Constructions</b><br>Thanjavur | 10°48\'59.7"N 78°40\'07.8"E',
         showarrow=False, font=dict(size=11, color='#2C3E50')),
]

# ════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════
fig = go.Figure(data=traces)

fig.update_layout(
    title=dict(
        text='BuildSight GeoAI — 3D Risk Zone Map<br>'
             '<sup>Maran Constructions G+1 Residential | Green Build AI | IGBC AP</sup>',
        font=dict(size=16, color='#2C3E50'),
        x=0.5
    ),
    scene=dict(
        xaxis=dict(title='East (m)', showgrid=True, gridcolor='#ECF0F1',
                   backgroundcolor='#F8F9FA'),
        yaxis=dict(title='North (m)', showgrid=True, gridcolor='#ECF0F1',
                   backgroundcolor='#F8F9FA'),
        zaxis=dict(title='Height (m)', showgrid=True, gridcolor='#ECF0F1',
                   backgroundcolor='#EBF5FB', range=[0, 10]),
        bgcolor='#FDFEFE',
        camera=dict(
            eye=dict(x=-1.8, y=-2.2, z=1.2),
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='manual',
        aspectratio=dict(x=2.0, y=1.8, z=0.8),
        annotations=annotations
    ),
    legend=dict(
        x=0.02, y=0.95,
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='#BDC3C7',
        borderwidth=1,
        font=dict(size=11)
    ),
    paper_bgcolor='white',
    margin=dict(l=0, r=0, t=80, b=0),
    height=750,
)

# ── Annotation box (risk legend) ──
fig.add_annotation(
    text=(
        '<b>Risk Level Legend</b><br>'
        '🔴 HIGH / CRITICAL — scaffolding + staircase<br>'
        '🟠 MODERATE / WARNING — interior work zone<br>'
        '🟢 LOW / ADVISORY — car parking area<br>'
        '📷 CAM-01 — neighbour building @ 25ft (7.62m)<br>'
        '─────────────────────────────<br>'
        'Building: 20ft (6.10m) | Site: 62\'×32\' (18.9×9.75m)<br>'
        'Ref: BOCW Act 1996 | NBC 2016 | IS 4130'
    ),
    xref='paper', yref='paper',
    x=0.99, y=0.01,
    xanchor='right', yanchor='bottom',
    align='left',
    bgcolor='rgba(255,255,255,0.90)',
    bordercolor='#BDC3C7',
    borderwidth=1,
    font=dict(size=10, color='#2C3E50'),
    showarrow=False
)

print("=" * 55)
print("  BuildSight 3D Risk Zone Visualization")
print("  Maran Constructions, Thanjavur")
print("=" * 55)
print("Opening in browser...")
print("  → Drag  : rotate view")
print("  → Scroll: zoom in/out")
print("  → Double-click: reset view")
print("  → Click legend items to show/hide zones")
print("=" * 55)

fig.show()

# Also save as standalone HTML (for presentation/offline use)
fig.write_html("buildsight_3d_zones.html")
print("\nSaved: buildsight_3d_zones.html")
print("→ Open this file in any browser for offline demo.")
