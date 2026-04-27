"""
BuildSight — 3D Risk Zone Visualization v2
Maran Constructions, Thanjavur | Green Build AI | IGBC AP

Key fix: Height-based risk zoning (as seen in CCTV image)
  CRITICAL : Roof slab / parapet edge (4.88m → 6.10m) — fall risk
  HIGH     : Upper first floor near slab (3.05m → 4.88m)
  MODERATE : Mid first floor construction (1.52m → 3.05m)
  LOW      : Ground level / basement (0.00m → 1.52m)

Camera: Neighbour building, south-east, 7.62m height, looking NW into site.

Run: python buildsight_3d_v2.py
"""

import plotly.graph_objects as go
import numpy as np

# ─── Site dimensions (meters, origin = SW corner) ───
BW = 18.90      # East-West  (62'-0")
BD =  9.75      # North-South (32'-0")
CAR = 2.74      # Car parking east strip

# ─── Vertical zone boundaries ───
Z0 = 0.000      # Road level
Z1 = 1.524      # Basement top (5ft)
Z2 = 3.050      # ~10ft — mid first floor
Z3 = 4.877      # Ground floor slab (16ft)
Z4 = 6.096      # Building top / parapet (20ft)

# ─── Camera ───
CAM_X = 12.29   # East of SW corner (slightly east of center)
CAM_Y = -8.00   # 8m south of south wall
CAM_Z =  7.620  # 25ft above road level


# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════

def box_mesh(x0,y0,z0, x1,y1,z1, color, name, opacity=0.40):
    vx=[x0,x1,x1,x0,x0,x1,x1,x0]
    vy=[y0,y0,y1,y1,y0,y0,y1,y1]
    vz=[z0,z0,z0,z0,z1,z1,z1,z1]
    i=[0,0,0,0,4,4,2,2,1,1,3,3]
    j=[1,2,3,5,5,6,6,7,5,6,7,6]
    k=[2,3,7,4,6,7,7,3,2,2,6,2]
    return go.Mesh3d(
        x=vx,y=vy,z=vz,i=i,j=j,k=k,
        color=color,opacity=opacity,name=name,
        flatshading=True,showscale=False,
        lighting=dict(ambient=0.75,diffuse=0.8,roughness=0.4,specular=0.2),
        lightposition=dict(x=50,y=100,z=200),
        hovertemplate=f"<b>{name}</b><br>"
                      f"Zone: {x1-x0:.1f}m × {y1-y0:.1f}m × {z1-z0:.2f}m tall<br>"
                      f"Elevation: {z0:.2f}m → {z1:.2f}m<extra></extra>"
    )

def box_edges(x0,y0,z0, x1,y1,z1, color, width=1.5):
    lx,ly,lz=[],[],[]
    edges=[
        [(x0,y0,z0),(x1,y0,z0)],[(x1,y0,z0),(x1,y1,z0)],
        [(x1,y1,z0),(x0,y1,z0)],[(x0,y1,z0),(x0,y0,z0)],
        [(x0,y0,z1),(x1,y0,z1)],[(x1,y0,z1),(x1,y1,z1)],
        [(x1,y1,z1),(x0,y1,z1)],[(x0,y1,z1),(x0,y0,z1)],
        [(x0,y0,z0),(x0,y0,z1)],[(x1,y0,z0),(x1,y0,z1)],
        [(x1,y1,z0),(x1,y1,z1)],[(x0,y1,z0),(x0,y1,z1)],
    ]
    for p1,p2 in edges:
        lx+=[p1[0],p2[0],None]
        ly+=[p1[1],p2[1],None]
        lz+=[p1[2],p2[2],None]
    return go.Scatter3d(x=lx,y=ly,z=lz,mode='lines',
                        line=dict(color=color,width=width),
                        hoverinfo='skip',showlegend=False)

def horiz_band_label(x, y, z, text, color):
    """Floating text label for height zone."""
    return go.Scatter3d(
        x=[x],y=[y],z=[z],mode='text',
        text=[text],textposition='middle right',
        textfont=dict(size=9,color=color),
        hoverinfo='skip',showlegend=False
    )


# ════════════════════════════════════════════════════════
# TRACES
# ════════════════════════════════════════════════════════
traces = []

# ── Ground plane ──
gx=[-3,BW+3,BW+3,-3,-3]
gy=[-12,-12,BD+4,BD+4,-12]
traces.append(go.Scatter3d(
    x=gx,y=gy,z=[0]*5,mode='lines',
    line=dict(color='#BDC3C7',width=1),
    hoverinfo='skip',showlegend=False,name='Ground'))

# ══════════════════════════════════════════════════════
# HEIGHT-BASED RISK ZONES (full building footprint)
# These stack vertically — lower = safer, top = critical
# ══════════════════════════════════════════════════════

# ZONE 1 — LOW RISK: Ground level / basement (0 → 1.52m)
# Full footprint at ground level — people just walking
traces.append(box_mesh(
    0,0,Z0, BW,BD,Z1,
    color='#27AE60', name='LOW — ground level (0→1.52m)',
    opacity=0.25))
traces.append(box_edges(0,0,Z0,BW,BD,Z1,'#1E8449',width=1))

# ZONE 2 — MODERATE RISK: Mid first floor (1.52m → 3.05m)
# Interior construction, masonry work
traces.append(box_mesh(
    0,0,Z1, BW,BD,Z2,
    color='#F39C12', name='MODERATE — mid floor (1.52→3.05m)',
    opacity=0.30))
traces.append(box_edges(0,0,Z1,BW,BD,Z2,'#D68910',width=1))

# ZONE 3 — HIGH RISK: Upper first floor / near slab (3.05m → 4.88m)
# Formwork, reinforcement, concrete pouring
traces.append(box_mesh(
    0,0,Z2, BW,BD,Z3,
    color='#E67E22', name='HIGH — upper floor (3.05→4.88m)',
    opacity=0.35))
traces.append(box_edges(0,0,Z2,BW,BD,Z3,'#CA6F1E',width=1.5))

# ZONE 4 — CRITICAL: Roof slab + parapet edge (4.88m → 6.10m)
# Exactly what the worker in the CCTV image is doing — top of wall, edge work
traces.append(box_mesh(
    0,0,Z3, BW,BD,Z4,
    color='#C0392B', name='CRITICAL — roof/parapet edge (4.88→6.10m)',
    opacity=0.50))
traces.append(box_edges(0,0,Z3,BW,BD,Z4,'#922B21',width=2))

# ══════════════════════════════════════════════════════
# HORIZONTAL RISK OVERLAYS (plan-level, from previous GIS)
# Scaffolding perimeter ring (always HIGH regardless of height)
# ══════════════════════════════════════════════════════

# Scaffolding outer perimeter band (2m outside, at all heights)
for z_lo, z_hi, col, op in [
    (Z0,Z1,'#922B21',0.12),
    (Z1,Z2,'#922B21',0.15),
    (Z2,Z3,'#922B21',0.20),
    (Z3,Z4,'#7B241C',0.30),
]:
    # North wall strip
    traces.append(box_mesh(-2,BD,z_lo, BW+2,BD+2,z_hi,
                           col,'Scaffolding N',op))
    # South wall strip
    traces.append(box_mesh(-2,-2,z_lo, BW+2,0,z_hi,
                           col,'Scaffolding S',op))
    # West wall strip
    traces.append(box_mesh(-2,-2,z_lo, 0,BD+2,z_hi,
                           col,'Scaffolding W',op))
    # East wall strip
    traces.append(box_mesh(BW,-2,z_lo, BW+2,BD+2,z_hi,
                           col,'Scaffolding E',op))

# Scaffolding perimeter edges (full height)
traces.append(box_edges(-2,-2,Z0,BW+2,BD+2,Z4,'#E74C3C',width=2))

# Staircase zone — always CRITICAL (NW corner)
traces.append(box_mesh(0,BD-2.3,Z0, 4.88,BD,Z4,
                       '#8B0000','CRITICAL — staircase (NW)',0.60))
traces.append(box_edges(0,BD-2.3,Z0,4.88,BD,Z4,'#FF2200',width=2.5))

# Car parking — LOW throughout
traces.append(box_mesh(BW-CAR,0,Z0, BW,BD,Z4,
                       '#1A5276','LOW — car parking',0.20))
traces.append(box_edges(BW-CAR,0,Z0,BW,BD,Z4,'#1F618D',width=1))

# ══════════════════════════════════════════════════════
# BUILDING STRUCTURAL ELEMENTS (visual reference)
# ══════════════════════════════════════════════════════

# Columns at corners and intermediate points (RCC columns visible in CCTV)
col_positions = [
    (0,0),(BW/3,0),(BW*2/3,0),(BW,0),
    (0,BD/2),(BW,BD/2),
    (0,BD),(BW/3,BD),(BW*2/3,BD),(BW,BD),
    (BW/3,BD/2),(BW*2/3,BD/2),
]
for cx,cy in col_positions:
    traces.append(box_mesh(cx-0.15,cy-0.15,Z0, cx+0.15,cy+0.15,Z4,
                           '#566573','Column',0.80))

# ══════════════════════════════════════════════════════
# CAMERA — correct placement (SE of site, in neighbour bldg)
# ══════════════════════════════════════════════════════

# Neighbour building footprint (south, partial)
traces.append(box_mesh(5,-12,Z0, BW-2,-1,Z3,
                       '#AEB6BF','Neighbour building',0.15))
traces.append(box_edges(5,-12,Z0,BW-2,-1,Z3,'#808B96',width=1))

# Camera position marker
traces.append(go.Scatter3d(
    x=[CAM_X],y=[CAM_Y],z=[CAM_Z],
    mode='markers+text',
    marker=dict(size=14,color='#2471A3',symbol='diamond',
                line=dict(color='white',width=2)),
    text=['📷 CAM-01'],
    textposition='top center',
    textfont=dict(size=11,color='#1A5276'),
    name='CAM-01 (25ft / 7.62m)',
    hovertemplate=(
        '<b>CAM-01 — Neighbour Building</b><br>'
        f'Position: ({CAM_X:.1f}m E, {abs(CAM_Y):.0f}m S of site)<br>'
        f'Height: {CAM_Z:.2f}m (25ft above road level)<br>'
        f'Elevation above building roof: {CAM_Z-Z4:.2f}m<br>'
        'Coverage: Full site top-down angled view<extra></extra>'
    )
))

# Camera vertical reference line
traces.append(go.Scatter3d(
    x=[CAM_X,CAM_X],y=[CAM_Y,CAM_Y],z=[0,CAM_Z],
    mode='lines',line=dict(color='#2471A3',width=2,dash='dash'),
    hoverinfo='skip',showlegend=False))

# Camera FOV lines to site corners + center
fov_targets = [
    (0,0,Z0),(BW,0,Z0),(BW,BD,Z0),(0,BD,Z0),   # ground corners
    (BW/2,BD/2,Z3),                              # center at roof slab
    (0,BD,Z4),(BW,BD,Z4),                        # far corners at roof
]
lx,ly,lz=[],[],[]
for tx,ty,tz in fov_targets:
    lx+=[CAM_X,tx,None]; ly+=[CAM_Y,ty,None]; lz+=[CAM_Z,tz,None]
traces.append(go.Scatter3d(
    x=lx,y=ly,z=lz,mode='lines',
    line=dict(color='#1D9E75',width=1,dash='dot'),
    name='Camera FOV rays',opacity=0.5,
    hoverinfo='skip'))

# FOV coverage cone (filled)
cone_x=[CAM_X]+[-2,-2,BW+2,BW+2,CAM_X]
cone_y=[CAM_Y]+[-2,BD+2,BD+2,-2,CAM_Y]
cone_z=[CAM_Z]+[0,0,0,0,CAM_Z]
traces.append(go.Mesh3d(
    x=cone_x,y=cone_y,z=cone_z,
    i=[0,0,0,0],j=[1,2,3,4],k=[2,3,4,1],
    color='#1ABC9C',opacity=0.06,
    name='Camera FOV coverage',
    hoverinfo='skip',showscale=False))

# ══════════════════════════════════════════════════════
# HEIGHT REFERENCE LEVELS (horizontal dotted planes)
# ══════════════════════════════════════════════════════
level_refs = [
    (Z0, 'Road level  0.0m',  '#95A5A6'),
    (Z1, 'Basement top 1.5m', '#7F8C8D'),
    (Z2, 'Mid floor   3.1m',  '#E67E22'),
    (Z3, 'Roof slab   4.9m',  '#C0392B'),
    (Z4, 'Parapet     6.1m',  '#922B21'),
    (CAM_Z,'Camera    7.6m',  '#2471A3'),
]
for z_ref, label, col in level_refs:
    traces.append(go.Scatter3d(
        x=[-3,BW+3],y=[BD+5,BD+5],z=[z_ref,z_ref],
        mode='lines+text',
        line=dict(color=col,width=1,dash='dot'),
        text=['',label],textposition='middle right',
        textfont=dict(size=9,color=col),
        hoverinfo='skip',showlegend=False))

# ══════════════════════════════════════════════════════
# WORKER ANNOTATION (from CCTV image — roof edge right side)
# Shows exactly where the worker in the photo is
# ══════════════════════════════════════════════════════
worker_x = BW - 1.5   # right side of building
worker_y = BD - 0.5   # near north wall (far side from camera)
worker_z = Z3 + 0.9   # at roof slab level, bending down

traces.append(go.Scatter3d(
    x=[worker_x],y=[worker_y],z=[worker_z],
    mode='markers+text',
    marker=dict(size=10,color='#F1C40F',symbol='circle',
                line=dict(color='#E74C3C',width=3)),
    text=['⚠ Worker (CCTV)'],
    textposition='top center',
    textfont=dict(size=10,color='#E74C3C'),
    name='Worker detected — CRITICAL zone',
    hovertemplate=(
        '<b>⚠ Worker — CRITICAL ZONE</b><br>'
        'Working on roof slab edge<br>'
        f'Elevation: ~{worker_z:.1f}m (roof level)<br>'
        'PPE: helmet ✅ vest ✅<br>'
        'Fall risk: HIGH — edge exposure<br>'
        'Alert: CRITICAL triggered<extra></extra>'
    )
))

# Worker vertical reference
traces.append(go.Scatter3d(
    x=[worker_x,worker_x],y=[worker_y,worker_y],z=[Z0,worker_z],
    mode='lines',line=dict(color='#F1C40F',width=1,dash='dot'),
    hoverinfo='skip',showlegend=False))


# ════════════════════════════════════════════════════════
# FIGURE LAYOUT
# ════════════════════════════════════════════════════════
fig = go.Figure(data=traces)

fig.update_layout(
    title=dict(
        text=(
            'BuildSight GeoAI — Height-Based 3D Risk Zone Map<br>'
            '<sup>Maran Constructions G+1 Residential | Thanjavur | '
            'Green Build AI | IGBC AP-Certified | NBC 2016 | BOCW Act 1996</sup>'
        ),
        font=dict(size=15,color='#1A252F'),x=0.5
    ),
    scene=dict(
        xaxis=dict(title='East →  (m)',showgrid=True,
                   gridcolor='#D5D8DC',backgroundcolor='#F2F3F4'),
        yaxis=dict(title='North →  (m)',showgrid=True,
                   gridcolor='#D5D8DC',backgroundcolor='#EBF5FB'),
        zaxis=dict(title='Height (m)',showgrid=True,
                   gridcolor='#D5D8DC',backgroundcolor='#FEF9E7',
                   range=[0,10]),
        bgcolor='#FDFEFE',
        # Camera angle matching actual CCTV perspective
        # (from south-east, elevated, looking NW down into site)
        camera=dict(
            eye=dict(x=1.6,y=-2.5,z=1.4),
            center=dict(x=0,y=0.2,z=-0.1),
            up=dict(x=0,y=0,z=1)
        ),
        aspectmode='manual',
        aspectratio=dict(x=2.2,y=2.0,z=0.85),
    ),
    legend=dict(
        x=0.01,y=0.99,
        bgcolor='rgba(255,255,255,0.88)',
        bordercolor='#BDC3C7',borderwidth=1,
        font=dict(size=10),
        tracegroupgap=2
    ),
    paper_bgcolor='white',
    margin=dict(l=0,r=0,t=90,b=0),
    height=800,
)

# ── Risk legend annotation ──
fig.add_annotation(
    text=(
        '<b>Height-Based Risk Zones</b><br>'
        '🔴 CRITICAL  4.88–6.10m  Roof slab / parapet edge<br>'
        '🟠 HIGH      3.05–4.88m  Upper floor near slab<br>'
        '🟡 MODERATE  1.52–3.05m  Mid floor construction<br>'
        '🟢 LOW       0.00–1.52m  Ground / basement level<br>'
        '──────────────────────────────────<br>'
        '⚠ Worker shown from CCTV (2025-12-29)<br>'
        '📷 CAM-01: neighbour bldg, 25ft, looking NW<br>'
        '──────────────────────────────────<br>'
        'Scaffolding perimeter = always HIGH→CRITICAL<br>'
        'Staircase (NW) = always CRITICAL (fall risk)<br>'
        '──────────────────────────────────<br>'
        'Ref: BOCW Act 1996 | NBC 2016 | IS 4130 | IS 3521'
    ),
    xref='paper',yref='paper',
    x=0.99,y=0.01,xanchor='right',yanchor='bottom',
    align='left',
    bgcolor='rgba(255,255,255,0.92)',
    bordercolor='#2C3E50',borderwidth=1,
    font=dict(size=9.5,color='#1A252F'),
    showarrow=False
)

print("=" * 58)
print("  BuildSight 3D Risk Zone Visualization v2")
print("  Height-based zoning | Correct camera placement")
print("  Maran Constructions, Thanjavur")
print("=" * 58)
print("\nZone boundaries (from CCTV image analysis):")
print("  CRITICAL  4.88m → 6.10m  (roof slab + parapet edge)")
print("  HIGH      3.05m → 4.88m  (upper floor construction)")
print("  MODERATE  1.52m → 3.05m  (mid floor work)")
print("  LOW       0.00m → 1.52m  (ground level)")
print("\nCamera: SE position, 7.62m, looking NW into site")
print("Worker from CCTV marked at roof slab level (CRITICAL)")
print("\nOpening in browser...")
print("  → Drag   : rotate")
print("  → Scroll : zoom")
print("  → Legend : click to toggle zones")
print("=" * 58)

fig.show()
fig.write_html("buildsight_3d_v2.html")
print("\nSaved: buildsight_3d_v2.html  (open in browser for demo)")
EOF
