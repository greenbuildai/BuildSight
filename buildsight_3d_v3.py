"""
BuildSight — 3D Risk Zone Visualization v3
Maran Constructions, Thanjavur | Green Build AI | IGBC AP

CCTV perspective correction:
  - Camera SOUTH of building, looking NORTH (matches actual neighbour building position)
  - Initial view matches CCTV image angle (from SE elevated, looking NW)
  - Worker at SE corner at slab level (matches CCTV bottom-right position)
  - FOV rays project FROM south camera TOWARD north interior

Height-based zoning (verified from CCTV image):
  CRITICAL  4.88–6.10m  Roof slab / parapet edge (worker in CCTV is here)
  HIGH      3.05–4.88m  Upper floor near slab
  MODERATE  1.52–3.05m  Mid floor construction
  LOW       0.00–1.52m  Ground / basement level

Run: python buildsight_3d_v3.py
"""

import plotly.graph_objects as go
import numpy as np

# ─── Site dimensions ───
BW   = 18.90    # East-West  (62'-0")
BD   =  9.75    # North-South (32'-0")
CAR  =  2.74    # Car parking east strip

# ─── Vertical zone boundaries ───
Z0 = 0.000      # Road level
Z1 = 1.524      # Basement top   (5ft)
Z2 = 3.050      # Mid floor      (~10ft)
Z3 = 4.877      # Roof slab      (16ft)
Z4 = 6.096      # Parapet top    (20ft)

# ─── Camera (south neighbour building, SE of site) ───
CAM_X = BW * 0.65   # 12.29m east of SW corner
CAM_Y = -8.00       # 8m south of south wall
CAM_Z =  7.620      # 25ft above road level

# ─── Worker from CCTV image ───
# Bottom-right of frame = SE corner of building, at slab level
WORK_X = BW - 1.2   # 17.7m east (right side of building)
WORK_Y = 0.80        # 0.8m from south wall (bottom of frame)
WORK_Z = Z3 + 0.40  # 5.28m — standing on roof slab edge


# ════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════

def box_mesh(x0,y0,z0, x1,y1,z1, color, name, opacity=0.38, show_legend=True):
    vx=[x0,x1,x1,x0,x0,x1,x1,x0]
    vy=[y0,y0,y1,y1,y0,y0,y1,y1]
    vz=[z0,z0,z0,z0,z1,z1,z1,z1]
    ii=[0,0,0,0,4,4,2,2,1,1,3,3]
    jj=[1,2,3,5,5,6,6,7,5,6,7,6]
    kk=[2,3,7,4,6,7,7,3,2,2,6,2]
    return go.Mesh3d(
        x=vx,y=vy,z=vz,i=ii,j=jj,k=kk,
        color=color,opacity=opacity,name=name,
        flatshading=True,showscale=False,showlegend=show_legend,
        lighting=dict(ambient=0.72,diffuse=0.85,roughness=0.45,specular=0.25),
        lightposition=dict(x=CAM_X*10,y=CAM_Y*10,z=CAM_Z*10),
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Footprint: {abs(x1-x0):.1f}m × {abs(y1-y0):.1f}m<br>"
            f"Elevation: {z0:.2f}m → {z1:.2f}m ({z1-z0:.2f}m tall)"
            "<extra></extra>"
        )
    )

def box_edges(x0,y0,z0, x1,y1,z1, color, width=1.5):
    lx,ly,lz=[],[],[]
    for p1,p2 in [
        [(x0,y0,z0),(x1,y0,z0)],[(x1,y0,z0),(x1,y1,z0)],
        [(x1,y1,z0),(x0,y1,z0)],[(x0,y1,z0),(x0,y0,z0)],
        [(x0,y0,z1),(x1,y0,z1)],[(x1,y0,z1),(x1,y1,z1)],
        [(x1,y1,z1),(x0,y1,z1)],[(x0,y1,z1),(x0,y0,z1)],
        [(x0,y0,z0),(x0,y0,z1)],[(x1,y0,z0),(x1,y0,z1)],
        [(x1,y1,z0),(x1,y1,z1)],[(x0,y1,z0),(x0,y1,z1)],
    ]:
        lx+=[p1[0],p2[0],None]; ly+=[p1[1],p2[1],None]; lz+=[p1[2],p2[2],None]
    return go.Scatter3d(x=lx,y=ly,z=lz,mode='lines',
                        line=dict(color=color,width=width),
                        hoverinfo='skip',showlegend=False)

def fov_line(tx,ty,tz):
    return go.Scatter3d(
        x=[CAM_X,tx,None],y=[CAM_Y,ty,None],z=[CAM_Z,tz,None],
        mode='lines',line=dict(color='#1ABC9C',width=0.8,dash='dot'),
        hoverinfo='skip',showlegend=False)


# ════════════════════════════════════════════════════════
# BUILD TRACES
# ════════════════════════════════════════════════════════
traces = []

# ── Ground plane ──
gx=np.array([-3,BW+3,BW+3,-3,-3])
gy=np.array([-12,-12,BD+4,BD+4,-12])
traces.append(go.Scatter3d(
    x=gx,y=gy,z=np.zeros(5),mode='lines',
    line=dict(color='#AAB7B8',width=1),
    hoverinfo='skip',showlegend=False))

# ══════════════════════════════════════════════
# HEIGHT-BASED RISK ZONES (stacked vertically)
# LOW at bottom → CRITICAL at top
# ══════════════════════════════════════════════

# LOW (0 → 1.52m) — ground level
traces.append(box_mesh(0,0,Z0, BW,BD,Z1,
    '#27AE60','🟢 LOW  0–1.52m  (ground level)',0.22))
traces.append(box_edges(0,0,Z0,BW,BD,Z1,'#1E8449',1.0))

# MODERATE (1.52 → 3.05m) — mid floor
traces.append(box_mesh(0,0,Z1, BW,BD,Z2,
    '#F4D03F','🟡 MODERATE  1.52–3.05m  (mid floor)',0.28))
traces.append(box_edges(0,0,Z1,BW,BD,Z2,'#D4AC0D',1.0))

# HIGH (3.05 → 4.88m) — upper floor near slab
traces.append(box_mesh(0,0,Z2, BW,BD,Z3,
    '#E67E22','🟠 HIGH  3.05–4.88m  (upper floor)',0.35))
traces.append(box_edges(0,0,Z2,BW,BD,Z3,'#CA6F1E',1.5))

# CRITICAL (4.88 → 6.10m) — roof slab / parapet edge
traces.append(box_mesh(0,0,Z3, BW,BD,Z4,
    '#C0392B','🔴 CRITICAL  4.88–6.10m  (roof/parapet)',0.52))
traces.append(box_edges(0,0,Z3,BW,BD,Z4,'#7B241C',2.5))

# ══════════════════════════════════════════════
# HORIZONTAL PERIMETER OVERLAYS
# Scaffolding ring — HIGH/CRITICAL at all levels
# ══════════════════════════════════════════════
SCAF = 2.0  # scaffold band width outside wall

for label, x0,y0,x1,y1 in [
    ('N wall', -SCAF, BD,    BW+SCAF, BD+SCAF),
    ('S wall', -SCAF, -SCAF, BW+SCAF, 0      ),
    ('W wall', -SCAF, -SCAF, 0,       BD+SCAF),
    ('E wall', BW,   -SCAF,  BW+SCAF, BD+SCAF),
]:
    for z_lo,z_hi,col,op in [
        (Z0,Z2,'#A93226',0.12),(Z2,Z3,'#A93226',0.18),(Z3,Z4,'#7B241C',0.28)]:
        traces.append(box_mesh(x0,y0,z_lo,x1,y1,z_hi,
                               col,f'Scaffolding ({label})',op,show_legend=False))

# Scaffolding outer boundary lines
traces.append(box_edges(-SCAF,-SCAF,Z0,BW+SCAF,BD+SCAF,Z4,'#E74C3C',2.0))

# ── Staircase NW — always CRITICAL ──
traces.append(box_mesh(0,BD-2.3,Z0, 4.88,BD,Z4,
    '#6E2222','🔴 CRITICAL — staircase NW (fall risk)',0.65))
traces.append(box_edges(0,BD-2.3,Z0,4.88,BD,Z4,'#FF1A00',2.5))

# ── Car parking E strip — LOW ──
traces.append(box_mesh(BW-CAR,0,Z0,BW,BD,Z4,
    '#1A5276','🔵 LOW — car parking (E)',0.18))
traces.append(box_edges(BW-CAR,0,Z0,BW,BD,Z4,'#1F618D',1.0))

# ══════════════════════════════════════════════
# RCC COLUMNS (visible in CCTV image)
# ══════════════════════════════════════════════
col_pts = [
    (0,0),(BW/3,0),(BW*2/3,0),(BW,0),
    (0,BD/2),(BW/3,BD/2),(BW*2/3,BD/2),(BW,BD/2),
    (0,BD),(BW/3,BD),(BW*2/3,BD),(BW,BD),
]
for cx,cy in col_pts:
    traces.append(box_mesh(cx-0.15,cy-0.15,Z0,cx+0.15,cy+0.15,Z4,
        '#566573','RCC Columns',0.85,show_legend=False))

# ══════════════════════════════════════════════
# NEIGHBOUR BUILDING (south, camera mounted here)
# ══════════════════════════════════════════════
traces.append(box_mesh(3,-12,Z0, BW-1,-1,Z3,
    '#D5D8DC','Neighbour building (camera host)',0.12))
traces.append(box_edges(3,-12,Z0,BW-1,-1,Z3,'#909497',0.8))

# ══════════════════════════════════════════════
# CAM-01 — correct south position, looking NORTH
# ══════════════════════════════════════════════

# Camera marker
traces.append(go.Scatter3d(
    x=[CAM_X],y=[CAM_Y],z=[CAM_Z],
    mode='markers+text',
    marker=dict(size=14,color='#2471A3',symbol='diamond',
                line=dict(color='white',width=2)),
    text=['📷 CAM-01'],
    textposition='top center',
    textfont=dict(size=11,color='#154360'),
    name='📷 CAM-01 (25ft, looking NORTH)',
    hovertemplate=(
        '<b>CAM-01 — Neighbour Building</b><br>'
        f'Real position: ({CAM_X:.1f}m E, {abs(CAM_Y):.0f}m S of site wall)<br>'
        f'Height: {CAM_Z:.2f}m (25ft above road level)<br>'
        f'Above building roof: +{CAM_Z-Z4:.2f}m<br>'
        'Direction: looking NORTH into construction site<br>'
        'Coverage: full site top-down angled view'
        '<extra></extra>'
    )
))

# Camera vertical pole
traces.append(go.Scatter3d(
    x=[CAM_X,CAM_X],y=[CAM_Y,CAM_Y],z=[Z0,CAM_Z],
    mode='lines',line=dict(color='#2471A3',width=2,dash='dash'),
    hoverinfo='skip',showlegend=False))

# FOV rays FROM camera TO site (south→north direction, correct)
fov_pts = [
    # Ground corners
    (0,   0,   Z0), (BW,  0,   Z0),
    (BW,  BD,  Z0), (0,   BD,  Z0),
    # Slab corners (top of building)
    (0,   0,   Z4), (BW,  0,   Z4),
    (BW,  BD,  Z4), (0,   BD,  Z4),
    # Center of site at mid height
    (BW/2,BD/2,Z2),
    # Worker position
    (WORK_X,WORK_Y,WORK_Z),
]
fov_lx,fov_ly,fov_lz=[],[],[]
for tx,ty,tz in fov_pts:
    fov_lx+=[CAM_X,tx,None]
    fov_ly+=[CAM_Y,ty,None]
    fov_lz+=[CAM_Z,tz,None]
traces.append(go.Scatter3d(
    x=fov_lx,y=fov_ly,z=fov_lz,mode='lines',
    line=dict(color='#1ABC9C',width=0.8,dash='dot'),
    name='Camera FOV rays (→ NORTH)',opacity=0.6,
    hoverinfo='skip'))

# FOV cone surface (from camera looking north into site)
cone_vx=[CAM_X,-2,-2,BW+2,BW+2]
cone_vy=[CAM_Y, 0, BD, BD,  0  ]
cone_vz=[CAM_Z, 0,  0,  0,  0  ]
traces.append(go.Mesh3d(
    x=cone_vx,y=cone_vy,z=cone_vz,
    i=[0,0,0,0],j=[1,2,3,4],k=[2,3,4,1],
    color='#1ABC9C',opacity=0.07,
    name='Camera FOV coverage',showlegend=False,
    hoverinfo='skip',showscale=False))

# ══════════════════════════════════════════════
# WORKER MARKER (SE corner, slab level — matches CCTV bottom-right)
# ══════════════════════════════════════════════
traces.append(go.Scatter3d(
    x=[WORK_X],y=[WORK_Y],z=[WORK_Z],
    mode='markers+text',
    marker=dict(size=12,color='#F1C40F',symbol='circle',
                line=dict(color='#E74C3C',width=3)),
    text=['⚠ Worker'],
    textposition='top center',
    textfont=dict(size=11,color='#7B241C'),
    name='⚠ Worker — CRITICAL zone (CCTV ref)',
    hovertemplate=(
        '<b>⚠ CRITICAL ALERT — Worker Detected</b><br>'
        'Location: SE corner, south wall edge<br>'
        f'Position: {WORK_X:.1f}m E  |  {WORK_Y:.1f}m from S wall<br>'
        f'Elevation: {WORK_Z:.2f}m (roof slab level)<br>'
        'Zone: CRITICAL — parapet edge exposure<br>'
        'PPE from CCTV: helmet ✅  safety vest ✅<br>'
        'Risk: FALL HAZARD — unguarded slab edge<br>'
        'Alert: 🔴 CRITICAL triggered → BOCW §40'
        '<extra></extra>'
    )
))

# Worker drop line to ground
traces.append(go.Scatter3d(
    x=[WORK_X,WORK_X],y=[WORK_Y,WORK_Y],z=[Z0,WORK_Z],
    mode='lines',line=dict(color='#F1C40F',width=1.5,dash='dot'),
    hoverinfo='skip',showlegend=False))

# ══════════════════════════════════════════════
# HEIGHT REFERENCE BANDS
# ══════════════════════════════════════════════
ref_levels = [
    (Z0,  'Road level   0.0m', '#95A5A6'),
    (Z1,  'Basement top 1.5m', '#7F8C8D'),
    (Z2,  'Mid floor    3.1m', '#E67E22'),
    (Z3,  'Roof slab    4.9m', '#C0392B'),
    (Z4,  'Parapet      6.1m', '#922B21'),
    (CAM_Z,'Camera      7.6m', '#2471A3'),
]
for z_ref,label,col in ref_levels:
    traces.append(go.Scatter3d(
        x=[-3,BW+4],y=[BD+3.5,BD+3.5],z=[z_ref,z_ref],
        mode='lines+text',
        line=dict(color=col,width=1,dash='dot'),
        text=['',label],textposition='middle right',
        textfont=dict(size=9,color=col),
        hoverinfo='skip',showlegend=False))


# ════════════════════════════════════════════════════════
# FIGURE + LAYOUT
# ════════════════════════════════════════════════════════
fig = go.Figure(data=traces)

fig.update_layout(
    title=dict(
        text=(
            'BuildSight GeoAI — Height-Based 3D Risk Zone Map  |  v3<br>'
            '<sup>Maran Constructions G+1 | Thanjavur | Green Build AI | '
            'IGBC AP | NBC 2016 | BOCW Act 1996</sup>'
        ),
        font=dict(size=14,color='#1A252F'),x=0.5
    ),
    scene=dict(
        xaxis=dict(
            title='East → (m)',
            showgrid=True,gridcolor='#D5D8DC',
            backgroundcolor='#F4F6F7',
            range=[-3,BW+5]
        ),
        yaxis=dict(
            title='North → (m)',
            showgrid=True,gridcolor='#D5D8DC',
            backgroundcolor='#EBF5FB',
            range=[-12,BD+5]
        ),
        zaxis=dict(
            title='Height (m)',
            showgrid=True,gridcolor='#D5D8DC',
            backgroundcolor='#FEF9E7',
            range=[0,10]
        ),
        bgcolor='#FDFEFE',

        # ── KEY: Eye position matches CCTV perspective ──
        # CCTV camera is SOUTH-EAST of site, elevated, looking NORTH-WEST
        # Plotly eye must be at positive X, negative Y, positive Z
        # to render the initial view matching the CCTV image:
        #   - South wall visible at front/bottom
        #   - North wall at rear/top
        #   - Worker at front-right (SE) ✓
        #   - Staircase at rear-left (NW) ✓
        camera=dict(
            eye=dict(x=1.05, y=-2.50, z=1.10),
            center=dict(x=-0.05, y=0.15, z=-0.10),
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='manual',
        aspectratio=dict(x=2.2, y=2.0, z=0.80),
    ),
    legend=dict(
        x=0.01,y=0.99,
        bgcolor='rgba(255,255,255,0.90)',
        bordercolor='#BDC3C7',borderwidth=1,
        font=dict(size=10),tracegroupgap=1
    ),
    paper_bgcolor='white',
    margin=dict(l=0,r=0,t=85,b=0),
    height=820,
)

# ── Risk legend box ──
fig.add_annotation(
    text=(
        '<b>Risk Zones — Height Based</b><br>'
        '🔴 CRITICAL  4.88–6.10m  Roof slab / parapet edge<br>'
        '🟠 HIGH      3.05–4.88m  Upper floor near slab<br>'
        '🟡 MODERATE  1.52–3.05m  Mid floor construction<br>'
        '🟢 LOW       0.00–1.52m  Ground / basement level<br>'
        '─────────────────────────────────<br>'
        '<b>Horizontal overlays</b><br>'
        '🔴 Scaffolding perimeter = HIGH→CRITICAL<br>'
        '🔴 Staircase NW corner = always CRITICAL<br>'
        '🔵 Car parking E strip = LOW<br>'
        '─────────────────────────────────<br>'
        '<b>Camera</b><br>'
        '📷 CAM-01: neighbour bldg, 25ft, looking NORTH<br>'
        '⚠  Worker: SE corner slab edge (CCTV 2025-12-29)<br>'
        '─────────────────────────────────<br>'
        'NBC 2016 | BOCW Act 1996 | IS 4130 | IS 3521'
    ),
    xref='paper',yref='paper',
    x=0.99,y=0.01,xanchor='right',yanchor='bottom',
    align='left',
    bgcolor='rgba(255,255,255,0.93)',
    bordercolor='#2C3E50',borderwidth=1,
    font=dict(size=9,color='#1A252F'),
    showarrow=False
)

# ── CCTV orientation note ──
fig.add_annotation(
    text=(
        '↑ View matches CCTV perspective<br>'
        'Camera SOUTH → looking NORTH<br>'
        'Worker at SE corner (front-right)'
    ),
    xref='paper',yref='paper',
    x=0.01,y=0.01,xanchor='left',yanchor='bottom',
    align='left',
    bgcolor='rgba(26,146,175,0.12)',
    bordercolor='#2471A3',borderwidth=1,
    font=dict(size=9,color='#154360'),
    showarrow=False
)

print("=" * 60)
print("  BuildSight 3D Risk Zone Map  v3")
print("  Camera perspective corrected to match CCTV image")
print("=" * 60)
print()
print("  CCTV orientation:")
print("    Camera:  SOUTH of site  →  looking NORTH ✓")
print("    Worker:  SE corner, slab level (bottom-right of CCTV) ✓")
print("    Staircase: NW corner (top-left of CCTV) ✓")
print()
print("  Height zones:")
print("    CRITICAL  4.88→6.10m  roof/parapet edge")
print("    HIGH      3.05→4.88m  upper floor")
print("    MODERATE  1.52→3.05m  mid floor")
print("    LOW       0.00→1.52m  ground level")
print()
print("  Controls:")
print("    Drag          → rotate")
print("    Scroll        → zoom")
print("    Double-click  → reset to CCTV perspective")
print("    Legend items  → toggle zones on/off")
print()
print("Opening in browser...")

fig.show()
fig.write_html("buildsight_3d_v3.html")
print("Saved: buildsight_3d_v3.html")
