"""
BuildSight GeoAI — Module 3: Homography Calibration Tool
=========================================================
Maran Constructions, Thanjavur | Green Build AI | IGBC AP

PURPOSE:
  Maps CCTV pixel coordinates → real-world site coordinates (meters).
  This is the BRIDGE between YOLO detections and the GIS zone system.

USAGE:
  pip install opencv-python numpy
  python module3_homography_calibrate.py --source rtsp://camera_url
  python module3_homography_calibrate.py --source video.mp4
  python module3_homography_calibrate.py --source frame.jpg
  python module3_homography_calibrate.py --test camera_cam01_H.npy
"""

import cv2, numpy as np, argparse, json, os
from datetime import datetime

# Real-world corner coordinates (meters from SW corner)
WORLD_POINTS = {
    "SW": (0.00,  0.00),
    "SE": (18.90, 0.00),
    "NE": (18.90, 9.75),
    "NW": (0.00,  9.75),
}
POINT_COLORS = {
    "SW": (0,255,0), "SE": (255,165,0),
    "NE": (0,0,255), "NW": (255,0,255),
}
CLICK_ORDER = ["SW","SE","NE","NW"]


class HomographyCalibrator:
    def __init__(self, source, camera_id="CAM-01"):
        self.source    = source
        self.camera_id = camera_id
        self.frame     = None
        self.display   = None
        self.clicked   = {}
        self.H         = None
        self.mean_error= 999.0
        self.H_path    = f"camera_{camera_id.replace('-','').lower()}_H.npy"
        self.meta_path = f"camera_{camera_id.replace('-','').lower()}_meta.json"

    def load_frame(self):
        ext = os.path.splitext(str(self.source))[1].lower()
        if ext in (".jpg",".jpeg",".png",".bmp"):
            self.frame = cv2.imread(self.source)
        else:
            cap = cv2.VideoCapture(self.source)
            for _ in range(15):
                ret, frame = cap.read()
            cap.release()
            self.frame = frame
        if self.frame is None:
            raise RuntimeError(f"Cannot read source: {self.source}")
        print(f"Frame: {self.frame.shape[1]}×{self.frame.shape[0]}px")
        self.display = self.frame.copy()

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        remaining = [p for p in CLICK_ORDER if p not in self.clicked]
        if not remaining:
            return
        label = remaining[0]
        self.clicked[label] = (x, y)
        col = POINT_COLORS[label]
        wx, wy = WORLD_POINTS[label]
        cv2.drawMarker(self.display,(x,y),col,cv2.MARKER_CROSS,22,2)
        cv2.circle(self.display,(x,y),9,col,2)
        cv2.putText(self.display,f"{label}({wx:.0f},{wy:.0f}m)",
                    (x+14,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)
        print(f"  ✓ {label}: pixel({x},{y}) → world({wx:.2f}m,{wy:.2f}m)")
        if len(self.clicked)==4:
            self._compute()
            self._overlay_grid()

    def _compute(self):
        px_pts = np.float32([self.clicked[k] for k in CLICK_ORDER])
        wd_pts = np.float32([WORLD_POINTS[k] for k in CLICK_ORDER])
        self.H, _ = cv2.findHomography(px_pts, wd_pts, cv2.RANSAC, 5.0)
        errors=[]
        for k in CLICK_ORDER:
            px,py = self.clicked[k]
            gx,gy = WORLD_POINTS[k]
            out   = cv2.perspectiveTransform(
                        np.array([[[float(px),float(py)]]],dtype=np.float32),self.H)
            ex,ey = out[0][0]
            err   = np.sqrt((ex-gx)**2+(ey-gy)**2)
            errors.append(err)
        self.mean_error = float(np.mean(errors))
        status = "✅ GOOD" if self.mean_error<0.5 else "⚠ ACCEPTABLE" if self.mean_error<1.0 else "❌ POOR"
        print(f"\n  Mean reprojection error: {self.mean_error:.4f}m  {status}")

    def _overlay_grid(self):
        H_inv = np.linalg.inv(self.H)
        def w2p(wx,wy):
            out = cv2.perspectiveTransform(
                np.array([[[wx,wy]]],dtype=np.float32), H_inv)
            return int(out[0][0][0]),int(out[0][0][1])
        h,w = self.display.shape[:2]
        # 2m grid dots
        for gx in np.arange(0,18.91,2):
            for gy in np.arange(0,9.76,2):
                px,py = w2p(gx,gy)
                if 0<px<w and 0<py<h:
                    cv2.circle(self.display,(px,py),3,(255,255,0),-1)
        # Zone outlines
        zones = [
            ([(-2,-2),(20.9,-2),(20.9,11.75),(-2,11.75)],(0,0,200),"HIGH-Scaffold"),
            ([(1,1),(15.66,1),(15.66,8.75),(1,8.75)],(0,140,255),"MODERATE"),
            ([(16.16,0),(18.9,0),(18.9,9.75),(16.16,9.75)],(0,180,0),"LOW-Parking"),
        ]
        for corners,col,lbl in zones:
            pts=[w2p(c[0],c[1]) for c in corners]
            if all(0<=p[0]<w and 0<=p[1]<h for p in pts):
                cv2.polylines(self.display,[np.array(pts,dtype=np.int32)],True,col,2)
                cx=int(np.mean([p[0] for p in pts]))
                cy=int(np.mean([p[1] for p in pts]))
                cv2.putText(self.display,lbl,(cx-30,cy),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
        # Status bar
        bar_col=(0,200,0) if self.mean_error<0.5 else (0,165,255)
        cv2.rectangle(self.display,(0,h-36),(w,h),(20,20,20),-1)
        cv2.putText(self.display,
                    f"Error:{self.mean_error:.4f}m  |  S=Save  R=Reset  ESC=Quit",
                    (10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.52,bar_col,1)

    def _draw_instructions(self):
        h,w=self.display.shape[:2]
        overlay=self.display.copy()
        cv2.rectangle(overlay,(w-310,8),(w-8,240),(10,10,10),-1)
        cv2.addWeighted(overlay,0.65,self.display,0.35,0,self.display)
        y=28
        for txt,col in [
            ("BuildSight — Homography Cal",(255,255,255)),
            ("",(0,0,0)),
            ("Click IN ORDER:",(200,200,200)),
            ("1. SW — bottom-left (Kitchen)",POINT_COLORS["SW"]),
            ("2. SE — bottom-right (Parking)",POINT_COLORS["SE"]),
            ("3. NE — top-right (Toilet)",POINT_COLORS["NE"]),
            ("4. NW — top-left (Staircase)",POINT_COLORS["NW"]),
            ("",(0,0,0)),
            ("S=Save  R=Reset  ESC=Quit",(180,255,180)),
        ]:
            if txt:
                cv2.putText(self.display,txt,(w-305,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.44,col,1)
            y+=20

    def save(self):
        if self.H is None:
            print("Click all 4 points first.")
            return
        np.save(self.H_path, self.H)
        meta={
            "camera_id":self.camera_id,
            "calibrated_at":datetime.now().isoformat(),
            "frame_size":[self.frame.shape[1],self.frame.shape[0]],
            "pixel_points":{k:list(v) for k,v in self.clicked.items()},
            "world_points":{k:list(v) for k,v in WORLD_POINTS.items()},
            "mean_reprojection_error_m":self.mean_error,
            "H_matrix":self.H.tolist(),
            "site":{"sw_corner":"10°48'59.7\"N 78°40'07.8\"E",
                    "width_m":18.90,"depth_m":9.75}
        }
        with open(self.meta_path,"w") as f:
            json.dump(meta,f,indent=2)
        print(f"  ✅ {self.H_path}")
        print(f"  ✅ {self.meta_path}")

    def reset(self):
        self.clicked={}; self.H=None; self.mean_error=999.0
        self.display=self.frame.copy()
        self._draw_instructions()
        print("  Reset — click SW first.")

    def run(self):
        self.load_frame()
        self._draw_instructions()
        win=f"BuildSight Module 3 — {self.camera_id}"
        cv2.namedWindow(win,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win,self._mouse_callback)
        print("\n"+"="*52)
        print("  BuildSight Module 3 — Homography Calibration")
        print("="*52)
        for i,k in enumerate(CLICK_ORDER):
            print(f"  {i+1}. Click {k}: world({WORLD_POINTS[k][0]:.1f}m,{WORLD_POINTS[k][1]:.1f}m)")
        print("\n  S=Save  R=Reset  ESC=Quit")
        while True:
            cv2.imshow(win,self.display)
            key=cv2.waitKey(30)&0xFF
            if   key==27:          break
            elif key==ord('s'):    self.save()
            elif key==ord('r'):    self.reset()
        cv2.destroyAllWindows()


def test_transform(H_path):
    H=np.load(H_path)
    print(f"\nTesting: {H_path}")
    for px,py,lbl in [(640,540,"Center"),(100,480,"BL"),(1180,480,"BR")]:
        out=cv2.perspectiveTransform(
            np.array([[[float(px),float(py)]]],dtype=np.float32),H)
        print(f"  {lbl}: pixel({px},{py}) → world({out[0][0][0]:.2f}m,{out[0][0][1]:.2f}m)")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="0")
    ap.add_argument("--camera-id",default="CAM-01")
    ap.add_argument("--test",default=None)
    args=ap.parse_args()
    if args.test:
        test_transform(args.test)
    else:
        src=int(args.source) if args.source.isdigit() else args.source
        HomographyCalibrator(src,args.camera_id).run()
