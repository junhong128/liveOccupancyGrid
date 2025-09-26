import os
import sys
import math
import time
import threading
from typing import Tuple
import numpy as np
import cv2
import pyrealsense2 as rs



#************************ CONFIG ************************
#gui
SHOW_GUI = True
VIS_SCALE = 10
SHOW_DEBUG = False
last_endpoints = None

#depth stream res (424×240@30, 640×480@15–30, 320×240@30)
DEPTH_WIDTH = 424
DEPTH_HEIGHT = 240
DEPTH_FPS = 30

#cam mount
CAM_HEIGHT_M = 0.20     #measure n change
CAM_PITCH_DEG = 5.0     #measure n change (downward degree)

#depth range
MIN_DEPTH_M = 0.25
MAX_DEPTH_M = 1

#cam -> grid config
GRID_RES_M = 0.05           #cell size
GRID_X_MAX_M = 1            #max forward
GRID_X_MIN_M = -0.2         #max backward (negative)
GRID_Y_MAX_M = 0.6          #max left
GRID_Y_MIN_M = -0.6         #max right (negative)

#logodds config
LOG_ODDS_OCC = 0.8          #inc for occ
LOG_ODDS_FREE = -0.3        #dec for free
LOG_ODDS_MIN = -2.5         #min clamp
LOG_ODDS_MAX = 2.5          #max clamp
PROB_THRESHOLD = 0.65       #p > PROB_THRESHOLD = occ

#raycasting n sampling
SUBSAMPLE_STEP_V = 2        #rows
SUBSAMPLE_STEP_U = 2        #cols
MAX_RAY_STEPS = 60          #max no. of cells to mark free 

#cell colors
COLOR_OCC = (0, 0, 0)
COLOR_FREE = (200, 200, 200) 
COLOR_UNKNOWN = (130, 130, 130)



#************************ FUNCTIONS ************************
def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s,  c]], dtype=np.float32)
    return R

def logodds_to_prob(L: np.ndarray) -> np.ndarray:
    #return probabilities
    return 1.0 - 1.0 / (1.0 + np.exp(L))

def world_to_grid(x_m: np.ndarray, y_m: np.ndarray, grid_shape):
    #world point to grid coord (row, col)
    rows, cols = grid_shape
    col = np.floor((y_m - GRID_Y_MIN_M) / GRID_RES_M).astype(np.int32)
    row = np.floor((GRID_X_MAX_M - x_m) / GRID_RES_M).astype(np.int32)
    inb = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return row, col, inb

def bresenham_line(r0, c0, r1, c1, max_steps=None):
    #return points along the ray
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = (dr - dc)
    pts = []
    r, c = r0, c0
    steps = 0
    while True:
        pts.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return pts

def draw_grid(logodds):
    probs = logodds_to_prob(logodds)

    #mark cells
    occupied = probs > PROB_THRESHOLD
    unknown = (logodds == 0.0)
    free = (~occupied) & (~unknown)

    #paint grid
    img = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
    img[unknown] = COLOR_UNKNOWN
    img[free] = COLOR_FREE
    img[occupied] = COLOR_OCC

    #flip y (left of cam = left of red dot)
    img = cv2.flip(img, 1)

    if SHOW_DEBUG and last_endpoints is not None:
        rr, cc = last_endpoints
        cc_flipped = (grid_cols - 1) - cc
        mask = (rr >= 0) & (rr < grid_rows) & (cc_flipped >= 0) & (cc_flipped < grid_cols)
        img[rr[mask], cc_flipped[mask]] = (0, 128, 0)

    #red dot -> camera
    r_draw = origin_row
    c_draw = (grid_cols - 1) - origin_col
    if 0 <= r_draw < img.shape[0] and 0 <= c_draw < img.shape[1]:
        img = np.ascontiguousarray(img)
        cv2.circle(img, (int(c_draw), int(r_draw)), 2, (0, 0, 255), -1)

    img = cv2.resize(img, (grid_cols * VIS_SCALE, grid_rows * VIS_SCALE), interpolation=cv2.INTER_NEAREST)
    return img

def update_grid_with_points(xs, ys):
    global logodds, last_endpoints
    H, W = logodds.shape

    #convert meter to grid cell
    rows, cols, inb = world_to_grid(xs, ys, logodds.shape)
    rows = rows[inb]
    cols = cols[inb]

    #robot cell
    if not (0 <= origin_row < H and 0 <= origin_col < W):
        return
    
    eps_r = []
    eps_c = []

    for r_end, c_end in zip(rows, cols):
        #mark free cells
        pts = bresenham_line(origin_row, origin_col, int(r_end), int(c_end), max_steps=MAX_RAY_STEPS)
        if not pts:
            continue

        #check if endpoint reached or hit occ
        end_reached = (pts[-1][0] == int(r_end)) and (pts[-1][1] == int(c_end))

        #update logodds
        free_pts = pts if not end_reached else pts[:-1]
        for rr, cc in free_pts:
            if 0 <= rr < H and 0 <= cc < W:
                logodds[rr, cc] += LOG_ODDS_FREE
        if end_reached:
            rr, cc = pts[-1]
            if 0 <= rr < H and 0 <= cc < W:
                logodds[rr, cc] += LOG_ODDS_OCC
                eps_r.append(rr)
                eps_c.append(cc)

    last_endpoints = (np.array(eps_r, dtype=np.int32), np.array(eps_c, dtype=np.int32)) if eps_r else None

    #clamp
    np.clip(logodds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=logodds)



#************************ CAM SETUP ************************
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
stream_profile = profile.get_stream(rs.stream.depth)
intr = stream_profile.as_video_stream_profile().get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

#precompute pixel directions
u = np.arange(DEPTH_WIDTH, dtype=np.float32)
v = np.arange(DEPTH_HEIGHT, dtype=np.float32)
uu, vv = np.meshgrid(u, v)
dir_x = (uu - cx) / fx
dir_y = (vv - cy) / fy

#cam downwarsds tilt
pitch_rad = math.radians(CAM_PITCH_DEG)
R_cam_to_robot = rotation_x(pitch_rad)



#************************ GRID SETUP ************************
grid_rows = int(np.ceil((GRID_X_MAX_M - GRID_X_MIN_M) / GRID_RES_M))
grid_cols = int(np.ceil((GRID_Y_MAX_M - GRID_Y_MIN_M) / GRID_RES_M))

logodds = np.zeros((grid_rows, grid_cols), dtype=np.float32)

origin_row, origin_col, _ = world_to_grid(
    np.array([0.0], dtype=np.float32),
    np.array([0.0], dtype=np.float32),
    logodds.shape
)
origin_row = int(origin_row[0])
origin_col = int(origin_col[0])





#************************ MAIN ************************
print("grid shape:", logodds.shape, "origin (row,col):", origin_row, origin_col)
def main():
    global logodds
    print("Starting stream... Press 'q' in the window to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            #get depth (np arr) n filter
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            depth[(depth < MIN_DEPTH_M) | (depth > MAX_DEPTH_M)] = 0.0
            depth = cv2.medianBlur(depth, 5)

            #subsample
            d = depth[::SUBSAMPLE_STEP_V, ::SUBSAMPLE_STEP_U]
            dirx = dir_x[::SUBSAMPLE_STEP_V, ::SUBSAMPLE_STEP_U]
            diry = dir_y[::SUBSAMPLE_STEP_V, ::SUBSAMPLE_STEP_U]

            #get points
            z = d                   #front back
            x = dirx * z            #left right
            y = -1 * (diry * z)     #up down

            #remove invalid
            valid = z > 0.0
            x = x[valid]
            y = y[valid]
            z = z[valid]

            #align to robot from cam (use cam pitch n height)
            pts_cam = np.stack([x, y, z], axis=0)
            pts_robot = R_cam_to_robot @ pts_cam
            pts_robot[2, :] += CAM_HEIGHT_M

            #keep points near ground
            z_robot = pts_robot[2, :]
            keep = (z_robot > -0.10) & (z_robot < 1.00)

            #get position of dots
            xr = pts_robot[0, keep]
            yr = pts_robot[1, keep]

            update_grid_with_points(xr, yr)

            if SHOW_GUI:
                vis = draw_grid(logodds)
                cv2.imshow("Live Occupancy Grid", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                time.sleep(0.01)
    finally:
        pipeline.stop()
        if SHOW_GUI:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
