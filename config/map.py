import numpy as np
from skimage.draw import line
from config.settings import MAP_SIZE, MAP_SIZE_M, MAP_RESOLUTION

# 地图参数
MAP_SIZE = 50
MAP_SIZE_M = 15.0
RESOLUTION = MAP_SIZE_M / MAP_SIZE

# 老师给的障碍物线段
SEGMENTS = [
   {
    "start": [0, 0],
    "end": [2, 0]
   },
   {
    "start": [2, 0],
    "end": [2, 2]
   },
   {
    "start": [0, 0],
    "end": [0, 15]
   },
   {
    "start": [0, 11],
    "end": [2, 11]
   },
   {
    "start": [2, 11],
    "end": [2, 6]
   },
   {
    "start": [2, 6],
    "end": [4, 6]
   },
   {
    "start": [0, 15],
    "end": [11, 15]
   },
   {
    "start": [2, 15],
    "end": [2, 13]
   },
   {
    "start": [2, 13],
    "end": [9, 13]
   },
   {
    "start": [4, 13],
    "end": [4, 8]
   },
   {
    "start": [6, 13],
    "end": [6, 10]
   },
   {
    "start": [6, 10],
    "end": [9, 10]
   },
   {
    "start": [9, 10],
    "end": [9, 13]
   },
   {
    "start": [11, 15],
    "end": [11, 10]
   },
   {
    "start": [4, 0],
    "end": [4, 2]
   },
   {
    "start": [4, 0],
    "end": [15, 0]
   },
    {
    "start": [11, 0],
    "end": [11, 2]
   },
   {
    "start": [11, 2],
    "end": [6, 2]
   },
   {
    "start": [6, 2],
    "end": [6, 6]
   },
   {
    "start": [6, 4],
    "end": [2, 4]
   },
   {
    "start": [9, 2],
    "end": [9, 4]
   },
   {
    "start": [15, 0],
    "end": [15, 15]
   },
   {
    "start": [15, 6],
    "end": [13, 6]
   },
   {
    "start": [13, 6],
    "end": [13, 2]
   },
   {
    "start": [15, 15],
    "end": [13, 15]
   },
   {
    "start": [13, 15],
    "end": [13, 8]
   },
   {
    "start": [13, 8],
    "end": [11, 8]
   },
   {
    "start": [4, 8],
    "end": [9, 8]
   },
   {
    "start": [9, 8],
    "end": [9, 6]
   },
   {
    "start": [9, 6],
    "end": [11, 6]
   },
   {
    "start": [11, 6],
    "end": [11, 4]
   }
  ]

def add_segments_to_grid(grid_map, segments, resolution):
    for seg in segments:
        (x0, y0), (x1, y1) = seg['start'], seg['end']
        gx0, gy0 = int(x0 / resolution), int(y0 / resolution)
        gx1, gy1 = int(x1 / resolution), int(y1 / resolution)
        rr, cc = line(gy0, gx0, gy1, gx1)
        rr = np.clip(rr, 0, grid_map.shape[0] - 1)
        cc = np.clip(cc, 0, grid_map.shape[1] - 1)
        grid_map[rr, cc] = 1

def get_global_map():
    grid_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    add_segments_to_grid(grid_map, SEGMENTS, MAP_RESOLUTION)
    return grid_map