########################################################
#real-time occupancy grid construction
#selectively scans a single row
########################################################

import cv2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

#grid param
cell = 0.01
left, right = -0.15, 0.15
depthMax = 0.25



try:
    #get center of cam
    colorStream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = colorStream.get_intrinsics()
    ppx, ppy, fx, fy = intr.ppx, intr.ppy, intr.fx, intr.fy
    cy = int(round(intr.ppy))  

    #get meters per depth unit
    depthSensor = profile.get_device().first_depth_sensor()
    depthScale = depthSensor.get_depth_scale()

    #cells
    sideCellNo = int(np.ceil((right - left) / cell))
    depthCellNo = int(np.ceil(depthMax / cell))

    while True:
        frames = align.process(pipeline.wait_for_frames())
        depthFrame = frames.get_depth_frame()
        colorFrame = frames.get_color_frame()
        if not depthFrame or not colorFrame:
            continue
        
        color = np.asanyarray(colorFrame.get_data())
        depth = np.asanyarray(depthFrame.get_data())
        
        #get 1 row on cam
        h, w = depth.shape
        row = max(0, min(h - 1, cy))
        depthRow = depth[row:row+1, :]
        colorRow = color[row:row+1, :]

        #convert to meter depths
        depthRowM = (depthRow.astype(np.float32) * depthScale)[0]

        grid = np.zeros((depthCellNo, sideCellNo), dtype=np.uint8)

        for i in range(w):
            z = float(depthRowM[i])

            if 0 < z < depthMax:
                x = float((i - ppx) / fx * z)
                colIndex = int((x - left) / cell)
                rowIndex = int(z / cell)

                if 0 <= colIndex < sideCellNo and 0 <= rowIndex < depthCellNo:
                    grid[rowIndex][colIndex] = 1

        #flip top row to bottom row
        grid = np.flipud(grid)

        # 0->white, 1->black
        gridImg = (1 - grid.astype(np.uint8)) * 255
        scale = 20
        gridImg = cv2.resize(gridImg, (gridImg.shape[1]*scale, gridImg.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Occupancy Grid", gridImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()