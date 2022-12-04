# https://pysource.com
import pyrealsense2 as rs
import numpy as np


class RealsenseCamera:
    def __init__(self, stream_fps=15):
        # ====== Configure depth and color streams ======
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()
        self.stream_fps = stream_fps
        self.camera_info = None

        # ====== List of serial numbers for present cameras ======
        self.realsense_ctx = rs.context()
        self.connected_devices = []
        for i in range(len(self.realsense_ctx.devices)):
            detected_camera = self.realsense_ctx.devices[i].get_info(
                rs.camera_info.serial_number)
            print(f"{detected_camera}")
            self.connected_devices.append(detected_camera)

        # ====== In this example we are only using one camera ======
        self.device = self.connected_devices[0]

        # ====== Enable Streams ======
        self.config = rs.config()
        self.config.enable_device(self.device)
        # config.enable_stream(rs.stream.color, self.stream_width, self.stream_height, rs.format.bgr8, self.stream_fps)
        # config.enable_stream(rs.stream.depth, self.stream_width, self.stream_height, rs.format.z16, self.stream_fps)
        self.config.enable_stream(
            rs.stream.color, rs.format.bgr8, self.stream_fps)
        self.config.enable_stream(
            rs.stream.depth, rs.format.z16, self.stream_fps)
        self.profile = self.pipeline.start(self.config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # ====== Get depth Scale ======
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(
            f"\tDepth Scale for Camera SN {self.device} is: {self.depth_scale}")
        print(f"\tConfiguration Successful for SN {self.device}")

    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        self.depth_frame = depth_frame
        self.color_frame = color_frame

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None

        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        # # Create colormap to show the depth of the Objects
        # colorizer = rs.colorizer()
        # depth_colormap = np.asanyarray(
        #     colorizer.colorize(filled_depth).get_data())
        depth_colormap = None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # depth_image = np.asarray(depth_frame.get_data())
        # depth_colormap = cv2.applyColorMap(
        #     cv2.convertScaleAbs(depth_image, alpha=0.10), 2)

        return True, color_image, depth_image, depth_colormap

    def projection_pixel_point(self):
        self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        self.depth_to_color_extrin = self.depth_frame.profile.get_extrinsics_to(
            self.color_frame.profile)
        self.color_to_depth_extrin = self.color_frame.profile.get_extrinsics_to(
            self.depth_frame.profile)

    def release(self):
        self.pipeline.stop()

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(
        #     cv2.convertScaleAbs(depth_image, alpha=0.10), 2)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
        #     depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        # depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        # background_removed = np.where((depth_image_3d > clipping_distance) | (
        #     depth_image_3d <= 0), background_removed_color, color_image)

        # ====== Set clipping distance ======
        # clipping_distance_in_meters = 2
        # clipping_distance = clipping_distance_in_meters / depth_scale

        # ====== Get distance =====
        # distance = depth_frame.get_distance(int(696), int(517))
        # print("\ndistance", distance)
