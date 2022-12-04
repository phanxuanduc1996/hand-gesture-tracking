import math
import cv2
import numpy as np
import datetime as dt

from pymouse import PyMouse

import mediapipe
import pyrealsense2 as rs
from construct_equation import *
from transform_polygon2rectangle import *
from shapely.geometry import Point, Polygon
from realsense_camera import RealsenseCamera

import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Screen coordinates in frame.
# ------------- CHANGE VALUE ----------------
screen_tl_pixel = [252, 144]
screen_tr_pixel = [753, 223]
screen_bl_pixel = [331, 656]
screen_br_pixel = [816, 506]


def hand_detection(ratio_pixel_queue, ratio_pixel_lock):
    font = cv2.FONT_HERSHEY_SIMPLEX
    coord_num_hands = (20, 20)
    font_scale = 0.5
    color = (255, 0, 0)
    thickness = 1

    # ====== Realsense ======
    rs_camera = RealsenseCamera(stream_fps=15)

    # ====== Mediapipe ======
    mp_hands = mediapipe.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mediapipe.solutions.drawing_utils

    polygon_points = np.array([screen_tl_pixel, screen_tr_pixel,
                               screen_br_pixel, screen_bl_pixel])

    show_image_shape = True
    flag_detect_screen = False
    hand_side_label = {"0": "Right", "1": "Left"}
    counter_frame = 0
    while True:
        # Necessary for FPS calculations
        start_time = dt.datetime.today().timestamp()

        # Get frame in real time from Realsense camera
        ret, color_image, depth_image, depth_colormap = rs_camera.get_frame_stream()
        if not ret:
            continue

        # Process images
        depth_image_flipped = depth_image.copy()

        # cv2.imshow("depth_colormap", depth_colormap)
        depth_image_height = depth_image_flipped.shape[0]
        depth_image_width = depth_image_flipped.shape[1]
        if show_image_shape:
            print("\nAligned Color Image Shape: {}".format(color_image.shape))
            print("Aligned Depth Image Shape: {}\n".format(depth_image.shape))
            show_image_shape = False

        images = color_image.copy()
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        counter_frame += 1
        intersection_pixel = []
        is_inside = False
        if counter_frame:
            # Process hands
            results = hands.process(color_images_rgb)
            if results.multi_hand_landmarks:
                number_of_hands = len(results.multi_hand_landmarks)

                for hand_idx in range(number_of_hands):
                    hand_lms = results.multi_hand_landmarks[hand_idx]

                    mp_draw.draw_landmarks(
                        images, hand_lms, mp_hands.HAND_CONNECTIONS)
                    coord_hand_distance = (
                        20, coord_num_hands[1]+(20*(hand_idx+1)))
                    hand_side_classification_list = results.multi_handedness[hand_idx]
                    # hand_side = hand_side_classification_list.classification[0].label
                    hand_side_index = hand_side_classification_list.classification[0].index
                    hand_side = hand_side_label[str(hand_side_index)]

                    # Get information of index finger pip
                    index_finger_mcp = results.multi_hand_landmarks[hand_idx].landmark[7]
                    index_mcp_x = int(index_finger_mcp.x * depth_image_width)
                    index_mcp_y = int(index_finger_mcp.y * depth_image_height)
                    if index_mcp_x >= depth_image_width:
                        index_mcp_x = depth_image_width - 1
                    if index_mcp_y >= depth_image_height:
                        index_mcp_y = depth_image_height - 1

                    # Get information of index finger tip
                    index_finger_tip = results.multi_hand_landmarks[hand_idx].landmark[8]
                    index_tip_x = int(index_finger_tip.x * depth_image_width)
                    index_tip_y = int(index_finger_tip.y * depth_image_height)
                    if index_tip_x >= depth_image_width:
                        index_tip_x = depth_image_width - 1
                    if index_tip_y >= depth_image_height:
                        index_tip_y = depth_image_height - 1

                    # Get distance of index_finger_mcp and index_finger_tip
                    index_mcp_dis = float(depth_image_flipped[index_mcp_y,
                                                              index_mcp_x] * rs_camera.depth_scale)  # meters
                    # index_mcp_dis_feet = index_mcp_dis * 3.281  # feet

                    # index_tip_dis = float(depth_image_flipped[index_tip_y,
                    #                                           index_tip_x] * rs_camera.depth_scale)  # meters
                    # index_tip_dis_feet = index_tip_dis * 3.281  # feet

                    # Convert 2D -> 3D and 3D -> 2D
                    if flag_detect_screen == False:
                        rs_camera.projection_pixel_point()

                    # --------------- INDEX FINGER -------------------
                    index_mcp_depth_pixel = [index_mcp_x, index_mcp_y]
                    index_mcp_depth_point = rs.rs2_deproject_pixel_to_point(
                        rs_camera.depth_intrin, index_mcp_depth_pixel, index_mcp_dis)
                    # index_mcp_color_point = rs.rs2_transform_point_to_point(rs_camera.depth_to_color_extrin, index_mcp_depth_point)
                    # index_mcp_color_pixel = rs.rs2_project_point_to_pixel(rs_camera.color_intrin, index_mcp_color_point)

                    index_tip_depth_pixel = [index_tip_x, index_tip_y]
                    index_tip_depth_point = rs.rs2_deproject_pixel_to_point(
                        rs_camera.depth_intrin, index_tip_depth_pixel, index_mcp_dis)
                    # index_tip_color_point = rs.rs2_transform_point_to_point(rs_camera.depth_to_color_extrin, index_tip_depth_point)
                    # index_tip_color_pixel = rs.rs2_project_point_to_pixel(rs_camera.color_intrin, index_tip_color_point)

                    # --------------- SCREEN COORDINATES -------------------
                    if flag_detect_screen == False:
                        screen_tl_dis = float(depth_image_flipped[screen_tl_pixel[1],
                                                                  screen_tl_pixel[0]] * rs_camera.depth_scale)
                        screen_tl_point = rs.rs2_deproject_pixel_to_point(
                            rs_camera.depth_intrin, screen_tl_pixel, screen_tl_dis)
                        # print("\n\nscreen_tl_dis: {}".format(screen_tl_dis))
                        # print("screen_tl_point: {}".format(screen_tl_point))

                        screen_tr_dis = float(depth_image_flipped[screen_tr_pixel[1],
                                                                  screen_tr_pixel[0]] * rs_camera.depth_scale)
                        screen_tr_point = rs.rs2_deproject_pixel_to_point(
                            rs_camera.depth_intrin, screen_tr_pixel, screen_tr_dis)
                        # print("screen_tr_dis: {}".format(screen_tr_dis))
                        # print("screen_tr_point: {}".format(screen_tr_point))

                        screen_bl_dis = float(depth_image_flipped[screen_bl_pixel[1],
                                                                  screen_bl_pixel[0]] * rs_camera.depth_scale)
                        screen_bl_point = rs.rs2_deproject_pixel_to_point(
                            rs_camera.depth_intrin, screen_bl_pixel, screen_bl_dis)
                        # print("screen_bl_dis: {}".format(screen_bl_dis))
                        # print("screen_bl_point: {}".format(screen_bl_point))

                        screen_br_dis = float(depth_image_flipped[screen_br_pixel[1],
                                                                  screen_br_pixel[0]] * rs_camera.depth_scale)
                        screen_br_point = rs.rs2_deproject_pixel_to_point(
                            rs_camera.depth_intrin, screen_br_pixel, screen_br_dis)
                        # print("screen_br_dis: {}".format(screen_br_dis))
                        # print("screen_br_point: {}".format(screen_br_point))

                        plane_coeff = equation_plane_Oxyz(
                            point_1=screen_tl_point, point_2=screen_tr_point, point_3=screen_bl_point)

                        if (math.fabs(plane_coeff[0] * screen_br_point[0] + plane_coeff[1] * screen_br_point[1] + plane_coeff[2] * screen_br_point[2] + plane_coeff[3]) <= 1e-4):
                            flag_detect_screen = True
                        #     print("\nPoint 4 is in plane")
                        # else:
                        #     print("\nPoint 4 is not in plane : {}".format(
                        #         plane_coeff[0] * screen_br_point[0] + plane_coeff[1] * screen_br_point[1] + plane_coeff[2] * screen_br_point[2] + plane_coeff[3]))

                    # ----------------- Find equation of line and plane in 3D space ----------
                    line_coeff = equation_line_Oxyz(
                        point_1=index_mcp_depth_point, point_2=index_tip_depth_point)
                    intersection_point = intersection_line_plane_Oxyz(
                        line_coeff=line_coeff, plane_coeff=plane_coeff)

                    if intersection_point[0] is not None:
                        intersection_pixel_float = rs.rs2_project_point_to_pixel(
                            rs_camera.color_intrin, intersection_point)

                        for res in intersection_pixel_float:
                            intersection_pixel.append(int(res))
                        # print(
                        #     "3D - Coordinate of hand-screen intersection: {}".format(intersection_point))
                        # print(
                        #     "2D - Coordinate of hand-screen intersection: {}".format(result_pixel))

                        # ------------- Checking if a point is inside a polygon -------------
                        point_intersec = Point(
                            intersection_pixel[0], intersection_pixel[1])
                        polygon = Polygon(
                            [screen_tl_pixel, screen_tr_pixel, screen_br_pixel, screen_bl_pixel])
                        is_inside = polygon.contains(point_intersec)
                        # print("The intersection is in a Polygon or not: {}".format(is_inside))

                        if is_inside:
                            cv2.circle(images, intersection_pixel, radius=5,
                                       color=(255, 0, 0), thickness=5)

                    # ------------------- Draw information on image -------------------
                    images = cv2.putText(images, f"{hand_side} Hand Distance: ({index_mcp_dis:0.3} m) away",
                                         coord_hand_distance, font, font_scale, color, thickness, cv2.LINE_AA)
                    images = cv2.putText(
                        images, f"Hands: {number_of_hands}", coord_num_hands, font, font_scale, color, thickness, cv2.LINE_AA)
            else:
                images = cv2.putText(images, "No Hands", coord_num_hands, font,
                                     font_scale, color, thickness, cv2.LINE_AA)
            counter_frame = 0

        # Draw a Blue polygon
        images = cv2.polylines(
            images, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=1)

        warped, matrix = four_point_transform(images.copy(), polygon_points)
        if is_inside:
            intersection_p_after = warp_point(intersection_pixel, matrix)
            # cv2.circle(warped, intersection_p_after, radius=3,
            #                        color=(0, 0, 255), thickness=3)

            ratio_x = float(intersection_p_after[0] / warped.shape[1])
            ratio_y = float(intersection_p_after[1] / warped.shape[0])
            ratio_pixel_lock.acquire()
            ratio_pixel_queue.put([ratio_x, ratio_y])
            ratio_pixel_lock.release()

        # cv2.imshow("Image transform", warped)

        # Display FPS
        time_diff = dt.datetime.today().timestamp() - start_time
        fps = int(1 / time_diff)
        coord_fps = (20, coord_num_hands[1] + 60)
        images = cv2.putText(
            images, f"FPS: {fps}", coord_fps, font, font_scale, color, thickness, cv2.LINE_AA)
        name_of_window = 'SN: ' + str(rs_camera.device)

        # Display images
        cv2.imshow(name_of_window, images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {rs_camera.device}")
            break

    print(f"Exiting loop for SN: {rs_camera.device}")
    print(f"Application Closing.")
    rs_camera.release()
    print(f"Application Closed.")


def click_mouse(ratio_pixel_queue, ratio_pixel_lock):
    py_mouse = PyMouse()
    while True:
        if not ratio_pixel_queue.empty():
            ratio_pixel_lock.acquire()
            ratio_pixel = ratio_pixel_queue.get()
            ratio_pixel_lock.release()

            ratio_x, ratio_y = ratio_pixel[0], ratio_pixel[1]
            img_w, img_h = 1920, 1080

            if (ratio_x >= 0) and (ratio_x <= 1) and (ratio_y >= 0) and (ratio_y <= 1):
                coord_x = int(ratio_x * img_w)
                coord_y = int(ratio_y * img_h)
                print("\n\n--------- Absolute coordinates on the screen: ",
                      coord_x, coord_y)

                # py_mouse.move(coord_x, coord_y)
                py_mouse.click(coord_x, coord_y, 1)
                # py_keyboard.type_string("X: " + str(coord_x) + "   Y: " + str(coord_y))


if __name__ == "__main__":
    ratio_pixel_queue = mp.Queue()
    ratio_pixel_lock = threading.Lock()

    execute = ThreadPoolExecutor(max_workers=10)
    execute.submit(hand_detection, ratio_pixel_queue, ratio_pixel_lock)

    execute = ThreadPoolExecutor(max_workers=1)
    execute.submit(click_mouse, ratio_pixel_queue, ratio_pixel_lock)
