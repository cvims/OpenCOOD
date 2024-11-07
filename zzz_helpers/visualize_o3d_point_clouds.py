import pickle as pkl
import numpy as np
import open3d as o3d
import time
import os
import tqdm
import cv2


def get_pkl_data(pkl_element):
    """
    Get the data from the pickle element.

    Parameters
    ----------
    pkl_element : dict
        Dict of point cloud, predictions, and ground truth.

    Returns
    -------
    dict
        Dict of point cloud, predictions, and ground truth.
    """
    pcd = pkl_element['pcd']  # Dict of lists {points, colors}
    pred = pkl_element['oabbs_pred']  # List of dict {points, color}; points contain 8 points (3d box) and color is a list of 3 values
    gt = pkl_element['oabbs_gt']  # List of dict {points, color}; points contain 8 points (3d box) and color is a list of 3 values
    gt_temporal = pkl_element['oabbs_gt_temporal']  # List of dict {points, color}; points contain 8 points (3d box) and color is a list of 3 values
    gt_cavs = pkl_element['oabbs_gt_cav']  # List of dict {points, color}; points contain 8 points (3d box) and color is a list of 3 values

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.array(pcd['points']))
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.array(pcd['colors']))

    pred_o3d = []
    for i in range(len(pred)):
        # Convert the corners of the bounding box to a NumPy array
        corners = np.array(pred[i]['points']).astype(np.float64)  # shape: [8, 3]

        # Create a LineSet for the bounding box
        color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Red color
        bbox_lines = create_bounding_box_lines(corners, color)

        # Add the LineSet to the list
        pred_o3d.append(bbox_lines)

    gt_o3d = []
    for i in range(len(gt)):
        # Convert the corners of the bounding box to a NumPy array
        corners = np.array(gt[i]['points']).astype(np.float64)  # shape: [8, 3]

        # Create a LineSet for the bounding box
        color = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Green color
        bbox_lines = create_bounding_box_lines(corners, color)

        # Add the LineSet to the list
        gt_o3d.append(bbox_lines)
    
    temporal_recovered_gt_o3d = []
    for i in range(len(gt_temporal)):
        # Convert the corners of the bounding box to a NumPy array
        corners = np.array(gt_temporal[i]['points']).astype(np.float64)

        # Create a LineSet for the bounding box
        color = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Blue color
        bbox_lines = create_bounding_box_lines(corners, color)

        # Add the LineSet to the list
        temporal_recovered_gt_o3d.append(bbox_lines)
    
    cavs_gt_o3d = []
    for i in range(len(gt_cavs)):
        # Convert the corners of the bounding box to a NumPy array
        corners = np.array(gt_cavs[i]['points']).astype(np.float64)

        # Create a LineSet for the bounding box
        color = np.array([0.5, 0.5, 0.5], dtype=np.float32) # Gray color
        bbox_lines = create_bounding_box_lines(corners, color)
        cross_lines = create_cav_marker_lines(corners, color)

        # Add the LineSet to the list
        cavs_gt_o3d.append(bbox_lines)
        cavs_gt_o3d.append(cross_lines)
    
    return pcd_o3d, pred_o3d, gt_o3d, temporal_recovered_gt_o3d, cavs_gt_o3d


def create_cav_marker_lines(corners, color):
    # Create a LineSet that creates a cross connecting the corners as a marker for the CAV
    lines = [
        [0, 4], [1, 5], [7, 3], [2, 6]
    ]

    # Convert to a numpy array and create a LineSet
    lines = np.array(lines).astype(np.int32)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set the color for the lines
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (lines.shape[0], 1)).astype(np.float64))  # Repeat color for all lines

    return line_set


def create_bounding_box_lines(corners, color):
    # Define the edges of the bounding box by specifying the connections between corners
    lines = [
        [0, 1], [1, 7], [7, 2], [2, 0],  # Bottom face
        [3, 6], [6, 4], [4, 5], [5, 3],  # Top face
        [0, 3], [1, 6], [7, 4], [2, 5],  # Vertical edges
    ]

    # Convert to a numpy array and create a LineSet
    lines = np.array(lines).astype(np.int32)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Set the color for the lines
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (lines.shape[0], 1)).astype(np.float64))  # Repeat color for all lines

    return line_set


def create_open3d_visualization(element, width=1920, height=1080):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    
    pcd_o3d, pred_o3d, gt_o3d, gt_temporal_o3d, gt_cavs_o3d = get_pkl_data(element)

    vis.add_geometry(pcd_o3d)

    # Add all bounding box elements
    for box in pred_o3d + gt_o3d + gt_temporal_o3d + gt_cavs_o3d:
        vis.add_geometry(box)

    vis.run()
    vis.destroy_window()


def create_open3d_visualization_stream(vis, element):
    pcd_o3d, pred_o3d, gt_o3d, gt_temporal_o3d, gt_cavs_o3d = get_pkl_data(element)

    # clear
    vis.clear_geometries()

    vis.add_geometry(pcd_o3d)
    for box in pred_o3d + gt_o3d + gt_temporal_o3d + gt_cavs_o3d:
        vis.add_geometry(box)
    
    vis.poll_events()
    vis.update_renderer()

    return vis


def update_cv2_video_stream(vis, video_writer, width, height):
    # Capture the image from Open3D and convert it to an OpenCV format
    image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    image = (255 * image).astype(np.uint8)  # Convert to 8-bit
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    # make sure the size is correct
    image = cv2.resize(image, (width, height))

    video_writer.write(image)


def create_cv2_video_stream_from_pkl_files(root_path, video_output_path, width=1920, height=1080, fps=1):
    cv2_video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    _vis = o3d.visualization.Visualizer()
    _vis.create_window(width=width, height=height, visible=False)

    for pkl_element in iterate_pkl_folder(root_path):
        vis = create_open3d_visualization_stream(_vis, pkl_element)

        # set zoom
        vis.get_view_control().set_zoom(0.4)

        update_cv2_video_stream(vis, cv2_video_writer, width, height)

    cv2_video_writer.release()
    vis.destroy_window()


def create_o3d_visualization_from_pkl_files(root_path, width=1920, height=1080):
    for pkl_element in iterate_pkl_folder(root_path):
        create_open3d_visualization(pkl_element, width=width, height=height)


def read_pkl_file(file_path):
    """
    Read the pickle file.

    Parameters
    ----------
    file_path : str
        The file path.

    Returns
    -------
    object
        The object read from the pickle file.
    """
    with open(file_path, 'rb') as f:
        return pkl.load(f)


def iterate_pkl_folder(folder_path: str):
    """
    Read the pickle file.

    Parameters
    ----------
    folder_path : str
        The folder path.

    Returns
    -------
    object
        The object read from the pickle file.
    """
    for file in tqdm.tqdm(sorted(os.listdir(folder_path))):
        if file.endswith(".pkl"):
            yield read_pkl_file(os.path.join(folder_path, file))


if __name__ == '__main__':
    # Directory containing pkl files
    pkl_folder_path = r'C:\Users\roessle\Desktop\vis_pkls\scope'
    video_output_path = r'C:\Users\roessle\Desktop\vis_pkls\scope\pointcloud_video.avi'

    CREATE_O3D_VIS = False
    CREATE_VIDEO = True

    fps = 10
    width, height = 1920, 1080

    if CREATE_O3D_VIS:
        create_o3d_visualization_from_pkl_files(pkl_folder_path, width=width, height=height)

    if CREATE_VIDEO:
        create_cv2_video_stream_from_pkl_files(pkl_folder_path, video_output_path, width=width, height=height, fps=fps)
