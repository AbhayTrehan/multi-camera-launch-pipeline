import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

PACKAGE_NAME = "multi_camera_launch"

# Delay (seconds) between successive camera bring-ups
STAGGER_DELAY_SEC = 2.0
INITIAL_DELAY_SEC = 2.0


def evaluate_camera_pipeline(context, *args, **kwargs):
    robot_yaml_path = LaunchConfiguration("robot_config").perform(context)
    config_dir = os.path.dirname(robot_yaml_path)

    with open(robot_yaml_path, "r") as f:
        robot = yaml.safe_load(f)

    actions = []

    for index, camera_yaml in enumerate(robot["cameras"]):
        yaml_path = os.path.join(config_dir, camera_yaml)
        with open(yaml_path, "r") as f:
            camera_data = yaml.safe_load(f)

        serial = str(camera_data["camera"]["serial_number"])
        camera_ns = f"cam_{serial}"

        calib_path = os.path.abspath(
            os.path.join(config_dir, "..", "calibration", f"{serial}.yaml")
        )
        if not os.path.exists(calib_path):
            raise FileNotFoundError(
                f"Calibration file for camera {serial} not found at {calib_path}"
            )
        camera_info_url = f"file://{calib_path}"

        with open(calib_path, "r") as f:
            calib_data = yaml.safe_load(f)
        image_width = calib_data["image_width"]
        image_height = calib_data["image_height"]

        # ── One isolated container per camera ────────────────────────────
        # Each container gets its own GXF scheduler, thread pool, and
        # CUDA memory pools. Camera 0 can never starve Camera 1/2/3.
        camera_container = ComposableNodeContainer(
            name=f"container_{camera_ns}",
            namespace="",
            package="rclcpp_components",
            executable="component_container_mt",
            output="screen",
            arguments=[
                "--ros-args", "-p", "thread_pool_size:=4"
            ],
            composable_node_descriptions=[
                # --------------------------------------------------------
                # 1. FLIR Hardware Driver Component
                #
                # Publishes distorted Mono8 at the pinned sensor geometry
                # (binning off, 720x540 ROI at 360,270).  Not a NITROS node,
                # so it stays out of the intra-process path.
                # --------------------------------------------------------
                ComposableNode(
                    package="spinnaker_camera_driver",
                    plugin="spinnaker_camera_driver::CameraDriver",
                    name="driver",
                    namespace=camera_ns,
                    parameters=[
                        yaml_path,
                        {"camerainfo_url": camera_info_url},
                    ],
                    extra_arguments=[{"use_intra_process_comms": False}],
                ),
                # --------------------------------------------------------
                # 2. Isaac ROS GPU Rectification (mono8, distorted -> rect)
                #
                # REQUIRED: cuAprilTags takes only {fx, fy, cx, cy} -- the
                # nvAprilTagsCameraIntrinsics_t struct has nowhere to put D.
                # These lenses run k1 ~ -0.39, which displaces the frame
                # corners by ~43 px.  Feeding the detector distorted pixels
                # curves the quad edges (detections rejected near the frame
                # border) and shears the quad (mis-solved tilt).  Measured on
                # a real detection at r_n=0.3, unrectified input cost 7.7 deg
                # of orientation and 7% of range.
                #
                # Runs on mono8, ahead of the RGB expansion, so it warps a
                # third of the bytes.
                # --------------------------------------------------------
                ComposableNode(
                    package="isaac_ros_image_proc",
                    plugin=(
                        "nvidia::isaac_ros::image_proc::RectifyNode"
                    ),
                    name="rectify",
                    namespace=camera_ns,
                    parameters=[{
                        "output_width": image_width,
                        "output_height": image_height,
                        "num_blocks": 20,
                    }],
                    remappings=[
                        ("image_raw", "driver/image_raw"),
                        ("camera_info", "driver/camera_info"),
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                # --------------------------------------------------------
                # 3. Isaac ROS GPU Format Converter (mono8 -> rgb8)
                #
                # Present only because older isaac_ros_apriltag builds accept
                # nitros_image_rgb8/bgr8 but not nitros_image_mono8.  Check
                # yours with:
                #     ros2 component types | grep apriltag
                #     ros2 topic info -v /cam_<serial>/image_rect_color
                # If mono8 is supported, delete this node and point the
                # detector straight at image_rect -- it saves expanding
                # 720x540 mono to RGB at 40 fps on every camera, which
                # cuAprilTags then converts back to grayscale internally.
                # --------------------------------------------------------
                ComposableNode(
                    package="isaac_ros_image_proc",
                    plugin=(
                        "nvidia::isaac_ros::image_proc::"
                        "ImageFormatConverterNode"
                    ),
                    name="format_converter",
                    namespace=camera_ns,
                    parameters=[{
                        "encoding_desired": "rgb8",
                        "image_width": image_width,
                        "image_height": image_height,
                        "num_blocks": 20,
                    }],
                    remappings=[
                        ("image_raw", "image_rect"),
                        ("image", "image_rect_color"),
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                # --------------------------------------------------------
                # 4. Isaac ROS GPU AprilTag
                #
                # camera_info MUST come from the rectify node, not from the
                # driver: camera_info_rect carries D=0 and K=P describing the
                # rectified image.  Wiring this back to driver/camera_info
                # would hand the solver the distorted-image intrinsics again.
                #
                # `size` is the outer edge of the tag's BLACK border square in
                # metres -- not the printed sheet, and not including the white
                # quiet zone.  Pose scale is exactly linear in this value.
                # --------------------------------------------------------
                ComposableNode(
                    package="isaac_ros_apriltag",
                    plugin=(
                        "nvidia::isaac_ros::apriltag::AprilTagNode"
                    ),
                    name="apriltag_detector",
                    namespace=camera_ns,
                    parameters=[{
                        "size": 0.16,
                        "max_tags": 32,
                        "tag_family": "tag36h11",
                    }],
                    remappings=[
                        ("image", "image_rect_color"),
                        ("camera_info", "camera_info_rect"),
                        ("tag_detections", "tag_detections"),
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
            ],
        )

        # Stagger camera bring-up to avoid CUDA init contention
        actions.append(
            TimerAction(
                period=INITIAL_DELAY_SEC + index * STAGGER_DELAY_SEC,
                actions=[camera_container],
            )
        )

    return actions


def generate_launch_description():
    package_share = get_package_share_directory(PACKAGE_NAME)
    default_robot_yaml = os.path.join(
        package_share, "config", "robot.yaml"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_config",
            default_value=default_robot_yaml,
            description=(
                "Absolute path to the central robot.yaml config file"
            ),
        ),
        OpaqueFunction(function=evaluate_camera_pipeline),
    ])

