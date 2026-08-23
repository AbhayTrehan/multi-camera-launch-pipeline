#!/usr/bin/env python3
"""
calibrate_firefly.py

One-time intrinsic calibration utility for a FLIR Firefly FFY-U3-16S2C
(serial 25251936) using a ChArUco calibration target.

Pipeline:
    1. Connect to camera via PySpin using its serial number.
    2. Configure acquisition/exposure/gain/binning/ROI settings exactly.
    3. Read back every configured parameter and verify it against the
       expected value; abort on mismatch.
    4. Start acquisition and show a live preview with ChArUco detection.
    5. Let the user capture frames (SPACE), undo (R), finish (ENTER),
       or quit (ESC/Q).
    6. Run cv2.aruco.calibrateCameraCharuco() on the captured frames.
    7. Print a calibration report and save a ROS-style YAML file.
    8. Grab one more frame and show original / undistorted / side-by-side.

Requirements: PySpin (Spinnaker SDK 4.3), OpenCV 4.x (opencv-contrib for
aruco), NumPy, PyYAML.
"""

import sys
import datetime

import numpy as np
import cv2
import yaml

try:
    import PySpin
except ImportError:
    print("ERROR: PySpin is not installed. Install the Spinnaker Python "
          "bindings before running this script.")
    sys.exit(1)


# =============================================================================
# CONFIGURATION -- edit these values as needed
# =============================================================================

CAMERA_SERIAL = "25251936"

# --- Camera acquisition settings (must match EXACTLY after configuration) ---
CAMERA_SETTINGS = {
    "PixelFormat": "Mono8",
    "ExposureAuto": "Off",
    "ExposureTime": 4500.0,          # microseconds
    "GainAuto": "Off",
    "Gain": 0.0,                     # dB
    "GammaEnable": False,
    "AcquisitionFrameRateEnable": True,
    "AcquisitionFrameRate": 40.0,    # fps
    "BinningSelector": "Sensor",
    "BinningHorizontal": 2,
    "BinningVertical": 2,
    "Width": 720,
    "Height": 540,
    "TriggerMode": "Off",
}

# --- ChArUco board parameters (edit to match your physical target) ---
CHARUCO_SQUARES_X = 7          # number of chessboard squares, X direction
CHARUCO_SQUARES_Y = 5          # number of chessboard squares, Y direction
CHARUCO_SQUARE_LENGTH = 0.030  # meters (edge of one chessboard square)
CHARUCO_MARKER_LENGTH = 0.022  # meters (edge of one ArUco marker)
ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_100

# --- Calibration capture parameters ---
MIN_CHARUCO_CORNERS_TO_CAPTURE = 8   # min corners detected to allow SPACE
NUM_IMAGES_REQUIRED = 30             # target number of captured frames

# --- Output file ---
OUTPUT_YAML_PATH = f"{CAMERA_SERIAL}.yaml"

# =============================================================================


class CalibrationError(Exception):
    """Raised for any unrecoverable calibration/configuration failure."""
    pass


# -----------------------------------------------------------------------------
# Camera connection helpers
# -----------------------------------------------------------------------------

def find_camera_by_serial(system, serial):
    """Locate and return the PySpin camera object matching `serial`."""
    cam_list = system.GetCameras()
    try:
        cam = cam_list.GetBySerial(serial)
        if cam is None or not cam.IsValid():
            raise CalibrationError(
                f"No camera found with serial number '{serial}'. "
                f"{cam_list.GetSize()} camera(s) detected on the bus."
            )
        return cam
    except PySpin.SpinnakerException as ex:
        raise CalibrationError(f"Error locating camera by serial: {ex}")


def _set_enum(nodemap, node_name, entry_name):
    """Set an enumeration node to a given entry, by string name."""
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        raise CalibrationError(f"Enum node '{node_name}' not available/writable.")
    entry = node.GetEntryByName(entry_name)
    if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
        raise CalibrationError(
            f"Entry '{entry_name}' not available for enum node '{node_name}'."
        )
    node.SetIntValue(entry.GetValue())


def _get_enum(nodemap, node_name):
    """Read back the current string value of an enumeration node."""
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        raise CalibrationError(f"Enum node '{node_name}' not available/readable.")
    return node.GetCurrentEntry().GetSymbolic()


def _set_bool(nodemap, node_name, value):
    node = PySpin.CBooleanPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        raise CalibrationError(f"Bool node '{node_name}' not available/writable.")
    node.SetValue(bool(value))


def _get_bool(nodemap, node_name):
    node = PySpin.CBooleanPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        raise CalibrationError(f"Bool node '{node_name}' not available/readable.")
    return bool(node.GetValue())


def _set_float(nodemap, node_name, value):
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        raise CalibrationError(f"Float node '{node_name}' not available/writable.")
    lo, hi = node.GetMin(), node.GetMax()
    node.SetValue(max(lo, min(hi, value)))


def _get_float(nodemap, node_name):
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        raise CalibrationError(f"Float node '{node_name}' not available/readable.")
    return float(node.GetValue())


def _set_int(nodemap, node_name, value):
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        raise CalibrationError(f"Int node '{node_name}' not available/writable.")
    lo, hi = node.GetMin(), node.GetMax()
    inc = node.GetInc()
    v = max(lo, min(hi, value))
    v = lo + round((v - lo) / inc) * inc  # snap to increment
    node.SetValue(int(v))


def _get_int(nodemap, node_name):
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        raise CalibrationError(f"Int node '{node_name}' not available/readable.")
    return int(node.GetValue())


def configure_camera(cam, settings):
    """Apply every required setting to the camera, in a safe order."""
    nodemap = cam.GetNodeMap()

    # Binning must be set before Width/Height since it changes valid ranges.
    _set_enum(nodemap, "BinningSelector", settings["BinningSelector"])
    _set_int(nodemap, "BinningHorizontal", settings["BinningHorizontal"])
    _set_int(nodemap, "BinningVertical", settings["BinningVertical"])

    # Pixel format
    _set_enum(nodemap, "PixelFormat", settings["PixelFormat"])

    # ROI (Width/Height) after binning is applied
    _set_int(nodemap, "Width", settings["Width"])
    _set_int(nodemap, "Height", settings["Height"])

    # Exposure: must disable auto exposure before setting ExposureTime
    _set_enum(nodemap, "ExposureAuto", settings["ExposureAuto"])
    _set_float(nodemap, "ExposureTime", settings["ExposureTime"])

    # Gain: must disable auto gain before setting Gain
    _set_enum(nodemap, "GainAuto", settings["GainAuto"])
    _set_float(nodemap, "Gain", settings["Gain"])

    # Gamma
    _set_bool(nodemap, "GammaEnable", settings["GammaEnable"])

    # Frame rate
    _set_bool(nodemap, "AcquisitionFrameRateEnable",
              settings["AcquisitionFrameRateEnable"])
    _set_float(nodemap, "AcquisitionFrameRate", settings["AcquisitionFrameRate"])

    # Trigger mode
    _set_enum(nodemap, "TriggerMode", settings["TriggerMode"])


def verify_camera_settings(cam, settings):
    """Read back every setting and confirm it matches the expected value.

    Raises CalibrationError with a clear message on the first mismatch.
    """
    nodemap = cam.GetNodeMap()
    mismatches = []

    checks = [
        ("PixelFormat", _get_enum(nodemap, "PixelFormat"), settings["PixelFormat"]),
        ("ExposureAuto", _get_enum(nodemap, "ExposureAuto"), settings["ExposureAuto"]),
        ("ExposureTime", _get_float(nodemap, "ExposureTime"), settings["ExposureTime"]),
        ("GainAuto", _get_enum(nodemap, "GainAuto"), settings["GainAuto"]),
        ("Gain", _get_float(nodemap, "Gain"), settings["Gain"]),
        ("GammaEnable", _get_bool(nodemap, "GammaEnable"), settings["GammaEnable"]),
        ("AcquisitionFrameRateEnable",
         _get_bool(nodemap, "AcquisitionFrameRateEnable"),
         settings["AcquisitionFrameRateEnable"]),
        ("AcquisitionFrameRate", _get_float(nodemap, "AcquisitionFrameRate"),
         settings["AcquisitionFrameRate"]),
        ("BinningSelector", _get_enum(nodemap, "BinningSelector"),
         settings["BinningSelector"]),
        ("BinningHorizontal", _get_int(nodemap, "BinningHorizontal"),
         settings["BinningHorizontal"]),
        ("BinningVertical", _get_int(nodemap, "BinningVertical"),
         settings["BinningVertical"]),
        ("Width", _get_int(nodemap, "Width"), settings["Width"]),
        ("Height", _get_int(nodemap, "Height"), settings["Height"]),
        ("TriggerMode", _get_enum(nodemap, "TriggerMode"), settings["TriggerMode"]),
    ]

    for name, actual, expected in checks:
        ok = True
        if isinstance(expected, float):
            # Allow small floating point tolerance (device may round).
            if abs(actual - expected) > max(1.0, abs(expected) * 0.02):
                ok = False
        else:
            if actual != expected:
                ok = False
        if not ok:
            mismatches.append((name, expected, actual))

    if mismatches:
        print("\nERROR: Camera configuration verification FAILED:")
        for name, expected, actual in mismatches:
            print(f"  - {name}: expected={expected!r}, actual={actual!r}")
        raise CalibrationError("One or more camera parameters do not match "
                                "the requested configuration.")

    print("Camera configuration verified OK. All parameters match.")


# -----------------------------------------------------------------------------
# ChArUco board setup
# -----------------------------------------------------------------------------

def build_charuco_board():
    """Create the ArUco dictionary and ChArUco board objects."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_MARKER_LENGTH,
        aruco_dict,
    )
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board)
    return aruco_dict, board, aruco_detector, charuco_detector


def detect_charuco(gray_image, charuco_detector):
    """Detect ChArUco corners/ids and ArUco markers in a grayscale image."""
    try:
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray_image)
    except cv2.error:
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            None, None, None, None)
    return charuco_corners, charuco_ids, marker_corners, marker_ids


# -----------------------------------------------------------------------------
# Image acquisition helper
# -----------------------------------------------------------------------------

def spin_image_to_numpy(image_result, cam):
    """Convert a PySpin image object to an 8-bit numpy grayscale array."""
    if image_result.IsIncomplete():
        return None
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
    converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
    frame = converted.GetNDArray().copy()
    return frame


# -----------------------------------------------------------------------------
# Calibration report / YAML output
# -----------------------------------------------------------------------------

def print_calibration_report(rms, mean_reproj_err, camera_matrix, dist_coeffs,
                              image_size, num_images):
    print("\n" + "=" * 60)
    print(" CALIBRATION REPORT")
    print("=" * 60)
    print(f" Images used            : {num_images}")
    print(f" Image size (w x h)     : {image_size[0]} x {image_size[1]}")
    print(f" RMS reprojection error : {rms:.5f} px")
    print(f" Mean reprojection error: {mean_reproj_err:.5f} px")
    print(" Camera matrix:")
    for row in camera_matrix:
        print("   [" + "  ".join(f"{v: .6f}" for v in row) + "]")
    print(" Distortion coefficients (k1, k2, p1, p2, k3):")
    print("   " + "  ".join(f"{v: .6f}" for v in dist_coeffs.flatten()))
    print("=" * 60 + "\n")


def save_calibration_yaml(path, serial, image_size, camera_matrix, dist_coeffs,
                           rms, mean_reproj_err, camera_settings):
    """Save calibration results in a ROS-camera_info-style YAML file."""
    data = {
        "serial_number": serial,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(v) for v in camera_matrix.flatten()],
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(dist_coeffs.flatten()),
            "data": [float(v) for v in dist_coeffs.flatten()],
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(v) for v in np.eye(3).flatten()],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [float(v) for v in
                     np.hstack([camera_matrix, np.zeros((3, 1))]).flatten()],
        },
        "rms_error": float(rms),
        "mean_reprojection_error": float(mean_reproj_err),
        "calibration_date": datetime.datetime.now().isoformat(),
        "camera_settings": camera_settings,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Calibration saved to '{path}'")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    system = None
    cam = None
    cam_list = None

    try:
        # --- 1. Connect to the camera ---------------------------------------
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        cam = find_camera_by_serial(system, CAMERA_SERIAL)
        cam.Init()
        print(f"Connected to camera serial {CAMERA_SERIAL}.")

        # --- 2. Configure settings --------------------------------------------
        configure_camera(cam, CAMERA_SETTINGS)
        print("Camera configured with requested settings.")

        # --- 3. Verify settings ------------------------------------------------
        verify_camera_settings(cam, CAMERA_SETTINGS)

        # --- 4. Start acquisition ----------------------------------------------
        cam.BeginAcquisition()
        print("Acquisition started.")

        # --- Build ChArUco board / detector ------------------------------------
        aruco_dict, board, aruco_detector, charuco_detector = build_charuco_board()

        window_name = "Firefly Calibration - SPACE=capture R=undo ENTER=finish ESC/Q=quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        captured_object_points = []   # not used directly (board handles it)
        captured_charuco_corners = []
        captured_charuco_ids = []
        image_size = None

        print(f"\nCapturing calibration images. Need {NUM_IMAGES_REQUIRED}.")
        print("Controls: SPACE=capture, R=remove last, ENTER=finish, ESC/Q=quit\n")

        # --- 5/6/7/8. Live preview + capture loop -------------------------------
        while True:
            image_result = cam.GetNextImage(2000)
            frame = spin_image_to_numpy(image_result, cam)
            image_result.Release()

            if frame is None:
                continue

            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])  # (w, h)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                detect_charuco(frame, charuco_detector)

            num_corners = 0
            if marker_corners is not None and marker_ids is not None and \
                    len(marker_corners) > 0:
                cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)

            if charuco_corners is not None and charuco_ids is not None:
                num_corners = len(charuco_corners)
                if num_corners > 0:
                    cv2.aruco.drawDetectedCornersCharuco(
                        display_frame, charuco_corners, charuco_ids,
                        (0, 255, 0))

            status_color = (0, 255, 0) if num_corners >= MIN_CHARUCO_CORNERS_TO_CAPTURE \
                else (0, 0, 255)
            cv2.putText(display_frame,
                        f"Captured images: {len(captured_charuco_corners)} / "
                        f"{NUM_IMAGES_REQUIRED}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame,
                        f"ChArUco corners detected: {num_corners}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC / Q
                print("Quitting without saving.")
                cv2.destroyAllWindows()
                cam.EndAcquisition()
                cam.DeInit()
                del cam
                cam_list.Clear()
                system.ReleaseInstance()
                sys.exit(0)

            elif key == ord(' '):  # SPACE -> capture
                if charuco_corners is not None and \
                        num_corners >= MIN_CHARUCO_CORNERS_TO_CAPTURE:
                    captured_charuco_corners.append(charuco_corners)
                    captured_charuco_ids.append(charuco_ids)
                    print(f"Captured images: {len(captured_charuco_corners)} / "
                          f"{NUM_IMAGES_REQUIRED}")
                else:
                    print(f"Not enough ChArUco corners detected "
                          f"({num_corners} < {MIN_CHARUCO_CORNERS_TO_CAPTURE}). "
                          f"Capture skipped.")

            elif key == ord('r') or key == ord('R'):  # R -> remove last
                if captured_charuco_corners:
                    captured_charuco_corners.pop()
                    captured_charuco_ids.pop()
                    print(f"Removed last capture. Captured images: "
                          f"{len(captured_charuco_corners)} / {NUM_IMAGES_REQUIRED}")
                else:
                    print("No captured images to remove.")

            elif key == 13 or key == 10:  # ENTER -> finish
                if len(captured_charuco_corners) < 4:
                    print("Need at least a few captures before calibrating "
                          "(got fewer than 4). Keep capturing.")
                    continue
                print("Finishing capture, starting calibration...")
                break

        cv2.destroyAllWindows()

        # --- 10/11. Run calibration ---------------------------------------------
        print(f"Running cv2.aruco.calibrateCameraCharuco() on "
              f"{len(captured_charuco_corners)} images...")

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
            cv2.aruco.calibrateCameraCharuco(
                charucoCorners=captured_charuco_corners,
                charucoIds=captured_charuco_ids,
                board=board,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None,
            )

        # --- Compute mean reprojection error across all frames -----------------
        total_error = 0.0
        total_points = 0
        for i in range(len(captured_charuco_corners)):
            obj_pts, img_pts = board.matchImagePoints(
                captured_charuco_corners[i], captured_charuco_ids[i])
            if obj_pts is None or len(obj_pts) == 0:
                continue
            projected, _ = cv2.projectPoints(
                obj_pts, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
            total_error += error * len(projected)
            total_points += len(projected)
        mean_reproj_err = total_error / total_points if total_points > 0 else float("nan")

        # --- 12. Print report ----------------------------------------------------
        print_calibration_report(rms, mean_reproj_err, camera_matrix, dist_coeffs,
                                  image_size, len(captured_charuco_corners))

        # --- 13. Save YAML ---------------------------------------------------------
        save_calibration_yaml(
            OUTPUT_YAML_PATH, CAMERA_SERIAL, image_size, camera_matrix, dist_coeffs,
            rms, mean_reproj_err, CAMERA_SETTINGS)

        # --- 14. Grab one more frame and show undistorted comparison -----------
        print("Grabbing verification frame...")
        image_result = cam.GetNextImage(2000)
        verify_frame = spin_image_to_numpy(image_result, cam)
        image_result.Release()

        if verify_frame is not None:
            undistorted = cv2.undistort(verify_frame, camera_matrix, dist_coeffs)

            original_bgr = cv2.cvtColor(verify_frame, cv2.COLOR_GRAY2BGR)
            undistorted_bgr = cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR)
            side_by_side = np.hstack([original_bgr, undistorted_bgr])

            cv2.putText(side_by_side, "Original", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(side_by_side,
                        "Undistorted", (original_bgr.shape[1] + 15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Original", original_bgr)
            cv2.imshow("Undistorted", undistorted_bgr)
            cv2.imshow("Side by Side", side_by_side)
            print("Press any key in an image window to close and exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # --- Clean shutdown --------------------------------------------------------
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        print("Done.")

    except CalibrationError as ce:
        print(f"\nCALIBRATION ERROR: {ce}")
        _safe_shutdown(cam, cam_list, system)
        sys.exit(1)

    except PySpin.SpinnakerException as se:
        print(f"\nSPINNAKER ERROR: {se}")
        _safe_shutdown(cam, cam_list, system)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        _safe_shutdown(cam, cam_list, system)
        sys.exit(1)

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        _safe_shutdown(cam, cam_list, system)
        sys.exit(1)


def _safe_shutdown(cam, cam_list, system):
    """Best-effort cleanup of camera/system resources on error paths."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        if cam is not None:
            if cam.IsStreaming():
                cam.EndAcquisition()
            if cam.IsInitialized():
                cam.DeInit()
            del cam
    except Exception:
        pass
    try:
        if cam_list is not None:
            cam_list.Clear()
    except Exception:
        pass
    try:
        if system is not None:
            system.ReleaseInstance()
    except Exception:
        pass


if __name__ == "__main__":
    main()
