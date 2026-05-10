import argparse
import json
import os
import shutil
import sys
import math
import pandas as pd
import numpy as np
import tkinter as tk
import threading

# ---------------------------------------------------------
# 1. EVALUATION & AESTHETIC UI FUNCTIONS
# ---------------------------------------------------------

def calc_ndtw(pred_traj, gt_traj, threshold=3.0):
    if len(pred_traj) == 0 or len(gt_traj) == 0:
        return 0.0
    pred_xy = np.array(pred_traj)
    gt_xy = np.array(gt_traj)
    N, M = len(pred_xy), len(gt_xy)
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dist = np.linalg.norm(pred_xy[i - 1] - gt_xy[j - 1])
            dtw[i, j] = dist + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    dtw_distance = dtw[N, M]
    ndtw = np.exp(-dtw_distance / (threshold * len(gt_xy)))
    return float(ndtw)

def calculate_trajectory_length(coords):
    if len(coords) < 2:
        return 0.0
    total_length = 0.0
    for i in range(1, len(coords)):
        dist = math.sqrt((coords[i][0] - coords[i-1][0])**2 + 
                        (coords[i][1] - coords[i-1][1])**2)
        total_length += dist
    return total_length

def evaluate_single_episode(csv_file, episode_data):
    """
    Returns both a string (for console) and a dict (for the UI).
    """
    if not os.path.exists(csv_file):
        print("Error: CSV file not found.")
        return None, None

    df = pd.read_csv(csv_file)
    coords = df[['pos_x', 'pos_y']].values.tolist()

    filtered_coords = [coords[0]]
    for point in coords[1:]:
        if point != filtered_coords[-1]:
            filtered_coords.append(point)

    goal = episode_data['goals']['position']
    goal_x, goal_y = goal[0], goal[1]
    
    reference_path = episode_data['reference_path']
    gt_coords = [[pos[0], pos[1]] for pos in reference_path]

    last_x, last_y = filtered_coords[-1]
    dist_last = math.sqrt((goal_x - last_x)**2 + (goal_y - last_y)**2)
    sr = 1 if dist_last <= 3 else 0

    osr = 0
    for idx, (x, y) in enumerate(filtered_coords):
        if math.sqrt((goal_x - x)**2 + (goal_y - y)**2) <= 3:
            osr = 1
            break

    ndtw_score = calc_ndtw(filtered_coords, gt_coords, threshold=3.0)
    tl = calculate_trajectory_length(filtered_coords)
    tl_gt = calculate_trajectory_length(gt_coords)
    
    if tl_gt > 0 and tl > 0:
        spl = sr * (tl_gt / max(tl, tl_gt))
    else:
        spl = 0.0

    # Console Output
    result_str = (
        f"--- Episode Evaluation ---\n"
        f"SR: {sr}, OSR: {osr}, SPL: {spl:.3f}, nDTW: {ndtw_score:.3f}, TL: {tl:.2f}m"
    )
    print(f"\n{result_str}\n")
    
    # Dictionary for GUI
    metrics = {
        "SR": sr,
        "OSR": osr,
        "SPL": spl,
        "nDTW": ndtw_score,
        "TL": tl,
        "Goal Dist": dist_last
    }
    return result_str, metrics

def show_beautiful_popup(metrics):
    """
    Displays a styled, modern popup for results.
    """
    if metrics is None:
        return

    root = tk.Tk()
    root.title("Evaluation Results")
    
    # Calculate screen center
    w, h = 400, 450
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
    root.configure(bg="#f0f2f5")
    root.attributes('-topmost', True)
    root.resizable(False, False)

    # -- Header Section --
    is_success = metrics["SR"] == 1
    header_color = "#2ecc71" if is_success else "#e74c3c" # Green or Red
    header_text = "MISSION SUCCESS!" if is_success else "MISSION FAILED"
    
    header_frame = tk.Frame(root, bg=header_color, height=80)
    header_frame.pack(fill='x')
    
    lbl_header = tk.Label(
        header_frame, 
        text=header_text, 
        font=("Helvetica", 18, "bold"), 
        bg=header_color, 
        fg="white"
    )
    lbl_header.pack(pady=20)

    # -- Content Section --
    content_frame = tk.Frame(root, bg="white", padx=20, pady=20)
    content_frame.pack(fill='both', expand=True, padx=15, pady=15)

    # Helper to create rows
    def create_row(parent, label, value, row_idx):
        lbl = tk.Label(parent, text=label, font=("Arial", 11, "bold"), bg="white", fg="#555")
        lbl.grid(row=row_idx, column=0, sticky="w", pady=8)
        
        val_txt = f"{value:.3f}" if isinstance(value, float) else str(value)
        val = tk.Label(parent, text=val_txt, font=("Consolas", 12), bg="white", fg="#333")
        val.grid(row=row_idx, column=1, sticky="e", pady=8)

    # Grid config
    content_frame.columnconfigure(0, weight=1)
    content_frame.columnconfigure(1, weight=1)

    create_row(content_frame, "Success Rate (SR):", metrics["SR"], 0)
    create_row(content_frame, "Oracle Success (OSR):", metrics["OSR"], 1)
    create_row(content_frame, "SPL Score:", metrics["SPL"], 2)
    create_row(content_frame, "nDTW Score:", metrics["nDTW"], 3)
    create_row(content_frame, "Trajectory Length:", f"{metrics['TL']:.2f} m", 4)
    create_row(content_frame, "Dist to Goal:", f"{metrics['Goal Dist']:.2f} m", 5)

    # -- Footer / Button --
    btn = tk.Button(
        root, 
        text="Close & Finish", 
        command=root.destroy, 
        bg="#34495e", 
        fg="white", 
        font=("Arial", 11, "bold"),
        relief="flat",
        height=2
    )
    btn.pack(fill='x', padx=15, pady=(0, 15))

    root.mainloop()

def show_instruction_left_centered(instruction_text):
    """Shows instruction window vertically centered on the left edge."""
    def create_window():
        root = tk.Tk()
        root.title("Task")
        
        # Window Dimensions
        win_w, win_h = 300, 650
        
        # Get Screen Dimensions
        screen_h = root.winfo_screenheight()
        
        # Calculate Y to be in the middle
        pos_y = (screen_h - win_h) // 2
        pos_x = 0  # Left edge
        
        root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        root.attributes('-topmost', True)
        root.after(1500, lambda: root.attributes('-topmost', False))
        root.resizable(False, False)
        
        # Styling
        label = tk.Label(root, text=instruction_text, wraplength=280, justify='left',
            padx=15, pady=20, font=("Helvetica", 12, "bold"), fg="#2c3e50", bg="#f9f9f9")
        label.pack(expand=True, fill='both')
        
        hint_frame = tk.Frame(root, bg="#f9f9f9")
        hint_frame.pack(fill='x', pady=10)
        
        tk.Label(hint_frame, text="Press", font=("Arial", 10), bg="#f9f9f9", fg="#7f8c8d").pack(side="top")
        tk.Label(hint_frame, text="[ ENTER ]", font=("Courier", 12, "bold"), bg="#f9f9f9", fg="#e74c3c").pack(side="top")
        tk.Label(hint_frame, text="to finish recording", font=("Arial", 10), bg="#f9f9f9", fg="#7f8c8d").pack(side="top")

        btn = tk.Button(root, text="Minimize", command=root.iconify, font=("Arial", 9))
        btn.pack(pady=10)
        
        root.lift()
        root.mainloop()
    
    thread = threading.Thread(target=create_window, daemon=True)
    thread.start()

# ---------------------------------------------------------
# 2. ISAAC SIM SETUP
# ---------------------------------------------------------

config = {
    "launch_config": {
        "renderer": "RayTracedLighting",  
        "headless": True,
    },
    "resolution": [512, 512],
    "writer": "BasicWriter",
}

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, help="0-56")
parser.add_argument("--task", default="fine", help="fine | coarse")
parser.add_argument("--work_dir", default="/data/lsh/isaac_code/demo")
parser.add_argument("--headless", default=False, action="store_true")
parser.add_argument("--mode", default="manual", help="manual | teleop | policy")
parser.add_argument("--render", default=False, action="store_true")
parser.add_argument("--record", default=True, action="store_true")
parser.add_argument("--agent", default="go2", choices=["go2", "floating"], help="Agent backend to run.")
parser.add_argument("--go2-controller", default="physics", choices=["physics", "kinematic"], help="Go2 controller backend.")
parser.add_argument("--camera_mode", default="floating")
parser.add_argument("--camera_height", default=None, type=float, help="Camera height from the ground.")
parser.add_argument("--camera-follow-gt", default=False, action="store_true", help="Move only the floating camera along the current episode reference_path.")
parser.add_argument("--camera-gt-speed", default=0.75, type=float, help="Camera-only GT trajectory speed in m/s.")
parser.add_argument("--camera-gt-waypoint-radius", default=0.05, type=float, help="Waypoint acceptance radius for --camera-follow-gt.")
parser.add_argument("--camera-gt-record-every", default=10, type=int, help="Record camera-only GT trajectory every N follower steps.")
parser.add_argument("--camera-gt-print-every", default=100, type=int, help="Print camera-only GT follower status every N frames.")
parser.add_argument("--camera-gt-max-frames", default=10000, type=int, help="Headless safety limit for --camera-follow-gt.")
parser.add_argument("--sync-viewport-camera", dest="sync_viewport_camera", default=True, action=argparse.BooleanOptionalAction, help="Force the GUI viewport to render from /World/FloatingCamera for floating-camera runs.")
parser.add_argument("--no_sync_viewport_camera", dest="sync_viewport_camera", action="store_false", help=argparse.SUPPRESS)
parser.add_argument("--go2-usd-path", default=None, help="Override path or URL to Unitree Go2 USD.")
parser.add_argument("--go2-prim-path", default="/World/Go2", help="Stage prim path for the Go2 agent.")
parser.add_argument("--go2-base-height", default=0.4, type=float, help="Go2 root height above the scene floor.")
parser.add_argument("--go2-move-speed", default=0.45, type=float, help="Go2 kinematic forward speed in m/s.")
parser.add_argument("--go2-turn-speed", default=75.0, type=float, help="Go2 kinematic yaw speed in deg/s.")
parser.add_argument("--go2-camera-distance", default=1.0, type=float, help="Kinematic follow camera distance behind Go2.")
parser.add_argument("--go2-camera-height-offset", default=0.78, type=float, help="Kinematic follow camera height above Go2 root.")
parser.add_argument("--go2-camera-target-distance", default=1.15, type=float, help="Kinematic follow camera look-ahead distance.")
parser.add_argument("--go2-camera-target-height-offset", default=0.35, type=float, help="Kinematic follow camera target height above Go2 root.")
parser.add_argument("--go2-camera-horizontal-aperture", default=100.0, type=float, help="Kinematic follow camera horizontal aperture.")
parser.add_argument("--go2-heading-offset-deg", default=0.0, type=float, help="Extra heading offset for the Go2 USD model.")
parser.add_argument("--go2-enable-physics", default=False, action="store_true", help="Keep Go2 physics schemas enabled.")
parser.add_argument("--isaaclab-root", default="/home/kaurml/IsaacLab", help="Path to the IsaacLab checkout for physics Go2.")
parser.add_argument("--go2-physics-task", default="Isaac-Velocity-Rough-Unitree-Go2-Play-v0", help="IsaacLab Go2 velocity task id. Rough Go2 is the default because Kujiale scenes can contain thresholds and small height changes.")
parser.add_argument("--go2-physics-scene", default="kujiale", choices=["kujiale", "plane"], help="Physics scene backend.")
parser.add_argument("--go2-physics-checkpoint", default=None, help="Explicit RSL-RL checkpoint for the physics Go2 policy.")
parser.add_argument("--go2-physics-fallback-checkpoint", default="/home/kaurml/NaVILA-Bench-4.1/logs/rsl_rl/go2_base/2024-12-10_21-44-44/model_1999.pt", help="Local Go2 base checkpoint fallback.")
parser.add_argument("--go2-physics-experience", default="isaacsim.exp.base.python.kit", help="Isaac Sim experience file used by IsaacLab AppLauncher.")
parser.add_argument("--go2-physics-asset-root", default=None, help="Isaac Sim 4.5 asset root URL/path used for IsaacLab Go2 USD assets.")
parser.add_argument("--go2-physics-locomotion-adapter", default="walkable_proxy", choices=["none", "floor_proxy", "walkable_proxy"], help="Kujiale locomotion adapter. walkable_proxy builds a smoothed walkable-surface mesh for the rough-policy height scanner; floor_proxy is kept as a legacy alias.")
parser.add_argument("--go2-physics-height-proxy-prim", default="/World/ground/go2_height_floor_proxy", help="Stage prim path for the generated Kujiale height-scanner walkable proxy.")
parser.add_argument("--go2-physics-height-proxy-mode", default="walkable", choices=["raw", "walkable"], help="Height proxy generation mode. walkable rasterizes low horizontal surfaces and smooths small gaps/steps.")
parser.add_argument("--go2-physics-height-proxy-source-scopes", default="floor,other", help="Comma-separated Kujiale mesh scopes to consider for the height-scanner proxy.")
parser.add_argument("--go2-physics-height-proxy-grid-resolution", default=0.10, type=float, help="Grid resolution in meters for walkable height proxy generation.")
parser.add_argument("--go2-physics-height-proxy-gap-fill", default=0.45, type=float, help="Maximum distance in meters to fill small walkable proxy gaps.")
parser.add_argument("--go2-physics-height-proxy-smooth-passes", default=2, type=int, help="Number of smoothing passes for the walkable height proxy.")
parser.add_argument("--go2-physics-height-proxy-min-normal-z", default=0.75, type=float, help="Minimum absolute triangle normal z component for walkable proxy source triangles.")
parser.add_argument("--go2-physics-height-proxy-max-step-height", default=0.18, type=float, help="Maximum height above the median floor height to include as walkable threshold/step geometry.")
parser.add_argument("--go2-physics-height-proxy-max-drop-height", default=0.20, type=float, help="Maximum height below the median floor height to include as walkable lowered floor geometry.")
parser.add_argument("--go2-physics-collision-mode", default="nav_collision", choices=["all", "nav_collision", "floor_only", "floor_walls", "none"], help="Which Kujiale mesh scopes receive runtime static collision. nav_collision is the default navigation mode: it preserves authored collisions and only adds missing runtime collision to floor meshes. floor_only is kept as a legacy alias.")
parser.add_argument("--go2-physics-camera", default="viz_rgb_camera", choices=["rgbd_camera", "viz_rgb_camera", "robot_camera", "close_follow", "wide_follow", "first_person", "fixed_overview"], help="Default viewport camera for physics Go2.")
parser.add_argument("--go2-physics-use-usd-camera-prim", default=False, action="store_true", help="Use the authored USD camera prim for rgbd_camera/viz_rgb_camera instead of the tensor-driven viewport camera.")
parser.add_argument("--go2-physics-render-interval", default=None, type=int, help="Physics substeps between IsaacLab render calls. Defaults to the Go2 task decimation, normally 4 = 50 Hz.")
parser.add_argument("--go2-physics-render-sync", default=None, action=argparse.BooleanOptionalAction, help="Force an extra viewport render after Go2 pose/camera updates in GUI mode. Defaults to auto: on for GUI, off for headless.")
parser.add_argument("--go2-physics-render-sync-every", default=1, type=int, help="Run extra GUI viewport render sync every N Go2 control steps when render sync is enabled.")
parser.add_argument("--go2-physics-render-width", default=960, type=int, help="Isaac Sim render width for physics Go2.")
parser.add_argument("--go2-physics-render-height", default=540, type=int, help="Isaac Sim render height for physics Go2.")
parser.add_argument("--go2-physics-anti-aliasing", default=0, type=int, help="Isaac Sim anti-aliasing mode for physics Go2. 0 is fastest.")
parser.add_argument("--go2-physics-renderer", default="RaytracedLighting", help="Isaac Sim renderer for physics Go2.")
parser.add_argument("--go2-physics-multi-gpu", default=False, action=argparse.BooleanOptionalAction, help="Enable Isaac Sim multi-GPU rendering for physics Go2.")
parser.add_argument("--go2-physics-use-fabric", default=True, action=argparse.BooleanOptionalAction, help="Use PhysX Fabric so articulated Go2 poses are flushed to the GUI viewport. Keep enabled for rendering.")
parser.add_argument("--go2-physics-kit-log-level", default="error", choices=["verbose", "info", "warning", "error", "fatal"], help="Isaac/Kit console log level. Default hides noisy Kit warnings while preserving errors and Go2 diagnostics.")
parser.add_argument("--go2-physics-lin-vel", default=1.0, type=float, help="W forward velocity command in m/s.")
parser.add_argument("--go2-physics-backward-lin-vel", default=None, type=float, help="S backward command magnitude. Defaults to --go2-physics-lin-vel.")
parser.add_argument("--go2-physics-strafe-vel", default=0.25, type=float, help="Q/E lateral velocity command magnitude in m/s.")
parser.add_argument("--go2-physics-ang-vel", default=1.0, type=float, help="A/D yaw velocity command magnitude in rad/s.")
parser.add_argument("--go2-physics-warmup-steps", default=80, type=int, help="Zero-command warmup steps before keyboard control.")
parser.add_argument("--go2-physics-print-every", default=50, type=int, help="Print physics Go2 diagnostics every N control steps.")
parser.add_argument("--go2-physics-diagnostics", default=False, action="store_true", help="Print detailed Go2 scene/sensor/contact diagnostics for Kujiale locomotion debugging.")
parser.add_argument("--go2-physics-diagnostics-every", default=0, type=int, help="Detailed diagnostics interval in control steps. Defaults to --go2-physics-print-every when 0.")
parser.add_argument("--go2-physics-max-headless-steps", default=2000, type=int, help="Safety limit for headless physics runs without a scripted command.")
parser.add_argument("--go2-physics-poll-keyboard", default=True, action=argparse.BooleanOptionalAction, help="Poll held WASD keys from Carb/X11 every control step.")
parser.add_argument("--go2-physics-terminal-keyboard", default=True, action=argparse.BooleanOptionalAction, help="Also accept WASD from the launching terminal if Isaac viewport focus misses keyboard events.")
parser.add_argument("--go2-physics-terminal-key-hold-s", default=0.35, type=float, help="Seconds to keep a terminal keypress active while relying on terminal autorepeat.")
parser.add_argument("--go2-physics-key-release-grace-s", default=0.25, type=float, help="Seconds to wait before accepting key release events while polling held keys.")
parser.add_argument("--go2-physics-debug-keyboard", default=False, action="store_true", help="Print Go2 physics keyboard command diagnostics.")
parser.add_argument("--go2-physics-debug-keyboard-poll-every", default=20, type=int, help="Print held key state every N control steps when keyboard debug is enabled.")
parser.add_argument("--go2-physics-disable-base-contact-termination", default=False, action="store_true", help="Disable base_contact termination for custom-scene debugging.")
parser.add_argument("--go2-physics-scripted-command", default=None, choices=["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"], help="Headless smoke-test command for the physics Go2 controller.")
parser.add_argument("--go2-physics-scripted-steps", default=300, type=int, help="Number of steps for --go2-physics-scripted-command.")
parser.add_argument("--go2-physics-follow-gt", default=False, action="store_true", help="Follow the current episode reference_path with a simple velocity-command waypoint controller.")
parser.add_argument("--go2-physics-gt-waypoint-radius", default=0.35, type=float, help="Waypoint acceptance radius in meters for --go2-physics-follow-gt.")
parser.add_argument("--go2-physics-gt-final-radius", default=0.55, type=float, help="Final-goal acceptance radius in meters for --go2-physics-follow-gt.")
parser.add_argument("--go2-physics-gt-lookahead-distance", default=0.55, type=float, help="Lookahead distance in meters along the GT path for velocity tracking.")
parser.add_argument("--go2-physics-gt-min-waypoint-spacing", default=0.05, type=float, help="Drop duplicate GT waypoints closer than this distance in meters.")
parser.add_argument("--go2-physics-gt-linear-gain", default=0.9, type=float, help="Linear speed gain for GT trajectory following.")
parser.add_argument("--go2-physics-gt-lateral-gain", default=0.8, type=float, help="Lateral speed gain for GT trajectory following.")
parser.add_argument("--go2-physics-gt-yaw-gain", default=1.6, type=float, help="Yaw speed gain for GT trajectory following.")
parser.add_argument("--go2-physics-gt-turn-in-place-angle", default=0.9, type=float, help="Yaw error in radians above which the GT follower turns in place before walking forward.")
parser.add_argument("--go2-physics-gt-min-forward-vel", default=0.12, type=float, help="Minimum forward command while tracking a non-final GT waypoint.")
parser.add_argument("--go2-physics-gt-max-steps", default=0, type=int, help="Maximum control steps for GT following. 0 uses the normal headless safety limit in headless mode and no extra limit in GUI mode.")
args, unknown_args = parser.parse_known_args()

if args.camera_height is None:
    args.camera_height = 1.5

if args.go2_physics_backward_lin_vel is None:
    args.go2_physics_backward_lin_vel = args.go2_physics_lin_vel

if args.camera_follow_gt and args.agent != "floating":
    print("[INFO] --camera-follow-gt uses the floating camera agent; switching --agent to floating.")
    args.agent = "floating"

def load_demo_episode(task_name, episode_index):
    with open(f"{task_name}_grained_demo.json", "r") as f:
        demo_data = json.load(f)
    episode = demo_data[int(episode_index)]
    instruction = episode["instruction"]["instruction_text"]
    if task_name == "coarse":
        instruction = instruction["natural"]
    return episode, instruction


if args.agent == "go2" and args.go2_controller == "physics":
    cur_episode, instruction_text = load_demo_episode(args.task, args.index)
    if not args.headless:
        show_instruction_left_centered(instruction_text)

    from go2_physics_teleop import run_go2_physics_episode

    output_dir = args.work_dir
    os.makedirs(output_dir, exist_ok=True)
    print("Agent: go2")
    print("Go2 controller: physics")
    print(f"Instruction: {instruction_text}")
    print(" >>> PRESS [ENTER] TO FINISH RECORDING <<<")
    saved_csv_path, success = run_go2_physics_episode(cur_episode, args, os.getcwd())

    if not success:
        sys.exit(0)

    if saved_csv_path and os.path.exists(saved_csv_path):
        print("\nEvaluating Performance...")
        _, metrics = evaluate_single_episode(saved_csv_path, cur_episode)
        if not args.headless:
            show_beautiful_popup(metrics)

    print("Done.")
    sys.exit(0)

from isaacsim import SimulationApp
import carb

config["launch_config"]["headless"] = args.headless
simulation_app = SimulationApp(config["launch_config"])

def rotation_from_direction(direction, up_vector=np.array([0, 0, 1])):
    from scipy.spatial.transform import Rotation as R
    direction = np.array(direction, dtype=np.float64)
    if np.linalg.norm(direction) < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    forward = direction / np.linalg.norm(direction)
    right = np.cross(up_vector, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm
    up = np.cross(forward, right)
    rot_mat = np.column_stack((forward, right, up))
    quat = R.from_matrix(rot_mat).as_quat()
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def yaw_from_quaternion_wxyz(quat):
    from scipy.spatial.transform import Rotation as R
    quat = np.array(quat, dtype=np.float64)
    if quat.shape[0] != 4 or np.linalg.norm(quat) < 1e-6:
        return 0.0
    quat = quat / np.linalg.norm(quat)
    return float(R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")[2])


def heading_from_yaw(yaw):
    return np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)


def go2_heading_from_yaw(yaw):
    return np.array([math.sin(-yaw), math.cos(-yaw), 0.0], dtype=np.float64)


def quaternion_from_yaw(yaw):
    half_yaw = yaw * 0.5
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float64)


def define_camera(stage, camera_path, focal_length=10.0, horizontal_aperture=20.0, vertical_aperture=20.0):
    from pxr import UsdGeom

    camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    camera_prim.CreateFocalLengthAttr(float(focal_length))
    camera_prim.CreateClippingRangeAttr().Set((0.01, 1000.0))
    camera_prim.CreateVerticalApertureAttr(float(vertical_aperture))
    camera_prim.CreateHorizontalApertureAttr(float(horizontal_aperture))
    return camera_prim


def set_xform_pose(stage, prim_path, position, orientation_wxyz=None):
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    if orientation_wxyz is not None:
        quat = np.array(orientation_wxyz, dtype=np.float64)
        if np.linalg.norm(quat) < 1e-6:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat = quat / np.linalg.norm(quat)
        orient_op = xformable.AddOrientOp()
        orient_op.Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))


def set_camera_pose(stage, camera_path, position, look_direction):
    from pxr import Gf, UsdGeom
    from scipy.spatial.transform import Rotation as R

    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsValid():
        return

    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    orientation = rotation_from_direction(look_direction)
    rotation = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
    euler = rotation.as_euler("xyz", degrees=True)
    rotate_op = xformable.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3f(float(euler[0] + 90), float(euler[1]), float(euler[2] - 90)))


def join_asset_path(root_path, relative_path):
    return root_path.rstrip("/") + "/" + relative_path.lstrip("/")


def go2_usd_from_isaaclab_root(root_path):
    if root_path.endswith(".usd"):
        return root_path
    return join_asset_path(root_path, "Robots/Unitree/Go2/go2.usd")


def resolve_go2_usd_path():
    usd_path = args.go2_usd_path or os.environ.get("GO2_USD_PATH")
    if usd_path:
        return usd_path

    isaaclab_root = os.environ.get("ISAACLAB_NUCLEUS_DIR") or os.environ.get("ISAACLAB_ASSETS_ROOT")
    if isaaclab_root:
        return go2_usd_from_isaaclab_root(isaaclab_root)

    for module_name in ("omni.isaac.lab.utils.assets", "isaaclab.utils.assets"):
        try:
            import importlib

            assets_module = importlib.import_module(module_name)
            isaaclab_root = getattr(assets_module, "ISAACLAB_NUCLEUS_DIR", None)
            if isaaclab_root:
                return go2_usd_from_isaaclab_root(isaaclab_root)
        except Exception:
            pass

    try:
        cloud_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
        if cloud_root:
            return join_asset_path(cloud_root, "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd")
    except Exception:
        pass

    assets_root = os.environ.get("ISAAC_ASSETS_ROOT")
    if not assets_root:
        try:
            from isaacsim.storage.native import get_assets_root_path

            assets_root = get_assets_root_path()
        except Exception as exc:
            raise RuntimeError(
                "Could not resolve the Go2 USD path. Set --go2-usd-path, "
                "GO2_USD_PATH, ISAACLAB_NUCLEUS_DIR, or ISAAC_ASSETS_ROOT."
            ) from exc

    assets_root = assets_root.rstrip("/")
    if assets_root.endswith("/IsaacLab"):
        return join_asset_path(assets_root, "Robots/Unitree/Go2/go2.usd")
    if assets_root.endswith("/Isaac"):
        return assets_root + "/IsaacLab/Robots/Unitree/Go2/go2.usd"
    if assets_root.endswith("/Isaac/IsaacLab"):
        return assets_root + "/Robots/Unitree/Go2/go2.usd"
    if assets_root.endswith("/Isaac/4.5"):
        return assets_root + "/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
    return assets_root + "/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"


def disable_referenced_physics(stage, root_path):
    from pxr import Sdf, Usd

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return

    disabled_count = 0
    for prim in Usd.PrimRange(root_prim):
        rigid_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if rigid_attr.IsValid():
            rigid_attr.Set(False)
            disabled_count += 1

        collision_attr = prim.GetAttribute("physics:collisionEnabled")
        if collision_attr.IsValid():
            collision_attr.Set(False)

        if rigid_attr.IsValid() or prim.HasAttribute("physics:kinematicEnabled"):
            kinematic_attr = prim.GetAttribute("physics:kinematicEnabled")
            if not kinematic_attr.IsValid():
                kinematic_attr = prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool)
            kinematic_attr.Set(True)

    if disabled_count:
        print(f"Go2 visual agent: disabled physics on {disabled_count} rigid bodies.")


class FloatingCameraController:
    def __init__(self, world, scene_data):
        self.world = world
        self.state = 0

        self.camera_height = args.camera_height 
        self.floor_z = float(scene_data["start_location"][2])
        self.start_position = np.array(scene_data["start_location"], dtype=np.float64) + np.array([0, 0, self.camera_height])
        self.start_orientation = np.array(scene_data["start_orientation"])
        self.start_yaw = yaw_from_quaternion_wxyz(self.start_orientation)
        self.start_look_direction = heading_from_yaw(self.start_yaw)

        stage = simulation_app.context.get_stage()
        self.camera_path = "/World/FloatingCamera"
        define_camera(stage, self.camera_path)

        self.camera = self.camera_path 
        self.current_position = self.start_position.copy()
        self.current_orientation = self.start_orientation.copy()
        self.look_direction = self.start_look_direction.copy()

        self.move_speed = 0.25  
        self.turn_speed = 15.0  

        self.traj = scene_data.get("traj", None)
        if self.traj is not None:
            self.traj = np.asarray(self.traj, dtype=np.float64)
        self.traj_index = 0
        self.traj_dir = 1
        self.mode = args.mode
        self._base_command = np.array([0.0, 0.0, 0.0])
        self.mission_complete = False 
        self._follow_step = 0
        self._last_gt_print_index = -1
        self.recorded_transforms = []
        self._viewport_camera_synced = False
        self._viewport_camera_warning_shown = False

        if args.camera_follow_gt:
            if self.traj is None or len(self.traj) < 2:
                raise RuntimeError("Camera GT follower requires an episode reference_path with at least two waypoints.")
            print(
                "[INFO] Camera-only GT follower enabled: "
                f"waypoints={len(self.traj)}, speed={args.camera_gt_speed:.2f}m/s, "
                f"radius={args.camera_gt_waypoint_radius:.2f}m"
            )

    def reset(self):
        self.current_position = self.start_position.copy()
        self.current_orientation = self.start_orientation.copy()
        self.look_direction = self.start_look_direction.copy()
        self._update_camera()
        self.state = 0
        self.traj_index = 0
        self.traj_dir = 1
        self.mission_complete = False
        self._follow_step = 0
        self._last_gt_print_index = -1
        self.recorded_transforms = []
        print('=' * 10, "reset", "=" * 10)
        self.sync_viewport_camera(force=True)

    def sync_viewport_camera(self, force=False):
        if args.headless or not args.sync_viewport_camera:
            return False

        try:
            from omni.kit.viewport.utility import get_active_viewport
            from pxr import Sdf

            viewport = get_active_viewport()
            if viewport is None:
                return False

            active_path = viewport.camera_path
            active_path_str = active_path.pathString if hasattr(active_path, "pathString") else str(active_path)
            if force or active_path_str != self.camera_path:
                viewport.camera_path = Sdf.Path(self.camera_path)
                print(f"Viewport camera set to {self.camera_path}")
            self._viewport_camera_synced = True
            return True
        except Exception as exc:
            if not self._viewport_camera_warning_shown:
                print(f"[WARN] Could not set viewport camera to {self.camera_path}: {exc}")
                self._viewport_camera_warning_shown = True
            return False

    def init_manual(self):
        import omni.appwindow
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._base_command = np.array([0.0, 0.0, 0.0])
        pos_del = 1
        self._input_keyboard_mapping = {
            "W": [pos_del, 0., 0.],
            "A": [0., 0., pos_del],
            "D": [0., 0., -pos_del],
            "S": [-pos_del, 0., 0.]
        }

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Check for ENTER key to Finish Mission
            if event.input.name == "ENTER":
                print("ENTER pressed - Mission Complete.")
                self.mission_complete = True
                self.state = "Done"
                return True
                
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _update_camera(self):
        stage = simulation_app.context.get_stage()
        set_camera_pose(stage, self.camera_path, self.current_position, self.look_direction)

    def get_camera_transform(self):
        return {
            "pos_x": self.current_position[0],
            "pos_y": self.current_position[1],
            "pos_z": self.current_position[2],
            "look_x": self.look_direction[0],
            "look_y": self.look_direction[1],
            "look_z": self.look_direction[2],
        }

    def _record_gt_transform(self):
        if not args.record:
            return
        transform = self.get_camera_transform()
        transform["frame"] = self._follow_step
        self.recorded_transforms.append(transform)

    def run(self, step_size):
        from scipy.spatial.transform import Rotation as R
        if args.camera_follow_gt:
            self._run_gt_trajectory(step_size)
        elif self.mode == 'manual':
            if abs(self._base_command[0]) > 0:
                self.current_position += self.look_direction * self._base_command[0] * self.move_speed * step_size
            if abs(self._base_command[1]) > 0:
                right_dir = np.cross(self.look_direction, np.array([0, 0, 1]))
                right_dir = right_dir / np.linalg.norm(right_dir)
                self.current_position += right_dir * self._base_command[1] * self.move_speed * step_size
            if abs(self._base_command[2]) > 0:
                angle = self._base_command[2] * self.turn_speed * step_size
                r = R.from_euler('z', angle, degrees=True)
                self.look_direction = r.apply(self.look_direction)
            self.current_position[2] = self.floor_z + self.camera_height
        self._update_camera()

    def _run_gt_trajectory(self, step_size):
        if self.mission_complete:
            return

        self._follow_step += 1
        radius = max(float(args.camera_gt_waypoint_radius), 1e-3)
        speed = max(float(args.camera_gt_speed), 0.0)

        current_xy = self.current_position[:2]
        while self.traj_index < len(self.traj) - 1:
            waypoint_xy = self.traj[self.traj_index, :2]
            if np.linalg.norm(waypoint_xy - current_xy) > radius:
                break
            self.traj_index += 1

        target_xy = self.traj[self.traj_index, :2]
        delta_xy = target_xy - current_xy
        distance = float(np.linalg.norm(delta_xy))

        if self.traj_index >= len(self.traj) - 1 and distance <= radius:
            self.current_position[:2] = target_xy
            self.current_position[2] = self.floor_z + self.camera_height
            if len(self.traj) >= 2:
                final_dir = self.traj[-1, :2] - self.traj[-2, :2]
                if np.linalg.norm(final_dir) > 1e-6:
                    self.look_direction = np.array([final_dir[0], final_dir[1], 0.0], dtype=np.float64)
                    self.look_direction = self.look_direction / np.linalg.norm(self.look_direction)
            print(
                "[INFO] Camera-only GT follower finished: "
                f"target={self.traj_index}/{len(self.traj)-1}, "
                f"pos=[{self.current_position[0]:+.2f}, {self.current_position[1]:+.2f}, {self.current_position[2]:+.2f}]"
            )
            self._record_gt_transform()
            self.mission_complete = True
            return

        if distance > 1e-6 and speed > 0.0:
            direction_xy = delta_xy / distance
            step_distance = min(speed * float(step_size), distance)
            self.current_position[:2] += direction_xy * step_distance
            self.look_direction = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)

        self.current_position[2] = self.floor_z + self.camera_height
        record_every = max(int(args.camera_gt_record_every), 1)
        if self._follow_step == 1 or self._follow_step % record_every == 0 or self.traj_index != self._last_gt_print_index:
            self._record_gt_transform()
        print_every = max(int(args.camera_gt_print_every), 1)
        if self._follow_step % print_every == 0 or self.traj_index != self._last_gt_print_index:
            print(
                "[INFO] camera_gt "
                f"step={self._follow_step} target={self.traj_index}/{len(self.traj)-1} "
                f"dist={distance:.2f} pos=[{self.current_position[0]:+.2f}, {self.current_position[1]:+.2f}, {self.current_position[2]:+.2f}]"
            )
            self._last_gt_print_index = self.traj_index


class Go2AgentController:
    def __init__(self, world, scene_data):
        self.world = world
        self.state = 0
        self.mode = args.mode
        self.robot_root_path = args.go2_prim_path
        self.robot_asset_path = self.robot_root_path.rstrip("/") + "/Asset"
        self.camera_path = "/World/FloatingCamera"
        self.floor_z = float(scene_data["start_location"][2])
        self.base_height = args.go2_base_height
        self.move_speed = args.go2_move_speed
        self.turn_speed = args.go2_turn_speed
        self.camera_distance = args.go2_camera_distance
        self.camera_height_offset = args.go2_camera_height_offset
        self.camera_target_distance = args.go2_camera_target_distance
        self.camera_target_height_offset = args.go2_camera_target_height_offset
        self.camera_horizontal_aperture = args.go2_camera_horizontal_aperture
        self.heading_offset = math.radians(args.go2_heading_offset_deg)

        self.start_position = np.array(scene_data["start_location"], dtype=np.float64)
        self.start_position[2] = self.floor_z + self.base_height
        self.start_orientation = np.array(scene_data["start_orientation"], dtype=np.float64)
        self.start_yaw = yaw_from_quaternion_wxyz(self.start_orientation)
        self.current_position = self.start_position.copy()
        self.yaw = self.start_yaw
        self.look_direction = go2_heading_from_yaw(self.yaw)
        self._base_command = np.array([0.0, 0.0, 0.0])
        self.mission_complete = False

        stage = simulation_app.context.get_stage()
        self._spawn_go2(stage)
        define_camera(
            stage,
            self.camera_path,
            horizontal_aperture=self.camera_horizontal_aperture,
            vertical_aperture=self.camera_horizontal_aperture,
        )
        self._update_robot()
        self._update_camera()

    def _spawn_go2(self, stage):
        from pxr import UsdGeom

        UsdGeom.Xform.Define(stage, self.robot_root_path)
        asset_prim = UsdGeom.Xform.Define(stage, self.robot_asset_path).GetPrim()
        go2_usd_path = resolve_go2_usd_path()
        asset_prim.GetReferences().AddReference(go2_usd_path)
        print(f"Loaded Go2 agent USD: {go2_usd_path}")

        if not args.go2_enable_physics:
            disable_referenced_physics(stage, self.robot_asset_path)

    def reset(self):
        self.current_position = self.start_position.copy()
        self.yaw = self.start_yaw
        self.look_direction = go2_heading_from_yaw(self.yaw)
        self._base_command = np.array([0.0, 0.0, 0.0])
        self.mission_complete = False
        self._update_robot()
        self._update_camera()
        print('=' * 10, "reset go2", "=" * 10)

    def init_manual(self):
        import omni.appwindow

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._base_command = np.array([0.0, 0.0, 0.0])
        pos_del = 1
        self._input_keyboard_mapping = {
            "W": [pos_del, 0.0, 0.0],
            "A": [0.0, 0.0, pos_del],
            "D": [0.0, 0.0, -pos_del],
            "S": [-pos_del, 0.0, 0.0],
        }

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "ENTER":
                print("ENTER pressed - Mission Complete.")
                self.mission_complete = True
                self.state = "Done"
                return True

            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _update_robot(self):
        stage = simulation_app.context.get_stage()
        orientation = quaternion_from_yaw(self.yaw + self.heading_offset)
        set_xform_pose(stage, self.robot_root_path, self.current_position, orientation)

    def _camera_position(self):
        camera_position = self.current_position.copy()
        camera_position[:2] -= self.look_direction[:2] * self.camera_distance
        camera_position[2] = self.current_position[2] + self.camera_height_offset
        return camera_position

    def _camera_target(self):
        camera_target = self.current_position.copy()
        camera_target[:2] += self.look_direction[:2] * self.camera_target_distance
        camera_target[2] = self.current_position[2] + self.camera_target_height_offset
        return camera_target

    def _update_camera(self):
        stage = simulation_app.context.get_stage()
        camera_position = self._camera_position()
        camera_target = self._camera_target()
        set_camera_pose(stage, self.camera_path, camera_position, camera_target - camera_position)

    def get_camera_transform(self):
        camera_position = self._camera_position()
        camera_target = self._camera_target()
        camera_look_direction = camera_target - camera_position
        camera_look_direction = camera_look_direction / np.linalg.norm(camera_look_direction)
        return {
            "agent": "go2",
            "pos_x": self.current_position[0],
            "pos_y": self.current_position[1],
            "pos_z": self.current_position[2],
            "camera_pos_x": camera_position[0],
            "camera_pos_y": camera_position[1],
            "camera_pos_z": camera_position[2],
            "look_x": camera_look_direction[0],
            "look_y": camera_look_direction[1],
            "look_z": camera_look_direction[2],
            "yaw": self.yaw,
        }

    def run(self, step_size):
        if self.mode == "manual":
            if abs(self._base_command[2]) > 0:
                angle = math.radians(self._base_command[2] * self.turn_speed * step_size)
                self.yaw += angle
                self.look_direction = go2_heading_from_yaw(self.yaw)
            if abs(self._base_command[0]) > 0:
                self.current_position += self.look_direction * self._base_command[0] * self.move_speed * step_size
            self.current_position[2] = self.floor_z + self.base_height

        self._update_robot()
        self._update_camera()


reset_needed = False
first_step = True

def run(scene_data, output_dir=None, episode_id=None):
    from pxr import Sdf
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import define_prim
    import omni.replicator.core as rep

    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 200.0, rendering_dt=8.0 / 200.0)
    my_world.scene.add_default_ground_plane(z_position=0, name="default_ground_plane", prim_path="/World/defaultGroundPlane")

    if "usd_path" in scene_data:
        prim = define_prim("/World/Ground", "Xform")
        asset_path = scene_data["usd_path"]
        prim.GetReferences().AddReference(asset_path, "/Root")

    if args.agent == "go2":
        agent_controller = Go2AgentController(world=my_world, scene_data=scene_data)
    else:
        agent_controller = FloatingCameraController(world=my_world, scene_data=scene_data)

    if args.mode == "manual" and not args.camera_follow_gt:
        agent_controller.init_manual()

    stage = simulation_app.context.get_stage()
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

    my_world.reset()
    agent_controller.reset()

    global reset_needed
    def on_physics_step(step_size):
        global first_step, reset_needed
        if first_step:
            agent_controller.reset()
            first_step = False
        elif reset_needed:
            my_world.reset(True)
            reset_needed = False
            first_step = True
        else:
            agent_controller.run(step_size)

    my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)
    
    camera_transforms = []
    frame = 0
    mission_complete_signal = False

    while simulation_app.is_running():
        my_world.step(render=True)
        if hasattr(agent_controller, "sync_viewport_camera"):
            agent_controller.sync_viewport_camera()
        if my_world.is_stopped() and not reset_needed: reset_needed = True
        
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                agent_controller.reset()
                frame = 0
                reset_needed = False
                camera_transforms = []

            if args.render:
                rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
            
            if args.record and (frame % 50 == 0):
                transform = agent_controller.get_camera_transform()
                transform["frame"] = frame
                camera_transforms.append(transform)

            if agent_controller.mission_complete:
                mission_complete_signal = True
                break

            headless_limit = args.camera_gt_max_frames if args.camera_follow_gt else 2000
            if args.headless and frame > headless_limit:
                break
            frame += 1

    if not mission_complete_signal:
        print("\n\n [ABORT] Simulation window closed without pressing ENTER.")
        return None, False

    if args.camera_follow_gt and getattr(agent_controller, "recorded_transforms", None):
        camera_transforms = agent_controller.recorded_transforms

    if args.record:
        print("save camera_transforms", len(camera_transforms))
        df = pd.DataFrame(camera_transforms)
        csv_save_path = os.path.join(output_dir, f"{episode_id}.csv")
        df.to_csv(csv_save_path, index=False)
        return csv_save_path, True
    
    return None, True

if __name__ == '__main__':
    work_dir = os.getcwd()

    task = args.task
    print(task)

    with open(f"{task}_grained_demo.json", 'r') as f:
        data = json.load(f)

    cur_episode = data[int(args.index)]
    scene_id = cur_episode['scan']
    loc = list(cur_episode['start_position'])
    ori = list(cur_episode['start_rotation'])
    instruction_text = cur_episode['instruction']['instruction_text']

    if task == "coarse":
        instruction_text = instruction_text["natural"]
    
    # 1. SHOW INSTRUCTION (Left Center)
    if not args.headless:
        show_instruction_left_centered(instruction_text)
    usd_path = os.path.join(work_dir, scene_id, f'{scene_id}.usda')
    
    traj = cur_episode.get("reference_path")
    if not traj:
        traj = []
        for i in range(10):
            traj.append([loc[0] + i * 0.5, loc[1], loc[2]])
    scene_data = {
        "usd_path": usd_path,
        "start_location": loc,
        "start_orientation": ori,
        "traj": np.array(traj),
    }
    
    output_dir = args.work_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Agent: {args.agent}")
    print(f"Instruction: {instruction_text}")
    print(" >>> PRESS [ENTER] TO FINISH RECORDING <<<")
    
    # 2. RUN SIM
    saved_csv_path, success = run(scene_data, output_dir, cur_episode['episode_id'])
    
    # simulation_app.close()

    if not success:
        sys.exit(0)

    # 3. EVALUATE & SHOW BEAUTIFUL POPUP
    if saved_csv_path and os.path.exists(saved_csv_path):
        print("\nEvaluating Performance...")
        # Get metrics dict
        _, metrics = evaluate_single_episode(saved_csv_path, cur_episode)
        
        # Show Beautiful GUI
        if not args.headless:
            show_beautiful_popup(metrics)
    
    print(f"Done.")
