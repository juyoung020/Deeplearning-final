from __future__ import annotations

import argparse
import atexit
import ctypes
import ctypes.util
import importlib.metadata as metadata  ###
import math
import os
import re  # ###2
import shlex
import select
import sys
import termios
import time
import traceback
import tty
from types import SimpleNamespace

import numpy as np
import pandas as pd

DEFAULT_ISAAC_45_ASSET_ROOT = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
PERSPECTIVE_CAMERA_PATH = "/OmniverseKit_Persp"
_ACTIVE_VIEWPORT_CAMERA_PATH = None
KUJIALE_HEIGHT_PROXY_ENTITY_NAME = "kujiale_locomotion_height_proxy"


def _prepend_path(path):
    if path and os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)


def configure_isaaclab_paths(isaaclab_root):
    source_root = os.path.join(isaaclab_root, "source")
    for path in (
        os.path.join(source_root, "isaaclab"),
        os.path.join(source_root, "isaaclab_assets"),
        os.path.join(source_root, "isaaclab_tasks"),
        os.path.join(source_root, "isaaclab_rl"),
        os.path.join(isaaclab_root, "scripts", "reinforcement_learning", "rsl_rl"),
    ):
        _prepend_path(path)


def _join_url_or_path(root_path, relative_path):
    return root_path.rstrip("/") + "/" + relative_path.lstrip("/")


def configure_isaac_asset_root(args):
    import carb

    settings = carb.settings.get_settings()
    asset_root = (
        args.go2_physics_asset_root
        or os.environ.get("ISAACLAB_ASSETS_ROOT")
        or os.environ.get("ISAAC_ASSETS_ROOT")
        or settings.get("/persistent/isaac/asset_root/cloud")
        or DEFAULT_ISAAC_45_ASSET_ROOT
    )
    asset_root = asset_root.rstrip("/")

    for setting_name in (
        "/persistent/isaac/asset_root/default",
        "/persistent/isaac/asset_root/cloud",
        "/persistent/isaac/asset_root/nvidia",
    ):
        settings.set(setting_name, asset_root)

    # If isaaclab.utils.assets was imported before the setting was patched, keep
    # its module constants consistent with the runtime carb settings.
    assets_module = sys.modules.get("isaaclab.utils.assets")
    if assets_module is not None:
        assets_module.NUCLEUS_ASSET_ROOT_DIR = asset_root
        assets_module.NVIDIA_NUCLEUS_DIR = _join_url_or_path(asset_root, "NVIDIA")
        assets_module.ISAAC_NUCLEUS_DIR = _join_url_or_path(asset_root, "Isaac")
        assets_module.ISAACLAB_NUCLEUS_DIR = _join_url_or_path(asset_root, "Isaac/IsaacLab")

    print(f"[INFO] IsaacLab asset root: {asset_root}")
    return asset_root


def go2_usd_path_from_asset_root(asset_root):
    return _join_url_or_path(asset_root, "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd")


def make_rsl_rl_args(args):
    return SimpleNamespace(
        seed=None,
        resume=None,
        load_run=None,
        checkpoint=None,
        experiment_name=None,  ###
        save_interval=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )


def scene_usd_from_episode(episode, repo_root):
    scene_id = episode["scan"]
    for suffix in (".usda", ".usd"):
        scene_path = os.path.join(repo_root, scene_id, f"{scene_id}{suffix}")
        if os.path.exists(scene_path):
            return scene_path
    raise FileNotFoundError(f"Scene USD not found for '{scene_id}' under {repo_root}")


def parse_scope_list(scope_text):
    return {scope.strip() for scope in str(scope_text).split(",") if scope.strip()}


def use_kujiale_locomotion_adapter(args):
    adapter_name = getattr(args, "go2_physics_locomotion_adapter", "none")
    return args.go2_physics_scene == "kujiale" and adapter_name in {"floor_proxy", "walkable_proxy"}


def transform_points(points, transform_matrix):
    transform = np.array(transform_matrix, dtype=np.float64).T
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 3)
    points = np.matmul(points, transform[:3, :3].T)
    points += transform[:3, 3]
    return points


def triangulate_mesh_faces(points, face_counts, face_indices):
    triangles = []
    offset = 0
    face_counts = np.asarray(face_counts, dtype=np.int64)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    for count in face_counts:
        count = int(count)
        if count < 3:
            offset += count
            continue
        face = face_indices[offset : offset + count]
        offset += count
        for index in range(1, count - 1):
            triangles.append(points[[face[0], face[index], face[index + 1]]])
    if not triangles:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.asarray(triangles, dtype=np.float64)


def triangle_normal_z_abs(triangles):
    if triangles.size == 0:
        return np.zeros(0, dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    normal_z = np.zeros_like(norm)
    valid = norm > 1e-8
    normal_z[valid] = np.abs(normals[valid, 2]) / norm[valid]
    return normal_z


def collect_kujiale_walkable_triangles(source_stage, source_scopes, args):
    from pxr import UsdGeom

    xform_cache = UsdGeom.XformCache()
    triangles_by_scope = {}
    mesh_counts_by_scope = {}

    for prim in source_stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = str(prim.GetPath())
        scope = scope_from_kujiale_path(prim_path)
        if scope not in source_scopes:
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr().Get()
        counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if not points_attr or not counts_attr or not indices_attr:
            continue

        points = transform_points(points_attr, xform_cache.GetLocalToWorldTransform(prim))
        if points.shape[0] == 0:
            continue
        triangles = triangulate_mesh_faces(points, counts_attr, indices_attr)
        if triangles.size == 0:
            continue
        triangles_by_scope.setdefault(scope, []).append(triangles)
        mesh_counts_by_scope[scope] = mesh_counts_by_scope.get(scope, 0) + 1

    if not triangles_by_scope:
        return np.zeros((0, 3, 3), dtype=np.float64), None, mesh_counts_by_scope, {}

    floor_triangles = np.concatenate(triangles_by_scope.get("floor", []), axis=0) if "floor" in triangles_by_scope else None
    base_floor_z = None
    min_normal_z = float(args.go2_physics_height_proxy_min_normal_z)
    if floor_triangles is not None and floor_triangles.size > 0:
        floor_normal_z = triangle_normal_z_abs(floor_triangles)
        floor_means = floor_triangles[:, :, 2].mean(axis=1)
        floor_mask = floor_normal_z >= min_normal_z
        if np.any(floor_mask):
            base_floor_z = float(np.median(floor_means[floor_mask]))
    if base_floor_z is None:
        all_triangles = np.concatenate([tris for group in triangles_by_scope.values() for tris in group], axis=0)
        all_normal_z = triangle_normal_z_abs(all_triangles)
        all_means = all_triangles[:, :, 2].mean(axis=1)
        all_mask = all_normal_z >= min_normal_z
        if np.any(all_mask):
            base_floor_z = float(np.median(all_means[all_mask]))
    if base_floor_z is None:
        return np.zeros((0, 3, 3), dtype=np.float64), None, mesh_counts_by_scope, {}

    min_z = base_floor_z - float(args.go2_physics_height_proxy_max_drop_height)
    max_z = base_floor_z + float(args.go2_physics_height_proxy_max_step_height)
    selected = []
    selected_counts_by_scope = {}
    for scope, triangle_groups in triangles_by_scope.items():
        triangles = np.concatenate(triangle_groups, axis=0)
        normal_z = triangle_normal_z_abs(triangles)
        z_mean = triangles[:, :, 2].mean(axis=1)
        z_min = triangles[:, :, 2].min(axis=1)
        z_max = triangles[:, :, 2].max(axis=1)
        mask = (normal_z >= min_normal_z) & (z_mean >= min_z) & (z_mean <= max_z)
        # Keep gently sloped small steps, but reject tall vertical-ish object slabs.
        mask &= (z_max - z_min) <= max(float(args.go2_physics_height_proxy_max_step_height), 0.05)
        if np.any(mask):
            selected.append(triangles[mask])
            selected_counts_by_scope[scope] = selected_counts_by_scope.get(scope, 0) + int(np.count_nonzero(mask))

    if not selected:
        return np.zeros((0, 3, 3), dtype=np.float64), base_floor_z, mesh_counts_by_scope, selected_counts_by_scope
    return np.concatenate(selected, axis=0), base_floor_z, mesh_counts_by_scope, selected_counts_by_scope


def rasterize_walkable_height_proxy(triangles, args):
    if triangles.size == 0:
        return None, None, None

    resolution = max(float(args.go2_physics_height_proxy_grid_resolution), 0.03)
    padding = max(float(getattr(args, "go2_physics_height_proxy_gap_fill", 0.0)), resolution)
    xy = triangles[:, :, :2].reshape(-1, 2)
    min_xy = xy.min(axis=0) - padding
    max_xy = xy.max(axis=0) + padding
    xs = np.arange(min_xy[0], max_xy[0] + resolution * 0.5, resolution, dtype=np.float64)
    ys = np.arange(min_xy[1], max_xy[1] + resolution * 0.5, resolution, dtype=np.float64)
    height = np.full((len(xs), len(ys)), np.nan, dtype=np.float64)

    for triangle in triangles:
        tri_xy = triangle[:, :2]
        tri_z = triangle[:, 2]
        area = (
            (tri_xy[1, 1] - tri_xy[2, 1]) * (tri_xy[0, 0] - tri_xy[2, 0])
            + (tri_xy[2, 0] - tri_xy[1, 0]) * (tri_xy[0, 1] - tri_xy[2, 1])
        )
        if abs(area) < 1e-10:
            continue
        ix0 = max(int(math.floor((tri_xy[:, 0].min() - min_xy[0]) / resolution)) - 1, 0)
        ix1 = min(int(math.ceil((tri_xy[:, 0].max() - min_xy[0]) / resolution)) + 1, len(xs) - 1)
        iy0 = max(int(math.floor((tri_xy[:, 1].min() - min_xy[1]) / resolution)) - 1, 0)
        iy1 = min(int(math.ceil((tri_xy[:, 1].max() - min_xy[1]) / resolution)) + 1, len(ys) - 1)
        if ix1 < ix0 or iy1 < iy0:
            continue
        gx, gy = np.meshgrid(xs[ix0 : ix1 + 1], ys[iy0 : iy1 + 1], indexing="ij")
        w0 = ((tri_xy[1, 1] - tri_xy[2, 1]) * (gx - tri_xy[2, 0]) + (tri_xy[2, 0] - tri_xy[1, 0]) * (gy - tri_xy[2, 1])) / area
        w1 = ((tri_xy[2, 1] - tri_xy[0, 1]) * (gx - tri_xy[2, 0]) + (tri_xy[0, 0] - tri_xy[2, 0]) * (gy - tri_xy[2, 1])) / area
        w2 = 1.0 - w0 - w1
        mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(mask):
            continue
        values = w0 * tri_z[0] + w1 * tri_z[1] + w2 * tri_z[2]
        target = height[ix0 : ix1 + 1, iy0 : iy1 + 1]
        finite = np.isfinite(target)
        target[mask & ~finite] = values[mask & ~finite]
        target[mask & finite] = np.maximum(target[mask & finite], values[mask & finite])

    height = fill_height_proxy_gaps(height, resolution, args)
    height = smooth_height_proxy(height, args)
    return xs, ys, height


def fill_height_proxy_gaps(height, resolution, args):
    max_gap = max(float(args.go2_physics_height_proxy_gap_fill), 0.0)
    steps = int(math.ceil(max_gap / max(resolution, 1e-6)))
    if steps <= 0:
        return height
    filled = height.copy()
    for _ in range(steps):
        missing = ~np.isfinite(filled)
        if not np.any(missing):
            break
        accum = np.zeros_like(filled)
        counts = np.zeros_like(filled)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.full_like(filled, np.nan)
                src_x = slice(max(-dx, 0), filled.shape[0] - max(dx, 0))
                dst_x = slice(max(dx, 0), filled.shape[0] - max(-dx, 0))
                src_y = slice(max(-dy, 0), filled.shape[1] - max(dy, 0))
                dst_y = slice(max(dy, 0), filled.shape[1] - max(-dy, 0))
                shifted[dst_x, dst_y] = filled[src_x, src_y]
                valid = np.isfinite(shifted)
                accum[valid] += shifted[valid]
                counts[valid] += 1.0
        can_fill = missing & (counts > 0)
        filled[can_fill] = accum[can_fill] / counts[can_fill]
    return filled


def smooth_height_proxy(height, args):
    passes = max(int(args.go2_physics_height_proxy_smooth_passes), 0)
    smoothed = height.copy()
    for _ in range(passes):
        accum = np.zeros_like(smoothed)
        counts = np.zeros_like(smoothed)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                shifted = np.full_like(smoothed, np.nan)
                src_x = slice(max(-dx, 0), smoothed.shape[0] - max(dx, 0))
                dst_x = slice(max(dx, 0), smoothed.shape[0] - max(-dx, 0))
                src_y = slice(max(-dy, 0), smoothed.shape[1] - max(dy, 0))
                dst_y = slice(max(dy, 0), smoothed.shape[1] - max(-dy, 0))
                shifted[dst_x, dst_y] = smoothed[src_x, src_y]
                valid = np.isfinite(shifted)
                accum[valid] += shifted[valid]
                counts[valid] += 1.0
        valid_center = np.isfinite(smoothed) & (counts > 0)
        smoothed[valid_center] = accum[valid_center] / counts[valid_center]
    return smoothed


def height_grid_to_mesh(xs, ys, height):
    index_by_node = {}
    points = []
    for ix in range(len(xs)):
        for iy in range(len(ys)):
            if np.isfinite(height[ix, iy]):
                index_by_node[(ix, iy)] = len(points)
                points.append([float(xs[ix]), float(ys[iy]), float(height[ix, iy])])

    counts = []
    indices = []
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            corners = ((ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1))
            if not all(corner in index_by_node for corner in corners):
                continue
            i0, i1, i2, i3 = (index_by_node[corner] for corner in corners)
            counts.extend([3, 3])
            indices.extend([i0, i1, i2, i0, i2, i3])
    return np.asarray(points, dtype=np.float32), counts, indices


def create_kujiale_height_proxy_usd(scene_usd, args, episode):
    try:
        from pxr import Sdf, Usd, UsdGeom
    except Exception as exc:
        print(f"[WARN] Could not import USD APIs for Kujiale locomotion adapter: {exc}")
        return None

    source_scopes = parse_scope_list(args.go2_physics_height_proxy_source_scopes)
    if not source_scopes:
        print("[WARN] Kujiale locomotion adapter has no source scopes; height proxy disabled.")
        return None

    source_stage = Usd.Stage.Open(scene_usd)
    if source_stage is None:
        print(f"[WARN] Could not open Kujiale USD for locomotion adapter: {scene_usd}")
        return None

    triangles, base_floor_z, mesh_counts_by_scope, selected_counts_by_scope = collect_kujiale_walkable_triangles(
        source_stage, source_scopes, args
    )
    if triangles.size == 0:
        print(
            "[WARN] Kujiale locomotion adapter could not find walkable proxy source meshes "
            f"for scopes={sorted(source_scopes)} in {scene_usd}."
        )
        return None

    proxy_dir = os.path.join(os.path.abspath(args.work_dir), "_go2_locomotion_adapter")
    os.makedirs(proxy_dir, exist_ok=True)
    scene_id = episode.get("scan", "scene")
    scope_token = "_".join(sorted(source_scopes))
    proxy_usd = os.path.join(proxy_dir, f"{scene_id}_{scope_token}_height_proxy.usda")
    if os.path.exists(proxy_usd):
        os.remove(proxy_usd)

    if args.go2_physics_height_proxy_mode == "walkable":
        xs, ys, height = rasterize_walkable_height_proxy(triangles, args)
        points, merged_counts, merged_indices = height_grid_to_mesh(xs, ys, height)
    else:
        points = triangles.reshape(-1, 3).astype(np.float32)
        merged_counts = [3] * len(triangles)
        merged_indices = list(range(points.shape[0]))

    if points.shape[0] == 0 or not merged_indices:
        print("[WARN] Kujiale locomotion adapter generated an empty height proxy.")
        return None

    proxy_stage = Usd.Stage.CreateNew(proxy_usd)
    root = UsdGeom.Xform.Define(proxy_stage, "/Root")
    proxy_stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute("iamgoodnavigator:sourceScene", Sdf.ValueTypeNames.String).Set(scene_usd)
    root.GetPrim().CreateAttribute("iamgoodnavigator:sourceScopes", Sdf.ValueTypeNames.String).Set(
        ",".join(sorted(source_scopes))
    )

    mesh = UsdGeom.Mesh.Define(proxy_stage, "/Root/Geometry")
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr(merged_counts)
    mesh.CreateFaceVertexIndicesAttr(merged_indices)
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Imageable(root.GetPrim()).MakeInvisible()
    proxy_stage.GetRootLayer().Save()

    print(
        "[INFO] Kujiale locomotion adapter: wrote height proxy "
        f"{proxy_usd} mode={args.go2_physics_height_proxy_mode}, "
        f"source_meshes={format_top_counts(mesh_counts_by_scope)}, "
        f"selected_triangles={format_top_counts(selected_counts_by_scope)}, "
        f"walkable_triangles={len(triangles)}, base_floor_z={base_floor_z:+.3f}, "
        f"points={points.shape[0]}, face_indices={len(merged_indices)}, scopes={sorted(source_scopes)}"
    )
    return proxy_usd


def insert_scene_cfg_entity_before(scene_cfg, entity_name, entity_cfg, before_name):
    items = list(scene_cfg.__dict__.items())
    scene_cfg.__dict__.clear()
    inserted = False
    for key, value in items:
        if key == entity_name:
            continue
        if key == before_name and not inserted:
            scene_cfg.__dict__[entity_name] = entity_cfg
            inserted = True
        scene_cfg.__dict__[key] = value
    if not inserted:
        scene_cfg.__dict__[entity_name] = entity_cfg


def add_kujiale_locomotion_adapter_to_scene(args, env_cfg, scene_usd, episode):
    proxy_usd = create_kujiale_height_proxy_usd(scene_usd, args, episode)
    if proxy_usd is None:
        return None
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
    except Exception as exc:
        print(f"[WARN] Could not import IsaacLab asset config for Kujiale locomotion adapter: {exc}")
        return None

    proxy_prim_path = args.go2_physics_height_proxy_prim.rstrip("/")
    proxy_cfg = AssetBaseCfg(
        prim_path=proxy_prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=proxy_usd),
    )
    insert_scene_cfg_entity_before(env_cfg.scene, KUJIALE_HEIGHT_PROXY_ENTITY_NAME, proxy_cfg, "height_scanner")
    print(f"[INFO] Kujiale locomotion adapter: height scanner target prim={proxy_prim_path}")
    return proxy_prim_path


def yaw_from_quaternion_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[0] != 4 or np.linalg.norm(quat) < 1e-6:
        return 0.0
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw_wxyz(yaw):
    half_yaw = 0.5 * float(yaw)
    return (
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )


def heading_from_quaternion_wxyz(quat):
    yaw = yaw_from_quaternion_wxyz(quat)
    return np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)


def configure_env_from_episode(args, env_cfg, episode, repo_root):
    if hasattr(env_cfg, "ui_window_class_type"):
        env_cfg.ui_window_class_type = None

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.episode_length_s = 60.0 * 60.0

    if getattr(env_cfg, "sim", None) is not None:
        env_cfg.sim.use_fabric = bool(getattr(args, "go2_physics_use_fabric", True))
        render_interval = args.go2_physics_render_interval
        if render_interval is None:
            render_interval = getattr(env_cfg, "decimation", 4)
        env_cfg.sim.render_interval = max(int(render_interval), 1)
        render_hz = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.sim.render_interval))
        control_hz = 1.0 / (float(env_cfg.sim.dt) * float(getattr(env_cfg, "decimation", 1)))
        physics_hz = 1.0 / float(env_cfg.sim.dt)
        args._go2_physics_control_dt = float(env_cfg.sim.dt) * float(getattr(env_cfg, "decimation", 1))  # ###2
        print(
            f"[INFO] Go2 timing: physics={physics_hz:.1f} Hz, "
            f"control={control_hz:.1f} Hz, render={render_hz:.1f} Hz "
            f"(render_interval={env_cfg.sim.render_interval}, use_fabric={env_cfg.sim.use_fabric})"
        )

    scene_usd = None
    height_scanner_target = "/World/ground"
    if args.go2_physics_scene == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.usd_path = None
        print("[INFO] Physics scene: IsaacLab plane")
    else:
        scene_usd = scene_usd_from_episode(episode, repo_root)
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.usd_path = scene_usd
        env_cfg.scene.terrain.prim_path = "/World/ground"
        print(f"[INFO] Physics scene USD: {scene_usd}")
        if use_kujiale_locomotion_adapter(args):
            proxy_target = add_kujiale_locomotion_adapter_to_scene(args, env_cfg, scene_usd, episode)
            if proxy_target is not None:
                height_scanner_target = proxy_target

    if hasattr(env_cfg.scene.terrain, "use_terrain_origins"):
        env_cfg.scene.terrain.use_terrain_origins = False
    if hasattr(env_cfg.scene.terrain, "env_spacing"):
        env_cfg.scene.terrain.env_spacing = 2.5
    if getattr(env_cfg.scene.terrain, "terrain_generator", None) is None:
        curriculum = getattr(env_cfg, "curriculum", None)
        if curriculum is not None and hasattr(curriculum, "terrain_levels"):
            curriculum.terrain_levels = None

    start_pos = list(episode["start_position"])
    root_pos = (start_pos[0], start_pos[1], start_pos[2] + args.go2_base_height)
    env_cfg.scene.robot.init_state.pos = root_pos
    start_yaw = yaw_from_quaternion_wxyz(episode["start_rotation"])
    env_cfg.scene.robot.init_state.rot = quaternion_from_yaw_wxyz(start_yaw)

    if getattr(env_cfg.scene, "height_scanner", None) is not None:
        env_cfg.scene.height_scanner.mesh_prim_paths = [height_scanner_target]
        if height_scanner_target != "/World/ground":
            print(f"[INFO] Go2 height scanner target: {height_scanner_target}")

    if args.go2_usd_path:
        env_cfg.scene.robot.spawn.usd_path = args.go2_usd_path
        print(f"[INFO] Go2 USD override: {args.go2_usd_path}")
    elif "None/" in str(env_cfg.scene.robot.spawn.usd_path):
        asset_root = args._resolved_go2_physics_asset_root
        env_cfg.scene.robot.spawn.usd_path = go2_usd_path_from_asset_root(asset_root)
        print(f"[INFO] Repaired Go2 USD path: {env_cfg.scene.robot.spawn.usd_path}")
    else:
        print(f"[INFO] Go2 USD path: {env_cfg.scene.robot.spawn.usd_path}")

    command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if command_cfg is not None:
        command_cfg.rel_standing_envs = 0.0
        command_cfg.rel_heading_envs = 0.0
        command_cfg.heading_command = False
        command_cfg.debug_vis = False
        command_cfg.ranges.lin_vel_x = (-max(args.go2_physics_lin_vel, args.go2_physics_backward_lin_vel), args.go2_physics_lin_vel)
        command_cfg.ranges.lin_vel_y = (-args.go2_physics_strafe_vel, args.go2_physics_strafe_vel)
        command_cfg.ranges.ang_vel_z = (-args.go2_physics_ang_vel, args.go2_physics_ang_vel)

    events = getattr(env_cfg, "events", None)
    if events is not None:
        if hasattr(events, "push_robot"):
            events.push_robot = None
        if hasattr(events, "base_external_force_torque"):
            events.base_external_force_torque = None
        if hasattr(events, "add_base_mass"):
            events.add_base_mass = None
        if hasattr(events, "base_com"):
            events.base_com = None
        reset_base = getattr(events, "reset_base", None)
        if reset_base is not None and hasattr(reset_base, "params"):
            reset_base.params = {
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }
        reset_joints = getattr(events, "reset_robot_joints", None)
        if reset_joints is not None and hasattr(reset_joints, "params"):
            reset_joints.params["position_range"] = (1.0, 1.0)
            reset_joints.params["velocity_range"] = (0.0, 0.0)

    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None and args.go2_physics_disable_base_contact_termination:
        if hasattr(terminations, "base_contact"):
            terminations.base_contact = None
            print("[INFO] Go2 base_contact termination disabled.")

    print(f"[INFO] Go2 root start: {env_cfg.scene.robot.init_state.pos}")
    print(f"[INFO] Go2 start yaw-only rotation wxyz: {env_cfg.scene.robot.init_state.rot}")
    print(f"[INFO] Episode start yaw deg: {math.degrees(start_yaw):+.2f}")
    return env_cfg


class X11KeyboardPoller:
    _KEYSYM_NAMES = {
        "W": "w",
        "S": "s",
        "A": "a",
        "D": "d",
        "Q": "q",
        "E": "e",
        "R": "r",
        "SPACE": "space",
        "ENTER": "Return",
        "NUMPAD_ENTER": "KP_Enter",
        "ESCAPE": "Escape",
    }

    def __init__(self, key_names):
        lib_path = ctypes.util.find_library("X11")
        if lib_path is None:
            raise RuntimeError("libX11 was not found")

        self._x11 = ctypes.cdll.LoadLibrary(lib_path)
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._x11.XCloseDisplay.restype = ctypes.c_int
        self._x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
        self._x11.XStringToKeysym.restype = ctypes.c_ulong
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self._x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char * 32)]
        self._x11.XQueryKeymap.restype = ctypes.c_int

        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise RuntimeError("Could not open X11 display")

        self._keycodes = {}
        for key_name in key_names:
            keysym_name = self._KEYSYM_NAMES.get(key_name)
            if keysym_name is None:
                continue
            keysym = self._x11.XStringToKeysym(keysym_name.encode("ascii"))
            if keysym == 0:
                continue
            keycode = self._x11.XKeysymToKeycode(self._display, keysym)
            if keycode != 0:
                self._keycodes[key_name] = int(keycode)

    def close(self):
        if getattr(self, "_display", None):
            self._x11.XCloseDisplay(self._display)
            self._display = None

    def pressed_keys(self):
        if not getattr(self, "_display", None):
            return set()
        keymap = (ctypes.c_char * 32)()
        if self._x11.XQueryKeymap(self._display, ctypes.byref(keymap)) == 0:
            return set()
        pressed = set()
        for key_name, keycode in self._keycodes.items():
            byte_index = keycode // 8
            bit_index = keycode % 8
            raw_byte = keymap[byte_index]
            byte_value = raw_byte[0] if isinstance(raw_byte, bytes) else int(raw_byte)
            if byte_value & (1 << bit_index):
                pressed.add(key_name)
        return pressed


class TerminalKeyboardPoller:
    _CHAR_TO_KEY = {
        "w": "W",
        "s": "S",
        "a": "A",
        "d": "D",
        "q": "Q",
        "e": "E",
        "r": "R",
        " ": "SPACE",
        "\r": "ENTER",
        "\n": "ENTER",
        "\x1b": "ESCAPE",
    }

    def __init__(self, key_names, hold_s):
        if not sys.stdin or not sys.stdin.isatty():
            raise RuntimeError("stdin is not an interactive TTY")

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        self._key_names = set(key_names)
        self._hold_s = max(float(hold_s), 0.05)
        self._last_press_by_key = {}
        self._closed = False
        tty.setcbreak(self._fd)
        atexit.register(self.close)

    def close(self):
        if self._closed:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass
        self._closed = True

    def pressed_keys(self):
        now = time.monotonic()
        try:
            while select.select([sys.stdin], [], [], 0.0)[0]:
                raw = os.read(self._fd, 1)
                if not raw:
                    break
                key_name = self._CHAR_TO_KEY.get(raw.decode("utf-8", errors="ignore").lower())
                if key_name in self._key_names:
                    self._last_press_by_key[key_name] = now
        except Exception:
            return set()

        expired = [
            key_name for key_name, press_time in self._last_press_by_key.items() if now - press_time > self._hold_s
        ]
        for key_name in expired:
            self._last_press_by_key.pop(key_name, None)
        return set(self._last_press_by_key)


class WASDVelocityKeyboard:
    _KEY_ALIASES = {
        "ESC": "ESCAPE",
        "RETURN": "ENTER",
        "KP_ENTER": "NUMPAD_ENTER",
        "KPENTER": "NUMPAD_ENTER",
        "NUMPADENTER": "NUMPAD_ENTER",
        " ": "SPACE",
    }

    def __init__(self, args):
        import carb
        import omni.appwindow

        self._args = args
        self._carb = carb
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._active_keys_by_source = {"isaac": set(), "carb": set(), "x11": set(), "terminal": set()}
        self._pending_release_by_source = {"isaac": {}}
        self.reset_requested = False
        self.quit_requested = False
        self.mission_complete = False
        # TODO(student): Fill velocity commands for keyboard teleoperation.
        # Command format is [vx, vy, wz]:
        #   vx: forward/backward linear velocity in m/s
        #   vy: left/right lateral velocity in m/s
        #   wz: yaw angular velocity in rad/s
        self._mapping = {
            "W": np.asarray([args.go2_physics_lin_vel, 0.0, 0.0], dtype=np.float32),
            "S": np.asarray([-args.go2_physics_lin_vel, 0.0, 0.0], dtype=np.float32),
            "A": np.asarray([0.0, 0.0, args.go2_physics_ang_vel], dtype=np.float32),
            "D": np.asarray([0.0, 0.0, -args.go2_physics_ang_vel], dtype=np.float32),
            "Q": np.asarray([0.0, -args.go2_physics_strafe_vel, 0.0], dtype=np.float32),
            "E": np.asarray([0.0, args.go2_physics_strafe_vel, 0.0], dtype=np.float32),
        }
        self._poll_key_names = set(self._mapping) | {"SPACE", "R", "ENTER", "NUMPAD_ENTER", "ESCAPE"}
        self._carb_key_inputs = self._create_carb_key_inputs()
        self._keyboard_poller = None
        self._terminal_poller = None
        self._last_debug_command = None
        self._last_debug_active_keys = set()
        self._advance_count = 0

        if getattr(args, "go2_physics_poll_keyboard", True):
            try:
                self._keyboard_poller = X11KeyboardPoller(self._poll_key_names)
                print("[INFO] Go2 physics keyboard polling enabled (Carb + X11).")
            except Exception as exc:
                print(f"[WARN] X11 keyboard polling is unavailable: {exc}. Falling back to Isaac window events.")
        if getattr(args, "go2_physics_terminal_keyboard", True):
            try:
                self._terminal_poller = TerminalKeyboardPoller(
                    self._poll_key_names, getattr(args, "go2_physics_terminal_key_hold_s", 0.35)
                )
                print("[INFO] Terminal WASD fallback enabled. If viewport focus misses keys, focus this terminal.")
            except Exception as exc:
                if getattr(args, "go2_physics_debug_keyboard", False):
                    print(f"[KEY] Terminal keyboard fallback unavailable: {exc}")

        print("[INFO] Keyboard ready: W/S/A/D drive, Q/E strafe, SPACE stop, R reset, ENTER finish, ESC quit.")

    def close(self):
        if getattr(self, "_sub_keyboard", None) is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            except Exception:
                pass
            self._sub_keyboard = None
        if getattr(self, "_keyboard_poller", None) is not None:
            self._keyboard_poller.close()
            self._keyboard_poller = None
        if getattr(self, "_terminal_poller", None) is not None:
            self._terminal_poller.close()
            self._terminal_poller = None

    def reset(self):
        for active_keys in self._active_keys_by_source.values():
            active_keys.clear()
        for pending_release in self._pending_release_by_source.values():
            pending_release.clear()
        self._last_debug_command = None
        self._last_debug_active_keys.clear()

    def advance(self):
        self.poll_events()

        command = np.zeros(3, dtype=np.float32)
        active_motion_keys = self._active_motion_keys()
        for key_name in sorted(active_motion_keys):
            command += self._mapping.get(key_name, 0.0)
        self._print_debug_velocity(command, active_motion_keys)
        return command

    def poll_events(self):
        self._advance_count += 1
        if getattr(self._args, "go2_physics_poll_keyboard", True) or self._terminal_poller is not None:
            self._poll_keyboard_state()
        self._flush_pending_releases()

    def _create_carb_key_inputs(self):
        key_inputs = {}
        keyboard_input_type = self._carb.input.KeyboardInput
        for key_name in self._poll_key_names:
            keyboard_input = getattr(keyboard_input_type, self._normalize_key_name(key_name), None)
            if keyboard_input is not None:
                key_inputs[key_name] = keyboard_input
        return key_inputs

    def _normalize_key_name(self, key_name):
        if hasattr(key_name, "name"):
            key_name = key_name.name
        key_name = str(key_name).split(".")[-1].upper()
        return self._KEY_ALIASES.get(key_name, key_name)

    def _active_motion_keys(self):
        active_keys = set()
        for source_keys in self._active_keys_by_source.values():
            active_keys.update(key for key in source_keys if key in self._mapping)
        return active_keys

    def _set_source_keys(self, source, current_keys):
        current_keys = {self._normalize_key_name(key_name) for key_name in current_keys}
        previous_keys = set(self._active_keys_by_source.get(source, set()))
        for key_name in sorted(current_keys - previous_keys):
            self._handle_press_name(key_name, source=source)
        for key_name in sorted(previous_keys - current_keys):
            self._handle_release_name(key_name, source=source, immediate=True)

    def _poll_carb_keyboard_state(self):
        current_keys = set()
        if not self._carb_key_inputs:
            self._set_source_keys("carb", current_keys)
            return
        try:
            for key_name, keyboard_input in self._carb_key_inputs.items():
                if self._input.get_keyboard_value(self._keyboard, keyboard_input) > 0.5:
                    current_keys.add(key_name)
        except Exception as exc:
            if getattr(self._args, "go2_physics_debug_keyboard", False):
                print(f"[KEY] Carb keyboard polling failed: {exc}")
            current_keys = set()
        self._set_source_keys("carb", current_keys)

    def _poll_keyboard_state(self):
        if getattr(self._args, "go2_physics_poll_keyboard", True):
            self._poll_carb_keyboard_state()
        if getattr(self._args, "go2_physics_poll_keyboard", True) and self._keyboard_poller is not None:
            self._set_source_keys("x11", self._keyboard_poller.pressed_keys())
        if self._terminal_poller is not None:
            self._set_source_keys("terminal", self._terminal_poller.pressed_keys())

    def _handle_press_name(self, key_name, source):
        key_name = self._normalize_key_name(key_name)
        if source not in self._active_keys_by_source:
            self._active_keys_by_source[source] = set()
        was_active_for_source = key_name in self._active_keys_by_source[source]
        self._pending_release_by_source.setdefault(source, {}).pop(key_name, None)
        self._active_keys_by_source[source].add(key_name)

        if was_active_for_source:
            return
        if key_name == "SPACE":
            self.reset()
        elif key_name == "R":
            self.reset_requested = True
        elif key_name in ("ENTER", "NUMPAD_ENTER"):
            print("ENTER pressed - Mission Complete.")
            self.mission_complete = True
        elif key_name == "ESCAPE":
            self.quit_requested = True

    def _handle_release_name(self, key_name, source, immediate=False):
        key_name = self._normalize_key_name(key_name)
        if source not in self._active_keys_by_source or key_name not in self._active_keys_by_source[source]:
            return
        if (
            not immediate
            and key_name in self._mapping
            and getattr(self._args, "go2_physics_key_release_grace_s", 0.0) > 0.0
        ):
            self._pending_release_by_source.setdefault(source, {})[key_name] = time.monotonic()
            return
        self._active_keys_by_source[source].discard(key_name)

    def _flush_pending_releases(self):
        grace_s = max(getattr(self._args, "go2_physics_key_release_grace_s", 0.0), 0.0)
        if grace_s <= 0.0:
            return
        now = time.monotonic()
        for source, pending_releases in self._pending_release_by_source.items():
            expired_keys = [
                key_name for key_name, release_time in pending_releases.items() if now - release_time >= grace_s
            ]
            for key_name in expired_keys:
                pending_releases.pop(key_name, None)
                self._active_keys_by_source.get(source, set()).discard(key_name)

    def _print_debug_velocity(self, command, active_keys):
        if not getattr(self._args, "go2_physics_debug_keyboard", False):
            return
        active_keys = set(active_keys)
        should_print_held = (
            getattr(self._args, "go2_physics_debug_keyboard_poll_every", 0) > 0
            and self._advance_count % self._args.go2_physics_debug_keyboard_poll_every == 0
            and bool(active_keys)
        )
        command_changed = self._last_debug_command is None or not np.allclose(command, self._last_debug_command)
        keys_changed = active_keys != self._last_debug_active_keys
        if not (command_changed or keys_changed or should_print_held):
            return
        print(
            "[KEY] command "
            f"vx={command[0]:+.2f}, vy={command[1]:+.2f}, wz={command[2]:+.2f}; "
            f"active={sorted(active_keys) if active_keys else 'none'}"
        )
        self._last_debug_command = command.copy()
        self._last_debug_active_keys = active_keys

    def _on_keyboard_event(self, event, *args, **kwargs):
        carb = self._carb
        event_input = getattr(event, "input", None)
        key_name = getattr(event_input, "name", event_input)
        if key_name is None:
            return True
        key_name = self._normalize_key_name(key_name)
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            self._handle_press_name(key_name, source="isaac")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._handle_release_name(key_name, source="isaac")
        return True


def policy_obs_tensor(obs):
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        try:
            return obs["policy"]
        except Exception:
            pass
    return obs


def get_command_slice(obs):
    tensor = policy_obs_tensor(obs)
    width = int(tensor.shape[1])
    if width >= 48:
        return slice(9, 12)
    return slice(6, 9)


def inject_velocity_command(obs, env, command):
    tensor = policy_obs_tensor(obs)
    command_slice = get_command_slice(obs)
    if tensor.shape[1] >= command_slice.stop:
        tensor[:, command_slice] = command

    command_manager = getattr(env.unwrapped, "command_manager", None)
    terms = getattr(command_manager, "_terms", {}) if command_manager else {}
    base_velocity_term = terms.get("base_velocity")
    if base_velocity_term is None or not hasattr(base_velocity_term, "vel_command_b"):
        return

    command_batch = command.view(1, 3).to(
        device=base_velocity_term.vel_command_b.device,
        dtype=base_velocity_term.vel_command_b.dtype,
    )
    base_velocity_term.vel_command_b[:, :] = command_batch
    if hasattr(base_velocity_term, "is_standing_env"):
        base_velocity_term.is_standing_env[:] = False
    if hasattr(base_velocity_term, "is_heading_env"):
        base_velocity_term.is_heading_env[:] = False


def get_scripted_command(args, torch, device, dtype):
    if args.go2_physics_scripted_command is None:
        return None
    mapping = {
        "forward": [args.go2_physics_lin_vel, 0.0, 0.0],
        "backward": [-args.go2_physics_backward_lin_vel, 0.0, 0.0],
        "turn_left": [0.0, 0.0, args.go2_physics_ang_vel],
        "turn_right": [0.0, 0.0, -args.go2_physics_ang_vel],
        "strafe_left": [0.0, -args.go2_physics_strafe_vel, 0.0],
        "strafe_right": [0.0, args.go2_physics_strafe_vel, 0.0],
    }
    return torch.tensor(mapping[args.go2_physics_scripted_command], device=device, dtype=dtype)


# ###2: Week 2 language-command parser and timed velocity executor.
LANGUAGE_COMMAND_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"  # ###2


def normalize_language_command_text(text):  # ###2
    text = str(text or "").strip().lower()  # ###2
    text = text.replace("°", " degrees")  # ###2
    text = re.sub(r"[_-]+", " ", text)  # ###2
    return re.sub(r"\s+", " ", text)  # ###2


def parse_language_velocity_command(raw_text, linear_speed_mps=0.5, yaw_speed_deg_s=30.0):  # ###2
    text = normalize_language_command_text(raw_text)  # ###2
    if not text:  # ###2
        raise ValueError("empty command")  # ###2

    if text in {"finish", "done"}:  # ###2
        return SimpleNamespace(kind="finish", raw=raw_text, label="finish", command=np.zeros(3, dtype=np.float32), duration_s=0.0)  # ###2
    if text in {"quit", "exit", "abort"}:  # ###2
        return SimpleNamespace(kind="quit", raw=raw_text, label="quit", command=np.zeros(3, dtype=np.float32), duration_s=0.0)  # ###2
    if text == "stop":  # ###2
        return SimpleNamespace(kind="stop", raw=raw_text, label="stop", command=np.zeros(3, dtype=np.float32), duration_s=0.0)  # ###2

    move_match = re.fullmatch(rf"move\s+forward\s+({LANGUAGE_COMMAND_NUMBER_RE})\s*(?:m|meter|meters)?", text)  # ###2
    if move_match:  # ###2
        distance_m = float(move_match.group(1))  # ###2
        if distance_m <= 0.0:  # ###2
            raise ValueError("move forward distance must be positive")  # ###2
        linear_speed_mps = max(float(linear_speed_mps), 1e-6)  # ###2
        duration_s = distance_m / linear_speed_mps  # ###2
        command = np.asarray([linear_speed_mps, 0.0, 0.0], dtype=np.float32)  # ###2
        return SimpleNamespace(kind="motion", raw=raw_text, label=f"move forward {distance_m:g}m", command=command, duration_s=duration_s)  # ###2

    turn_match = re.fullmatch(  # ###2
        rf"turn\s+(left|right)\s+({LANGUAGE_COMMAND_NUMBER_RE})\s*(?:degrees?|degree|degs?|deg)?", text  # ###2
    )  # ###2
    if turn_match:  # ###2
        direction = turn_match.group(1)  # ###2
        degrees = float(turn_match.group(2))  # ###2
        if degrees <= 0.0:  # ###2
            raise ValueError("turn angle must be positive")  # ###2
        yaw_speed_deg_s = max(float(yaw_speed_deg_s), 1e-6)  # ###2
        duration_s = degrees / yaw_speed_deg_s  # ###2
        yaw_speed_rad_s = math.radians(yaw_speed_deg_s)  # ###2
        signed_wz = yaw_speed_rad_s if direction == "left" else -yaw_speed_rad_s  # ###2
        command = np.asarray([0.0, 0.0, signed_wz], dtype=np.float32)  # ###2
        return SimpleNamespace(kind="motion", raw=raw_text, label=f"turn {direction} {degrees:g}deg", command=command, duration_s=duration_s)  # ###2

    raise ValueError(  # ###2
        'unsupported command. Use "move forward <meters>", "turn left <degrees>", "turn right <degrees>", or "stop"'  # ###2
    )  # ###2


class LanguageVelocityCommander:  # ###2
    def __init__(self, args, control_dt):  # ###2
        self.linear_speed_mps = max(float(getattr(args, "go2_physics_language_lin_vel", 0.5)), 1e-6)  # ###2
        self.yaw_speed_deg_s = max(float(getattr(args, "go2_physics_language_ang_vel_deg", 30.0)), 1e-6)  # ###2
        self.control_dt = max(float(control_dt or 0.02), 1e-6)  # ###2
        self._command = np.zeros(3, dtype=np.float32)  # ###2
        self._remaining_steps = 0  # ###2
        self._active_label = "stop"  # ###2
        self.mission_complete = False  # ###2
        self.quit_requested = False  # ###2
        print(  # ###2
            "[INFO] Interactive language prompt ready. Commands: "  # ###2
            '"move forward 0.75", "move forward 0.75m", "turn left 90", '  # ###2
            '"turn right 90degrees", "stop". Type "finish" to end successfully, or "quit" to abort.'  # ###2
        )  # ###2
        print(  # ###2
            "[INFO] Language velocity units: "  # ###2
            f"move={self.linear_speed_mps:.2f}m/s, turn={self.yaw_speed_deg_s:.1f}deg/s, "  # ###2
            f"control_dt={self.control_dt:.4f}s"  # ###2
        )  # ###2

    def reset(self):  # ###2
        self._command[:] = 0.0  # ###2
        self._remaining_steps = 0  # ###2
        self._active_label = "stop"  # ###2
        self.mission_complete = False  # ###2
        self.quit_requested = False  # ###2

    def advance(self):  # ###2
        if self._remaining_steps > 0:  # ###2
            return self._consume_active_command()  # ###2

        while True:  # ###2
            try:  # ###2
                raw_text = input("Command: ")  # ###2
            except (EOFError, KeyboardInterrupt):  # ###2
                print("\n[INFO] Language prompt closed; aborting Go2 physics run.")  # ###2
                self.quit_requested = True  # ###2
                return np.zeros(3, dtype=np.float32)  # ###2

            try:  # ###2
                parsed = parse_language_velocity_command(raw_text, self.linear_speed_mps, self.yaw_speed_deg_s)  # ###2
            except ValueError as exc:  # ###2
                print(f"[WARN] Rejected command: {exc}")  # ###2
                continue  # ###2

            if parsed.kind == "finish":  # ###2
                self.mission_complete = True  # ###2
                print("[INFO] Language command requested mission finish.")  # ###2
                return np.zeros(3, dtype=np.float32)  # ###2
            if parsed.kind == "quit":  # ###2
                self.quit_requested = True  # ###2
                print("[INFO] Language command requested quit.")  # ###2
                return np.zeros(3, dtype=np.float32)  # ###2
            if parsed.kind == "stop":  # ###2
                self._command[:] = 0.0  # ###2
                self._remaining_steps = 0  # ###2
                self._active_label = parsed.label  # ###2
                print("[INFO] Accepted command: stop (stop)")  # ###2
                return self._command.copy()  # ###2

            self._command = parsed.command.astype(np.float32, copy=True)  # ###2
            self._remaining_steps = max(int(math.ceil(parsed.duration_s / self.control_dt)), 1)  # ###2
            self._active_label = parsed.label  # ###2
            print(f"[INFO] Accepted command: {parsed.label} ({parsed.duration_s:.2f}s)")  # ###2
            return self._consume_active_command()  # ###2

    def _consume_active_command(self):  # ###2
        command = self._command.copy()  # ###2
        self._remaining_steps -= 1  # ###2
        if self._remaining_steps == 0:  # ###2
            print("[INFO] Command duration complete; velocity command reset to stop.")  # ###2
            self._command[:] = 0.0  # ###2
            self._active_label = "stop"  # ###2
        return command  # ###2

    def status_text(self):  # ###2
        if self._remaining_steps <= 0:  # ###2
            return ""  # ###2
        return f" language='{self._active_label}' remaining={self._remaining_steps * self.control_dt:.2f}s"  # ###2


class LearnedLanguageVelocityCommander:
    def __init__(self, args, control_dt):
        checkpoint = getattr(args, "go2_physics_language_model_checkpoint", None)
        if not checkpoint:
            default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "language_velocity_mlp.pt")
            if os.path.exists(default):
                checkpoint = default
                print(f"[INFO] No checkpoint specified; using default: {checkpoint}")
            else:
                raise ValueError("--go2-physics-language-model-checkpoint is required for learned language commands")

        from language_velocity_model import LanguageVelocityPredictor, format_velocity

        self.control_dt = max(float(control_dt or 0.02), 1e-6)
        self.command_duration_s = max(float(getattr(args, "go2_physics_language_command_duration", 1.0)), self.control_dt)
        self._format_velocity = format_velocity
        self._command = np.zeros(3, dtype=np.float32)
        self._remaining_steps = 0
        self._active_label = "stop"
        self.mission_complete = False
        self.quit_requested = False

        device = getattr(args, "go2_physics_language_model_device", "auto")
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        print(f"[INFO] Loading learned language velocity checkpoint: {checkpoint}")
        self.predictor = LanguageVelocityPredictor(checkpoint, device=device)
        print(
            "[INFO] Learned language prompt ready. Commands include "
            '"move forward 25cm", "move forward 50cm", "turn left 30 degree", "stop". '
            'Type "finish" to end successfully, or "quit" to abort.'
        )
        print(
            "[INFO] Learned command execution: "
            f"duration={self.command_duration_s:.2f}s, control_dt={self.control_dt:.4f}s"
        )

    def reset(self):
        self._command[:] = 0.0
        self._remaining_steps = 0
        self._active_label = "stop"
        self.mission_complete = False
        self.quit_requested = False

    def advance(self):
        if self._remaining_steps > 0:
            return self._consume_active_command()

        while True:
            try:
                raw_text = input("Command: ")
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Learned language prompt closed; aborting Go2 physics run.")
                self.quit_requested = True
                return np.zeros(3, dtype=np.float32)

            text = normalize_language_command_text(raw_text)
            if not text:
                continue
            if text in {"finish", "done"}:
                self.mission_complete = True
                print("[INFO] Learned language command requested mission finish.")
                return np.zeros(3, dtype=np.float32)
            if text in {"quit", "exit", "abort"}:
                self.quit_requested = True
                print("[INFO] Learned language command requested quit.")
                return np.zeros(3, dtype=np.float32)

            try:
                predicted_velocity = self.predictor.predict(raw_text)
            except Exception as exc:
                print(f"[WARN] Learned language inference failed: {exc}")
                continue

            self._command = np.asarray(predicted_velocity, dtype=np.float32)
            self._remaining_steps = max(int(math.ceil(self.command_duration_s / self.control_dt)), 1)
            self._active_label = text
            print(
                f"[INFO] Text input: {raw_text!r} "
                f"predicted_velocity={self._format_velocity(self._command)} "
                f"duration={self.command_duration_s:.2f}s"
            )
            return self._consume_active_command()

    def _consume_active_command(self):
        command = self._command.copy()
        self._remaining_steps -= 1
        if self._remaining_steps == 0:
            print("[INFO] Learned command duration complete; velocity command reset to stop.")
            self._command[:] = 0.0
            self._active_label = "stop"
        return command

    def status_text(self):
        if self._remaining_steps <= 0:
            return ""
        return f" language_model='{self._active_label}' remaining={self._remaining_steps * self.control_dt:.2f}s"


def wrap_to_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def root_xy_yaw(env):
    robot = env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    return root_pos[:2].astype(np.float64), yaw_from_quaternion_wxyz(root_quat)


class GTReferencePathFollower:
    def __init__(self, episode, args):
        raw_path = episode.get("reference_path") or []
        if len(raw_path) < 2:
            raise ValueError("Episode has no usable reference_path for --go2-physics-follow-gt.")

        min_spacing = max(float(args.go2_physics_gt_min_waypoint_spacing), 0.0)
        path = []
        for waypoint in raw_path:
            if len(waypoint) < 2:
                continue
            point = np.asarray(waypoint[:2], dtype=np.float64)
            if not np.all(np.isfinite(point)):
                continue
            if path and np.linalg.norm(point - path[-1]) < min_spacing:
                continue
            path.append(point)
        if len(path) < 2:
            raise ValueError("Episode reference_path collapsed to fewer than two waypoints.")

        self.path = np.stack(path, axis=0)
        self.args = args
        self.target_index = 0
        self.done = False
        self.last_distance = float("inf")
        self.last_final_distance = float("inf")
        self.last_yaw_error = 0.0
        self.last_target_index = 0
        self.last_target_xy = self.path[0].copy()
        print(
            "[INFO] GT trajectory follower enabled: "
            f"waypoints={len(self.path)} raw_waypoints={len(raw_path)}, "
            f"waypoint_radius={args.go2_physics_gt_waypoint_radius:.2f}m, "
            f"final_radius={args.go2_physics_gt_final_radius:.2f}m, "
            f"lookahead={args.go2_physics_gt_lookahead_distance:.2f}m"
        )

    def reset(self):
        self.target_index = 0
        self.done = False
        self.last_distance = float("inf")
        self.last_final_distance = float("inf")
        self.last_yaw_error = 0.0
        self.last_target_index = 0
        self.last_target_xy = self.path[0].copy()

    def _advance_target_index(self, robot_xy):
        waypoint_radius = max(float(self.args.go2_physics_gt_waypoint_radius), 0.01)
        while self.target_index < len(self.path) - 1:
            if np.linalg.norm(self.path[self.target_index] - robot_xy) > waypoint_radius:
                break
            self.target_index += 1

    def _lookahead_index(self, robot_xy):
        lookahead = max(float(self.args.go2_physics_gt_lookahead_distance), 0.0)
        target_index = self.target_index
        while target_index < len(self.path) - 1:
            if np.linalg.norm(self.path[target_index] - robot_xy) >= lookahead:
                break
            target_index += 1
        return target_index

    def advance(self, env, torch, device, dtype):
        if self.done:
            return torch.zeros(3, device=device, dtype=dtype)

        robot_xy, yaw = root_xy_yaw(env)
        self._advance_target_index(robot_xy)

        final_delta = self.path[-1] - robot_xy
        self.last_final_distance = float(np.linalg.norm(final_delta))
        if self.target_index >= len(self.path) - 1 and self.last_final_distance <= self.args.go2_physics_gt_final_radius:
            self.done = True
            return torch.zeros(3, device=device, dtype=dtype)

        target_index = self._lookahead_index(robot_xy)
        target_xy = self.path[target_index]
        delta = target_xy - robot_xy
        distance = float(np.linalg.norm(delta))
        self.last_distance = distance
        self.last_target_index = target_index
        self.last_target_xy = target_xy.copy()

        if distance < 1e-6:
            yaw_error = 0.0
            local_y = 0.0
        else:
            desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
            yaw_error = wrap_to_pi(desired_yaw - yaw)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            local_y = -sin_yaw * float(delta[0]) + cos_yaw * float(delta[1])
        self.last_yaw_error = yaw_error

        max_forward = max(float(self.args.go2_physics_lin_vel), 0.0)
        max_lateral = max(float(self.args.go2_physics_strafe_vel), 0.0)
        max_yaw = max(float(self.args.go2_physics_ang_vel), 0.0)
        turn_in_place_angle = max(float(self.args.go2_physics_gt_turn_in_place_angle), 0.0)

        if abs(yaw_error) > turn_in_place_angle:
            vx = 0.0
            vy = 0.0
        else:
            heading_scale = max(math.cos(yaw_error), 0.0)
            vx = min(max_forward, max(float(self.args.go2_physics_gt_linear_gain) * distance, 0.0)) * heading_scale
            if distance > self.args.go2_physics_gt_waypoint_radius and vx > 1e-4:
                vx = max(vx, min(float(self.args.go2_physics_gt_min_forward_vel), max_forward))
            vy = float(self.args.go2_physics_gt_lateral_gain) * local_y
            vy = min(max(vy, -max_lateral), max_lateral)

        wz = float(self.args.go2_physics_gt_yaw_gain) * yaw_error
        wz = min(max(wz, -max_yaw), max_yaw)
        return torch.tensor([vx, vy, wz], device=device, dtype=dtype)

    def status_text(self):
        return (
            f" gt={self.last_target_index}/{len(self.path)-1} "
            f"dist={self.last_distance:.2f} final={self.last_final_distance:.2f} "
            f"yaw_err={math.degrees(self.last_yaw_error):+.1f}deg"
        )


def warmup_policy(args, env, policy, obs, torch):
    warmup_steps = max(int(args.go2_physics_warmup_steps), 0)
    if warmup_steps == 0:
        print("[INFO] Go2 physics policy warmup skipped.")
        return obs

    tensor = policy_obs_tensor(obs)
    zero_cmd = torch.zeros(3, device=tensor.device, dtype=tensor.dtype)
    print(f"[INFO] Warming up Go2 physics policy for {warmup_steps} steps...")
    for _ in range(warmup_steps):
        inject_velocity_command(obs, env, zero_cmd)
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    if bool(getattr(args, "go2_physics_language_commands", True)):  # ###2
        print("[INFO] Warmup complete. Enter language commands in the terminal to drive Go2.")  # ###2
    else:  # ###2
        print("[INFO] Warmup complete. Use W/S/A/D to drive Go2, SPACE to stop, R to reset, ENTER to finish.")  # ###2
    return obs


def create_robot_cameras():
    try:
        from pxr import Gf, UsdGeom
        import omni.usd
    except Exception as exc:
        print(f"[WARN] Could not create Go2 viewport cameras: {exc}")
        return {}

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {}

    # USD cameras look along local -Z with local +Y as up. This orientation maps
    # camera forward to the Go2 base +X axis and camera up to world/base +Z.
    robot_forward_camera_rot = (0.5, 0.5, -0.5, -0.5)
    camera_defs = {
        "rgbd_camera": ("/World/envs/env_0/Robot/base/rgbd_camera", (0.32, 0.0, 0.16), robot_forward_camera_rot, 72.0),
        "viz_rgb_camera": ("/World/envs/env_0/Robot/base/viz_rgb_camera", (-1.0, 0.0, 0.55), robot_forward_camera_rot, 100.0),
    }
    created = {}
    for name, (path, pos, rot, aperture) in camera_defs.items():
        camera_prim = UsdGeom.Camera.Define(stage, path)
        camera_prim.CreateFocalLengthAttr(10.0)
        camera_prim.CreateClippingRangeAttr().Set((0.01, 1000.0))
        camera_prim.CreateHorizontalApertureAttr(float(aperture))
        camera_prim.CreateVerticalApertureAttr(float(aperture))
        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xformable.AddOrientOp().Set(Gf.Quatf(float(rot[0]), Gf.Vec3f(float(rot[1]), float(rot[2]), float(rot[3]))))
        created[name] = path
    return created


def set_active_viewport_camera(camera_prim_path, headless=False, quiet=False):
    global _ACTIVE_VIEWPORT_CAMERA_PATH
    if headless or not camera_prim_path:
        return False
    if _ACTIVE_VIEWPORT_CAMERA_PATH == camera_prim_path:
        return True
    try:
        import omni.kit.commands
        from omni.kit.viewport.utility import get_active_viewport

        viewport_api = get_active_viewport()
        if viewport_api is None:
            if not quiet:
                print("[WARN] Could not get active viewport; leaving viewport camera unchanged.")
            return False
        omni.kit.commands.execute("SetViewportCamera", camera_path=camera_prim_path, viewport_api=viewport_api)
        _ACTIVE_VIEWPORT_CAMERA_PATH = camera_prim_path
        if not quiet:
            print(f"[INFO] Active viewport camera: {camera_prim_path}")
        return True
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not set active viewport camera to {camera_prim_path}: {exc}")
        return False


def ensure_viewport_updates_enabled(headless=False, quiet=False):
    if headless:
        return
    try:
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport

        viewport_api = get_active_viewport()
        if viewport_api is not None and hasattr(viewport_api, "updates_enabled"):
            viewport_api.updates_enabled = True
        viewport_window = ui.Workspace.get_window("Viewport")
        if viewport_window is not None:
            viewport_window.visible = True
    except Exception as exc:
        if not quiet:
            print(f"[WARN] Could not enable active viewport updates: {exc}")


def reset_viewport_camera_cache():
    global _ACTIVE_VIEWPORT_CAMERA_PATH
    _ACTIVE_VIEWPORT_CAMERA_PATH = None


def compute_follow_camera_pose(robot_pos_w, robot_quat_w, mode):
    heading = heading_from_quaternion_wxyz(robot_quat_w)
    if np.linalg.norm(heading[:2]) < 1e-6:
        heading = np.asarray([1.0, 0.0, 0.0])
    heading = heading / max(np.linalg.norm(heading), 1e-6)
    if mode == "wide_follow":
        distance, height, target_distance, target_height = 3.0, 1.8, 1.0, 0.4
    elif mode == "viz_rgb_camera":
        distance, height, target_distance, target_height = 1.65, 0.72, 0.75, 0.28
    elif mode in ("rgbd_camera", "robot_camera", "first_person"):
        distance, height, target_distance, target_height = -0.32, 0.38, 1.25, 0.30
    else:
        distance, height, target_distance, target_height = 1.2, 0.9, 1.0, 0.35
    eye = np.asarray(robot_pos_w, dtype=np.float64).copy()
    eye[:2] -= heading[:2] * distance
    eye[2] += height
    target = np.asarray(robot_pos_w, dtype=np.float64).copy()
    target[:2] += heading[:2] * target_distance
    target[2] += target_height
    return eye, target


def compute_fixed_overview_camera_pose(args, robot_pos_w, robot_quat_w):
    cached_pose = getattr(args, "_go2_physics_fixed_camera_pose", None)
    if cached_pose is not None:
        return cached_pose

    heading = heading_from_quaternion_wxyz(robot_quat_w)
    if np.linalg.norm(heading[:2]) < 1e-6:
        heading = np.asarray([1.0, 0.0, 0.0])
    heading = heading / max(np.linalg.norm(heading), 1e-6)
    lateral = np.asarray([-heading[1], heading[0], 0.0], dtype=np.float64)
    robot_pos_w = np.asarray(robot_pos_w, dtype=np.float64)
    # Kujiale rooms are compact indoor USDs; a far overview often lands behind
    # walls or ceilings. Keep this static debug camera close to the start pose.
    eye = robot_pos_w - heading * 1.15 + lateral * 0.45 + np.asarray([0.0, 0.0, 0.72])
    target = robot_pos_w + heading * 0.18 + np.asarray([0.0, 0.0, 0.18])
    cached_pose = (eye, target)
    setattr(args, "_go2_physics_fixed_camera_pose", cached_pose)
    print(
        "[INFO] Fixed overview camera "
        f"eye=[{eye[0]:+.2f}, {eye[1]:+.2f}, {eye[2]:+.2f}] "
        f"target=[{target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}]"
    )
    return cached_pose


def update_viewport_camera(args, env, camera_paths):
    if args.headless:
        return
    if getattr(args, "go2_physics_use_usd_camera_prim", False) and args.go2_physics_camera in camera_paths:
        set_active_viewport_camera(camera_paths[args.go2_physics_camera], headless=args.headless, quiet=True)
        return
    try:
        set_active_viewport_camera(PERSPECTIVE_CAMERA_PATH, headless=args.headless, quiet=True)
        robot = env.unwrapped.scene["robot"]
        robot_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()
        if args.go2_physics_camera == "fixed_overview":
            eye, target = compute_fixed_overview_camera_pose(args, robot_pos_w, robot_quat_w)
        else:
            eye, target = compute_follow_camera_pose(robot_pos_w, robot_quat_w, args.go2_physics_camera)
        env.unwrapped.sim.set_camera_view(eye=eye, target=target, camera_prim_path=PERSPECTIVE_CAMERA_PATH)
    except Exception as exc:
        print(f"[WARN] Could not update follow camera: {exc}")


def resolve_render_sync(args):
    configured = getattr(args, "go2_physics_render_sync", None)
    if configured is None:
        return not bool(args.headless)
    return bool(configured)


def sync_viewport_render(args, env, frame=None, force=False):
    if args.headless or not getattr(args, "_effective_go2_physics_render_sync", False):
        return
    if not force and frame is not None:
        sync_every = max(int(getattr(args, "go2_physics_render_sync_every", 1)), 1)
        if frame % sync_every != 0:
            return
    try:
        ensure_viewport_updates_enabled(args.headless, quiet=True)
        env.unwrapped.sim.render()
    except Exception as exc:
        print(f"[WARN] Could not force viewport render sync: {exc}")


def collision_include_scopes(args):
    mode = getattr(args, "go2_physics_collision_mode", "all")
    if mode == "none":
        return set()
    if mode in {"nav_collision", "floor_only"}:
        return {"floor"}
    if mode == "floor_walls":
        return {"floor", "wall"}
    return None


def ensure_static_collision(root_paths, include_scopes=None):
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd
    except Exception as exc:
        print(f"[WARN] Could not import USD collision APIs: {exc}")
        return {}

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {}

    edited = 0
    seen_paths = set()
    stats_by_scope = {}
    for root_path in root_paths:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            prim_path = str(prim.GetPath())
            if prim_path in seen_paths:
                continue
            if "/go2_height_floor_proxy" in prim_path:
                continue
            seen_paths.add(prim_path)
            scope = scope_from_kujiale_path(prim_path)
            if include_scopes is not None and scope not in include_scopes:
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            try:
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                if not mesh_api.GetApproximationAttr().HasAuthoredValueOpinion():
                    mesh_api.CreateApproximationAttr("none")
            except Exception:
                pass
            stats_by_scope[scope] = stats_by_scope.get(scope, 0) + 1
            edited += 1
    if edited:
        print(f"[INFO] Ensured static USD collision on {edited} mesh prims.")
    return stats_by_scope


def scope_from_kujiale_path(path):
    parts = [part for part in str(path).split("/") if part]
    if "Meshes" in parts:
        mesh_index = parts.index("Meshes")
        if mesh_index + 1 < len(parts):
            return parts[mesh_index + 1]
    for candidate in ("floor", "wall", "ceiling", "door", "terrain", "other"):
        if candidate in parts:
            return candidate
    return "unknown"


def format_top_counts(counts, limit=12):
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    visible = ordered[:limit]
    text = ", ".join(f"{name}={count}" for name, count in visible)
    if len(ordered) > limit:
        text += f", ...(+{len(ordered) - limit} scopes)"
    return text


def stage_mesh_summary(root_path):
    try:
        from pxr import Usd, UsdGeom
        import omni.usd
    except Exception as exc:
        return {"error": f"USD APIs unavailable: {exc}"}

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {"error": "stage unavailable"}
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return {"error": "root prim not found"}

    first_mesh = None
    counts = {}
    total = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = str(prim.GetPath())
        if first_mesh is None:
            first_mesh = prim_path
        total += 1
        scope = scope_from_kujiale_path(prim_path)
        counts[scope] = counts.get(scope, 0) + 1
    return {"first_mesh": first_mesh, "counts": counts, "total": total}


def print_stage_mesh_diagnostics(root_paths):
    for root_path in root_paths:
        summary = stage_mesh_summary(root_path)
        error = summary.get("error")
        if error:
            print(f"[DIAG] stage root={root_path}: {error}")
            continue
        print(
            f"[DIAG] stage root={root_path}: mesh_count={summary['total']}, "
            f"first_mesh={summary['first_mesh'] or 'none'}, "
            f"scopes={format_top_counts(summary['counts'])}"
        )


def print_collision_scope_diagnostics(stats_by_scope):
    print(f"[DIAG] collision meshes by scope: {format_top_counts(stats_by_scope)}")


def get_scene_entity(env, name):
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        return None
    try:
        return scene[name]
    except Exception:
        pass
    sensors = getattr(scene, "sensors", None)
    if sensors is not None:
        try:
            return sensors[name]
        except Exception:
            pass
    return None


def tensor_to_numpy(value):
    if value is None:
        return None
    try:
        return value.detach().cpu().numpy()
    except Exception:
        return np.asarray(value)


def finite_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    if finite.size == 0:
        return f"finite=0, nan={nan_count}, inf={inf_count}"
    return (
        f"min={finite.min():+.3f}, max={finite.max():+.3f}, "
        f"mean={finite.mean():+.3f}, std={finite.std():.3f}, "
        f"nan={nan_count}, inf={inf_count}"
    )


def quaternion_to_rpy_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[0] != 4 or np.linalg.norm(quat) < 1e-6:
        return (0.0, 0.0, 0.0)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = float(np.clip(sin_pitch, -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def diagnose_height_scanner_setup(env):
    scanner = get_scene_entity(env, "height_scanner")
    if scanner is None:
        print("[DIAG] height_scanner: missing")
        return
    cfg = getattr(scanner, "cfg", None)
    mesh_paths = list(getattr(cfg, "mesh_prim_paths", []) or [])
    ray_alignment = getattr(cfg, "ray_alignment", "unknown")
    num_rays = getattr(scanner, "num_rays", "unknown")
    print(f"[DIAG] height_scanner cfg: mesh_prim_paths={mesh_paths}, ray_alignment={ray_alignment}, rays={num_rays}")
    for mesh_path in mesh_paths:
        summary = stage_mesh_summary(mesh_path)
        error = summary.get("error")
        if error:
            print(f"[DIAG] height_scanner mesh root={mesh_path}: {error}")
        else:
            print(
                f"[DIAG] height_scanner mesh root={mesh_path}: "
                f"mesh_count={summary['total']}, first_mesh={summary['first_mesh'] or 'none'}, "
                f"scopes={format_top_counts(summary['counts'])}"
            )


def height_scan_diagnostics(env, obs):
    scanner = get_scene_entity(env, "height_scanner")
    if scanner is None:
        return "height_scanner=missing"
    try:
        data = scanner.data
        sensor_pos = tensor_to_numpy(data.pos_w)[0]
        ray_hits = tensor_to_numpy(data.ray_hits_w)[0]
        hit_z = ray_hits[:, 2]
        raw_height = sensor_pos[2] - hit_z - 0.5
        text = (
            f"height_raw({finite_stats(raw_height)}), "
            f"hit_z({finite_stats(hit_z)})"
        )
    except Exception as exc:
        text = f"height_raw=unavailable({exc})"

    try:
        tensor = tensor_to_numpy(policy_obs_tensor(obs))[0]
        if tensor.shape[0] > 48:
            policy_height = tensor[48:]
            clipped = int((np.abs(policy_height) >= 0.999).sum())
            text += f", policy_height({finite_stats(policy_height)}, clipped={clipped}/{policy_height.size})"
        else:
            text += ", policy_height=absent(flat task)"
    except Exception as exc:
        text += f", policy_height=unavailable({exc})"
    return text


def contact_diagnostics(env, top_k=5):
    contact_sensor = get_scene_entity(env, "contact_forces")
    if contact_sensor is None:
        return "contact_forces=missing"
    try:
        forces = tensor_to_numpy(contact_sensor.data.net_forces_w)[0]
        norms = np.linalg.norm(forces, axis=-1)
        body_names = list(getattr(contact_sensor, "body_names", []))
        top_indices = np.argsort(-norms)[: max(int(top_k), 1)]
        top_parts = []
        for index in top_indices:
            force = float(norms[index])
            if force <= 1e-4:
                continue
            name = body_names[index] if index < len(body_names) else f"body_{index}"
            top_parts.append(f"{name}:{force:.1f}N")
        top_text = ", ".join(top_parts) if top_parts else "none"
        base_force = 0.0
        foot_force = 0.0
        thigh_force = 0.0
        for index, force in enumerate(norms):
            name = body_names[index].lower() if index < len(body_names) else ""
            if "base" in name:
                base_force = max(base_force, float(force))
            if "foot" in name:
                foot_force = max(foot_force, float(force))
            if "thigh" in name:
                thigh_force = max(thigh_force, float(force))
        return (
            f"contact max={float(norms.max()):.1f}N, "
            f"base={base_force:.1f}N, foot_max={foot_force:.1f}N, thigh_max={thigh_force:.1f}N, top={top_text}"
        )
    except Exception as exc:
        return f"contact_forces=unavailable({exc})"


def control_dt_from_env(env):
    for attr_name in ("step_dt", "physics_dt"):
        value = getattr(env.unwrapped, attr_name, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    sim = getattr(env.unwrapped, "sim", None)
    if sim is not None:
        for attr_name in ("dt", "physics_dt"):
            value = getattr(sim, attr_name, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
    return None


def print_go2_diagnostics(args, env, obs, command, frame, state):
    robot = env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
    root_ang_vel = robot.data.root_ang_vel_w[0].detach().cpu().numpy()
    roll, pitch, yaw = quaternion_to_rpy_wxyz(root_quat)

    last_root = state.get("last_root")
    last_frame = state.get("last_frame")
    xy_speed_text = "n/a"
    xy_delta_text = "n/a"
    if last_root is not None and last_frame is not None and frame > last_frame:
        delta = root_pos - last_root
        xy_delta = float(np.linalg.norm(delta[:2]))
        dt = control_dt_from_env(env)
        if dt is not None and dt > 0.0:
            xy_speed_text = f"{xy_delta / ((frame - last_frame) * dt):.3f}m/s"
        xy_delta_text = f"{xy_delta:.3f}m"
    state["last_root"] = root_pos.copy()
    state["last_frame"] = frame

    command_np = command.detach().cpu().numpy()
    print(
        "[DIAG] "
        f"step={frame} cmd=[{command_np[0]:+.2f}, {command_np[1]:+.2f}, {command_np[2]:+.2f}] "
        f"root=[{root_pos[0]:+.3f}, {root_pos[1]:+.3f}, {root_pos[2]:+.3f}] "
        f"rpy_deg=[{math.degrees(roll):+.1f}, {math.degrees(pitch):+.1f}, {math.degrees(yaw):+.1f}] "
        f"lin_w=[{root_lin_vel[0]:+.3f}, {root_lin_vel[1]:+.3f}, {root_lin_vel[2]:+.3f}] "
        f"ang_w=[{root_ang_vel[0]:+.3f}, {root_ang_vel[1]:+.3f}, {root_ang_vel[2]:+.3f}] "
        f"window_xy_delta={xy_delta_text} window_xy_speed={xy_speed_text}"
    )
    print(f"[DIAG] {height_scan_diagnostics(env, obs)}")
    print(f"[DIAG] {contact_diagnostics(env)}")


def describe_termination_state(env):
    manager = getattr(env.unwrapped, "termination_manager", None)
    if manager is None:
        return "unknown"

    active = []
    for name in getattr(manager, "active_terms", []):
        try:
            value = manager.get_term(name)
            is_active = bool(value[0].item()) if hasattr(value[0], "item") else bool(value[0])
        except Exception:
            continue
        if is_active:
            active.append(name)
    return ", ".join(active) if active else "unknown"


def robot_transform_row(env, frame):
    robot = env.unwrapped.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
    robot_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()
    eye, target = compute_follow_camera_pose(robot_pos_w, robot_quat_w, "close_follow")
    look = target - eye
    norm = np.linalg.norm(look)
    if norm > 1e-6:
        look = look / norm
    row = {
        "agent": "go2",
        "controller": "physics",
        "frame": frame,
        "pos_x": robot_pos_w[0],
        "pos_y": robot_pos_w[1],
        "pos_z": robot_pos_w[2],
        "rot_w": robot_quat_w[0],
        "rot_x": robot_quat_w[1],
        "rot_y": robot_quat_w[2],
        "rot_z": robot_quat_w[3],
        "camera_pos_x": eye[0],
        "camera_pos_y": eye[1],
        "camera_pos_z": eye[2],
        "look_x": look[0],
        "look_y": look[1],
        "look_z": look[2],
    }
    try:
        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
        row["joint_pos_l2"] = float(np.linalg.norm(joint_pos))
    except Exception:
        pass
    return row


def resolve_policy_checkpoint(args, agent_cfg, get_published_pretrained_checkpoint):
    if args.go2_physics_checkpoint:
        checkpoint = os.path.abspath(os.path.expanduser(args.go2_physics_checkpoint))
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"--go2-physics-checkpoint does not exist: {checkpoint}")
        print(f"[INFO] Loading explicit Go2 physics checkpoint: {checkpoint}")
        return checkpoint

    train_task_name = args.go2_physics_task.split(":")[-1].replace("-Play", "")
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    if checkpoint:
        print(f"[INFO] Loading IsaacLab native pretrained Go2 checkpoint: {checkpoint}")
        return checkpoint

    fallback = os.path.abspath(os.path.expanduser(args.go2_physics_fallback_checkpoint))
    if os.path.exists(fallback):
        print(
            "[WARN] IsaacLab native pretrained checkpoint was unavailable. "
            f"Trying local fallback checkpoint: {fallback}"
        )
        return fallback

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    raise FileNotFoundError(
        "No Go2 physics policy checkpoint was found. "
        f"Native pretrained task={train_task_name}, fallback={fallback}, local log root={log_root_path}"
    )


def checkpoint_actor_input_width(checkpoint_path, torch):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[WARN] Could not inspect checkpoint shape before loading: {exc}")
        return None, None

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "actor_critic_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

    if not isinstance(state_dict, dict):
        return None, None

    for name, value in state_dict.items():
        if not name.endswith("actor.0.weight"):
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1]), name
    return None, None


def validate_checkpoint_observation_width(checkpoint_path, obs, torch):
    obs_width = int(policy_obs_tensor(obs).shape[1])
    actor_width, actor_key = checkpoint_actor_input_width(checkpoint_path, torch)
    if actor_width is None:
        print(f"[WARN] Could not infer checkpoint observation width for: {checkpoint_path}")
        return
    if actor_width != obs_width:
        raise RuntimeError(
            "Go2 physics checkpoint does not match this IsaacLab task observation width. "
            f"checkpoint {actor_key} expects {actor_width}, environment policy obs has {obs_width}. "
            "Pass a matching --go2-physics-checkpoint or switch --go2-controller kinematic for visual-only fallback."
        )
    print(f"[INFO] Checkpoint observation width matches environment: {obs_width}")


def _legacy_rsl_rl_model_to_mlp_state(model_state, current_state, prefix, torch):  ###
    converted = {}  ###
    for key, current_value in current_state.items():  ###
        source_key = None  ###
        if key.startswith("mlp."):  ###
            mlp_key = key[len("mlp."):]  ###
            source_key = f"{prefix}.{mlp_key}"  ###
        elif key == "distribution.std_param":  ###
            source_key = "std"  ###
        elif key == "distribution.log_std_param":  ###
            source_key = "std"  ###
        if source_key not in model_state:  ###
            continue  ###
        value = model_state[source_key]  ###
        if key == "distribution.log_std_param":  ###
            value = torch.log(value)  ###
        converted[key] = value.to(current_value.device) if hasattr(value, "to") else value  ###
    return converted  ###


def load_rsl_rl_checkpoint_compatible(runner, checkpoint_path, torch):  ###
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)  ###
    if "model_state_dict" in checkpoint or "actor_state_dict" in checkpoint:  ###
        runner.load(checkpoint_path)  ###
        return  ###
    raise KeyError("model_state_dict")  ###
    print("[INFO] Loaded legacy RSL-RL checkpoint format through compatibility mapper.")  ###


def run_go2_physics_episode(episode, args, repo_root):
    configure_isaaclab_paths(args.isaaclab_root)

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = args.headless
    launcher_args.width = max(int(args.go2_physics_render_width), 320)
    launcher_args.height = max(int(args.go2_physics_render_height), 240)
    launcher_args.window_width = max(int(args.go2_physics_render_width), 320)
    launcher_args.window_height = max(int(args.go2_physics_render_height), 240)
    launcher_args.anti_aliasing = max(int(args.go2_physics_anti_aliasing), 0)
    launcher_args.renderer = args.go2_physics_renderer
    launcher_args.multi_gpu = bool(args.go2_physics_multi_gpu)
    if hasattr(launcher_args, "enable_cameras"):
        launcher_args.enable_cameras = True
    if hasattr(launcher_args, "experience"):
        launcher_args.experience = args.go2_physics_experience
    if hasattr(launcher_args, "kit_args"):
        kit_args = shlex.split(getattr(launcher_args, "kit_args", "") or "")
        kit_args.append(f"--/log/level={args.go2_physics_kit_log_level}")
        launcher_args.kit_args = " ".join(shlex.quote(item) for item in kit_args)

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        app_launcher = AppLauncher(launcher_args)
    finally:
        sys.argv = original_argv
    simulation_app = app_launcher.app
    args._effective_go2_physics_render_sync = resolve_render_sync(args)
    print(
        "[INFO] Go2 render config: "
        f"{launcher_args.width}x{launcher_args.height}, "
        f"renderer={launcher_args.renderer}, aa={launcher_args.anti_aliasing}, "
        f"multi_gpu={launcher_args.multi_gpu}, "
        f"extra_render_sync={args._effective_go2_physics_render_sync}, "
        f"sync_every={max(int(args.go2_physics_render_sync_every), 1)}, "
        f"kit_log_level={args.go2_physics_kit_log_level}"
    )
    if args.go2_physics_kit_log_level in ("error", "fatal"):
        print(
            "[INFO] Isaac/Kit warning spam is hidden from the console. "
            "Use --go2-physics-kit-log-level warning to show Kit warnings again."
        )
    args._resolved_go2_physics_asset_root = configure_isaac_asset_root(args)

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import cli_args
    import isaaclab_tasks  # noqa: F401
    # from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    ### IsaacLab newer versions moved pretrained_checkpoint into isaaclab_rl.utils.
    ### Keep compatibility with both newer and older IsaacLab installs.
    ### Try the newer import path first for the current environment.
    try:  ###
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  ###
    except ModuleNotFoundError:  ###
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  ###
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    try:  ###
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  ###
    except ImportError:  ###
        def handle_deprecated_rsl_rl_cfg(cfg, version):  ###
            return cfg  ###
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env = None
    keyboard = None
    language_commander = None  # ###2
    saved_csv_path = None
    success = False
    rows = []
    try:
        reset_viewport_camera_cache()
        if hasattr(args, "_go2_physics_fixed_camera_pose"):
            delattr(args, "_go2_physics_fixed_camera_pose")
        env_cfg = parse_env_cfg(args.go2_physics_task, num_envs=1)
        env_cfg = configure_env_from_episode(args, env_cfg, episode, repo_root)
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.go2_physics_task, make_rsl_rl_args(args))
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))  ###

        render_mode = None if args.headless else "human"
        env = gym.make(args.go2_physics_task, cfg=env_cfg, render_mode=render_mode)
        camera_paths = create_robot_cameras()
        kujiale_diag_roots = ("/World/ground", "/World/ground/terrain")
        if args.go2_physics_scene == "kujiale" and getattr(args, "go2_physics_diagnostics", False):
            print_stage_mesh_diagnostics(kujiale_diag_roots)
        if args.go2_physics_scene == "kujiale":
            include_scopes = collision_include_scopes(args)
            if include_scopes == set():
                collision_stats = {}
                print("[INFO] Kujiale runtime static collision disabled by --go2-physics-collision-mode none.")
            else:
                collision_stats = ensure_static_collision(kujiale_diag_roots, include_scopes=include_scopes)
            if getattr(args, "go2_physics_diagnostics", False):
                print_collision_scope_diagnostics(collision_stats)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if not args.headless:
            ensure_viewport_updates_enabled(args.headless, quiet=True)
            sim = env.unwrapped.sim
            print(
                "[INFO] Go2 viewport render state: "
                f"render_mode={getattr(sim, 'render_mode', 'unknown')}, "
                f"has_gui={sim.has_gui()}, has_rtx_sensors={sim.has_rtx_sensors()}, "
                f"use_fabric={sim.is_fabric_enabled()}"
            )

        resume_path = resolve_policy_checkpoint(args, agent_cfg, get_published_pretrained_checkpoint)
        obs = env.get_observations()
        if getattr(args, "go2_physics_diagnostics", False):
            diagnose_height_scanner_setup(env)
        validate_checkpoint_observation_width(resume_path, obs, torch)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # runner.load(resume_path)
        load_rsl_rl_checkpoint_compatible(runner, resume_path, torch)  ###
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        update_viewport_camera(args, env, camera_paths)
        sync_viewport_render(args, env, force=True)
        obs = warmup_policy(args, env, policy, obs, torch)
        update_viewport_camera(args, env, camera_paths)
        sync_viewport_render(args, env, force=True)

        tensor = policy_obs_tensor(obs)
        scripted_command = get_scripted_command(args, torch, tensor.device, tensor.dtype)
        gt_follower = GTReferencePathFollower(episode, args) if getattr(args, "go2_physics_follow_gt", False) else None
        if gt_follower is not None and scripted_command is not None:
            print("[WARN] --go2-physics-follow-gt is enabled; ignoring --go2-physics-scripted-command.")
            scripted_command = None
        language_commands_enabled = bool(getattr(args, "go2_physics_language_commands", True))  # ###2
        language_model_checkpoint = getattr(args, "go2_physics_language_model_checkpoint", None)
        if language_commands_enabled and scripted_command is None and gt_follower is None:  # ###2
            control_dt = control_dt_from_env(env) or getattr(args, "_go2_physics_control_dt", 0.02)  # ###2
            if language_model_checkpoint:
                language_commander = LearnedLanguageVelocityCommander(args, control_dt)
            else:
                language_commander = LanguageVelocityCommander(args, control_dt)  # ###2
            if not args.headless:  # ###2
                print("[INFO] WASD keyboard teleoperation is disabled while language prompt is active.")  # ###2
        elif language_commands_enabled and (scripted_command is not None or gt_follower is not None):  # ###2
            print("[WARN] Language command prompt disabled because scripted/GT control is active.")  # ###2

        keyboard = None if args.headless or language_commander is not None else WASDVelocityKeyboard(args)  # ###2
        if args.headless and scripted_command is None and gt_follower is None and language_commander is None:
            print("[WARN] Headless physics run has no scripted command; it will stop at --go2-physics-max-headless-steps.")

        robot = env.unwrapped.scene["robot"]
        initial_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        initial_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        termination_reported = False
        diagnostic_state = {}
        frame = 0

        while simulation_app.is_running():
            if keyboard is not None and keyboard.quit_requested:
                print("[INFO] Quit requested from keyboard.")
                break
            if language_commander is not None and language_commander.quit_requested:  # ###2
                print("[INFO] Quit requested from language prompt.")  # ###2
                break  # ###2

            if keyboard is not None and keyboard.reset_requested:
                obs, _ = env.reset()
                obs = warmup_policy(args, env, policy, obs, torch)
                keyboard.reset()
                keyboard.reset_requested = False
                if language_commander is not None:  # ###2
                    language_commander.reset()  # ###2
                frame = 0
                rows = []
                termination_reported = False
                diagnostic_state = {}
                if gt_follower is not None:
                    gt_follower.reset()
                reset_viewport_camera_cache()
                if hasattr(args, "_go2_physics_fixed_camera_pose"):
                    delattr(args, "_go2_physics_fixed_camera_pose")
                update_viewport_camera(args, env, camera_paths)
                sync_viewport_render(args, env, force=True)
                print("[INFO] Episode reset requested.")
                continue

            with torch.inference_mode():
                tensor = policy_obs_tensor(obs)
                if gt_follower is not None:
                    if keyboard is not None:
                        keyboard.poll_events()
                    command = gt_follower.advance(env, torch, tensor.device, tensor.dtype)
                elif scripted_command is not None:
                    command = scripted_command.clone() if frame < args.go2_physics_scripted_steps else torch.zeros(
                        3, device=tensor.device, dtype=tensor.dtype
                    )
                elif language_commander is not None:  # ###2
                    command_np = language_commander.advance()  # ###2
                    command = torch.tensor(command_np, device=tensor.device, dtype=tensor.dtype)  # ###2
                else:
                    command_np = keyboard.advance() if keyboard is not None else np.zeros(3, dtype=np.float32)
                    command = torch.tensor(command_np, device=tensor.device, dtype=tensor.dtype)

                inject_velocity_command(obs, env, command)
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

            update_viewport_camera(args, env, camera_paths)
            sync_viewport_render(args, env, frame=frame)

            if args.record and frame % 50 == 0:
                rows.append(robot_transform_row(env, frame))

            if frame % max(int(args.go2_physics_print_every), 1) == 0:
                root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                command_np = command.detach().cpu().numpy()
                print(
                    f"[INFO] step={frame} cmd=[{command_np[0]:+.2f}, {command_np[1]:+.2f}, {command_np[2]:+.2f}] "
                    f"root=[{root_pos[0]:+.2f}, {root_pos[1]:+.2f}, {root_pos[2]:+.2f}]"
                    f"{gt_follower.status_text() if gt_follower is not None else ''}"  # ###2
                    f"{language_commander.status_text() if language_commander is not None else ''}"  # ###2
                )

            diagnostics_every = int(getattr(args, "go2_physics_diagnostics_every", 0) or args.go2_physics_print_every)
            if getattr(args, "go2_physics_diagnostics", False) and frame % max(diagnostics_every, 1) == 0:
                print_go2_diagnostics(args, env, obs, command, frame, diagnostic_state)

            done_flag = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones)
            if done_flag:
                reason = describe_termination_state(env)
                if not termination_reported:
                    termination_action_text = (  # ###2
                        "Stopping language-command run."  # ###2
                        if language_commander is not None  # ###2
                        else "Keeping Isaac Sim open; press R to reset, ESC to quit, or ENTER to finish."  # ###2
                    )  # ###2
                    print(
                        f"[WARN] Go2 physics environment terminated ({reason}). "
                        f"{termination_action_text}"  # ###2
                    )
                    termination_reported = True
                if keyboard is not None:
                    keyboard.reset()
                if language_commander is not None:  # ###2
                    print("[ABORT] Language-command Go2 physics run stopped because the environment terminated.")  # ###2
                    break  # ###2
                if scripted_command is not None:
                    print("[ABORT] Scripted Go2 physics command stopped because the environment terminated.")
                    break

            if keyboard is not None and keyboard.mission_complete:
                success = True
                break
            if language_commander is not None and language_commander.mission_complete:  # ###2
                success = True  # ###2
                break  # ###2

            if gt_follower is not None and gt_follower.done:
                root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                print(
                    "[INFO] GT trajectory follower finished. "
                    f"root=[{root_pos[0]:+.2f}, {root_pos[1]:+.2f}, {root_pos[2]:+.2f}], "
                    f"final_error={gt_follower.last_final_distance:.2f}m, "
                    f"target_index={gt_follower.target_index}/{len(gt_follower.path)-1}"
                )
                success = True
                break

            if scripted_command is not None and frame >= args.go2_physics_scripted_steps:
                final_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
                final_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().copy()
                root_delta = final_root_pos - initial_root_pos
                joint_delta = float(np.linalg.norm(final_joint_pos - initial_joint_pos))
                print(
                    "[INFO] Scripted Go2 physics test finished. "
                    f"root_delta=[{root_delta[0]:+.3f}, {root_delta[1]:+.3f}, {root_delta[2]:+.3f}], "
                    f"joint_delta_l2={joint_delta:.4f}"
                )
                success = True
                break

            if gt_follower is not None and args.go2_physics_gt_max_steps > 0 and frame >= args.go2_physics_gt_max_steps:
                print("[INFO] GT trajectory follower step limit reached.")
                success = True
                break

            if args.headless and scripted_command is None and language_commander is None and frame >= args.go2_physics_max_headless_steps:  # ###2
                print("[INFO] Headless physics safety limit reached.")
                break

            frame += 1

        if args.record and rows:
            saved_csv_path = os.path.join(args.work_dir, f"{episode['episode_id']}.csv")
            pd.DataFrame(rows).to_csv(saved_csv_path, index=False)
            print(f"[INFO] Saved Go2 physics trajectory: {saved_csv_path}")
        elif not success:
            print("\n\n [ABORT] Simulation window closed without pressing ENTER.")

    except Exception as exc:
        print(f"[ERROR] Go2 physics controller failed: {exc}")
        traceback.print_exc()
        success = False
    finally:
        if keyboard is not None:
            keyboard.close()
        if env is not None:
            env.close()
        simulation_app.close()

    return saved_csv_path, success
