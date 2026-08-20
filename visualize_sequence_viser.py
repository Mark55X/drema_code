#!/usr/bin/env python
"""
Interactive 3D Web Visualizer for Dynamic VG-Mapping & RecurGS Gaussian Splatting sequences.

Built with Viser (https://viser.studio):
- Interactive time-slider and real-time playback of Gaussian Splatting evolution across timesteps.
- Semantic segmentation view (color-coded by Object ID) vs Photorealistic RGB vs Elevation heatmap.
- Interactive object visibility filters (isolate dynamic objects, hide table/background).
- Optional TSDF Marching Cubes solid mesh overlay, workspace bounding boxes, and camera presets.
"""

import os
import sys
import glob
import time
import json
import argparse
import colorsys
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch

try:
    import viser
    import viser.transforms as tf
    HAS_VISER = True
except ImportError:
    HAS_VISER = False


def generate_object_palette(max_id: int = 32) -> Dict[int, np.ndarray]:
    """Generates distinct, high-contrast RGB colors for semantic object segmentation."""
    palette = {
        0: np.array([0.65, 0.65, 0.65], dtype=np.float32),  # Background / Table: Neutral light gray
        -1: np.array([0.9, 0.2, 0.2], dtype=np.float32),    # Dynamic foreground fallback: Crimson
    }
    # Pre-defined vibrant palette for common object IDs
    preset_colors = [
        [0.92, 0.25, 0.20],  # 1: Coral Red
        [0.18, 0.55, 0.95],  # 2: Ocean Blue
        [0.15, 0.80, 0.40],  # 3: Emerald Green
        [0.98, 0.65, 0.10],  # 4: Amber Orange
        [0.72, 0.25, 0.88],  # 5: Purple
        [0.10, 0.85, 0.85],  # 6: Cyan
        [0.95, 0.30, 0.65],  # 7: Rose Pink
        [0.85, 0.90, 0.15],  # 8: Lime Yellow
        [0.45, 0.35, 0.85],  # 9: Indigo
        [0.85, 0.45, 0.20],  # 10: Terracotta
    ]
    for i, col in enumerate(preset_colors, start=1):
        palette[i] = np.array(col, dtype=np.float32)

    # For any larger IDs, use golden-ratio HSV sampling
    golden_ratio = 0.618033988749895
    h = 0.1
    for oid in range(len(preset_colors) + 1, max_id + 10):
        h = (h + golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(h, 0.85, 0.92)
        palette[oid] = np.array(rgb, dtype=np.float32)

    return palette


def load_gaussian_timestep(file_path: str) -> Dict[str, np.ndarray]:
    """
    Loads a timestep Gaussian state from either a PyTorch (.pt) or PLY (.ply) file.
    Returns dictionary with numpy arrays: 'xyz', 'rgb', 'scale', 'obj_id'.
    """
    if file_path.endswith(".pt"):
        try:
            data = torch.load(file_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(file_path, map_location="cpu")
        xyz = data['xyz'].numpy() if isinstance(data['xyz'], torch.Tensor) else np.array(data['xyz'])
        rgb = data['rgb'].numpy() if isinstance(data['rgb'], torch.Tensor) else np.array(data['rgb'])
        scale = data['scale'].numpy() if isinstance(data['scale'], torch.Tensor) else np.array(data['scale'])
        if 'obj_id' in data:
            obj_id = data['obj_id'].numpy() if isinstance(data['obj_id'], torch.Tensor) else np.array(data['obj_id'])
        else:
            obj_id = np.zeros(len(xyz), dtype=np.int32)
        return {'xyz': xyz, 'rgb': np.clip(rgb, 0.0, 1.0), 'scale': scale, 'obj_id': obj_id}

    elif file_path.endswith(".ply"):
        try:
            from plyfile import PlyData
            ply = PlyData.read(file_path)
            v = ply['vertex']
            xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)

            if 'red' in v:
                rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.float32) / 255.0
            elif 'f_dc_0' in v:
                SH_C0 = 0.28209479177387814
                f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)
                rgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
            else:
                rgb = np.ones_like(xyz) * 0.7

            if 'scale_0' in v:
                log_scale = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
                scale = np.exp(log_scale)
            else:
                scale = np.ones_like(xyz) * 0.005

            if 'obj_id' in v:
                obj_id = np.array(v['obj_id'], dtype=np.int32)
            else:
                obj_id = np.zeros(len(xyz), dtype=np.int32)

            return {'xyz': xyz, 'rgb': rgb, 'scale': scale, 'obj_id': obj_id}
        except Exception as e:
            print(f"Error reading PLY {file_path}: {e}")
            return {'xyz': np.empty((0, 3)), 'rgb': np.empty((0, 3)), 'scale': np.empty((0, 3)), 'obj_id': np.empty((0,), dtype=np.int32)}

    raise ValueError(f"Unsupported file extension: {file_path}")


def discover_sequence_timesteps(data_dir: str) -> List[Tuple[int, str]]:
    """
    Discovers all available timestep files in data_dir (checks sequence_manifest.json or scans gaussians/).
    Returns a sorted list of tuples: [(timestep_num, file_path), ...]
    """
    manifest_path = os.path.join(data_dir, "sequence_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            timesteps = []
            for item in manifest.get("timesteps", []):
                t_num = item["timestep"]
                files = item.get("files", {})
                # Prefer .pt for fast loading, fallback to .ply
                if "pt" in files:
                    full_p = os.path.join(data_dir, files["pt"])
                    if os.path.exists(full_p):
                        timesteps.append((t_num, full_p))
                        continue
                if "ply" in files:
                    full_p = os.path.join(data_dir, files["ply"])
                    if os.path.exists(full_p):
                        timesteps.append((t_num, full_p))
                        continue
            if len(timesteps) > 0:
                timesteps.sort(key=lambda x: x[0])
                return timesteps
        except Exception as e:
            print(f"[Warning] Failed to read sequence_manifest.json: {e}")

    # Fallback directory scan in gaussians/
    gaussians_dir = os.path.join(data_dir, "gaussians") if os.path.exists(os.path.join(data_dir, "gaussians")) else data_dir
    pt_files = sorted(glob.glob(os.path.join(gaussians_dir, "timestep_*.pt")))
    if len(pt_files) > 0:
        results = []
        for p in pt_files:
            base = os.path.basename(p).replace("timestep_", "").replace(".pt", "")
            if base.isdigit():
                results.append((int(base), p))
        results.sort(key=lambda x: x[0])
        return results

    ply_files = sorted(glob.glob(os.path.join(gaussians_dir, "timestep_*.ply")))
    if len(ply_files) > 0:
        results = []
        for p in ply_files:
            base = os.path.basename(p).replace("timestep_", "").replace(".ply", "")
            if base.isdigit():
                results.append((int(base), p))
        results.sort(key=lambda x: x[0])
        return results

    return []


def launch_viser_server(
    data_dir: str,
    port: int = 8080,
    host: str = "0.0.0.0",
    point_size: float = 0.006,
    share: bool = False,
    block: bool = True
):
    """
    Launches an interactive 3D Viser Web Visualizer for a sequence of 3D Gaussian scenes.
    """
    if not HAS_VISER:
        print("❌ Error: 'viser' is not installed! Run: pip install viser")
        return

    timesteps = discover_sequence_timesteps(data_dir)
    if len(timesteps) == 0:
        print(f"❌ Error: No timestep files (timestep_*.pt / timestep_*.ply) found in {data_dir}")
        return

    print("=" * 70)
    print(" DREMA Dynamic 3D Gaussian Splatting & RecurGS Interactive Visualizer")
    print("=" * 70)
    print(f"Data Directory  : {os.path.abspath(data_dir)}")
    print(f"Total Timesteps : {len(timesteps)} (t = {timesteps[0][0]} .. {timesteps[-1][0]})")
    print(f"Web GUI URL     : http://localhost:{port} (or http://<server-ip>:{port})")
    print("=" * 70)

    # Pre-cache sequence data
    print(">>> Pre-loading Gaussian states into memory for smooth playback...")
    sequence_cache = {}
    all_obj_ids = set()

    for t_num, f_path in timesteps:
        data = load_gaussian_timestep(f_path)
        sequence_cache[t_num] = data
        if len(data['obj_id']) > 0:
            for oid in np.unique(data['obj_id']):
                all_obj_ids.add(int(oid))

    sorted_obj_ids = sorted(list(all_obj_ids))
    palette = generate_object_palette(max(sorted_obj_ids) if len(sorted_obj_ids) > 0 else 10)
    print(f"✓ Cached {len(sequence_cache)} timesteps. Discovered object IDs: {sorted_obj_ids}")

    # Start Viser Server
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+z")

    if share:
        try:
            share_url = server.request_share_url()
            print(f"🔗 Public Share URL: {share_url}")
        except Exception as e:
            print(f"⚠️ Could not generate share URL: {e}")

    # App State
    state = {
        'current_t': timesteps[0][0],
        'is_playing': False,
        'fps': 5,
        'color_mode': "RGB (Photorealistic)",
        'render_mode': "Point Cloud (Fast)",
        'point_size': point_size,
        'active_objects': {oid: True for oid in sorted_obj_ids},
        'show_workspace_bounds': True,
        'show_tsdf_mesh': False,
        'loop': True
    }

    # Load TSDF mesh if available
    tsdf_mesh_path = os.path.join(data_dir, "final_tsdf_mesh.obj")
    tsdf_mesh_obj = None
    if os.path.exists(tsdf_mesh_path):
        try:
            import trimesh
            tsdf_mesh_obj = trimesh.load(tsdf_mesh_path)
            print(f"✓ Found TSDF Marching Cubes mesh: {tsdf_mesh_path}")
        except Exception:
            pass

    markdown_info = None

    # -------------------------------------------------------------
    # Render Function
    # -------------------------------------------------------------
    def render_current_frame():
        t = state['current_t']
        if t not in sequence_cache:
            return

        frame_data = sequence_cache[t]
        xyz = frame_data['xyz']
        rgb = frame_data['rgb']
        scale = frame_data['scale']
        obj_id = frame_data['obj_id']

        N = len(xyz)
        if N == 0:
            return

        # Apply semantic object filtering mask
        if len(obj_id) == N:
            keep_mask = np.array([state['active_objects'].get(int(oid), True) for oid in obj_id], dtype=bool)
            filtered_xyz = xyz[keep_mask]
            filtered_rgb = rgb[keep_mask]
            filtered_scale = scale[keep_mask]
            filtered_obj_id = obj_id[keep_mask]
        else:
            filtered_xyz, filtered_rgb, filtered_scale, filtered_obj_id = xyz, rgb, scale, obj_id

        if len(filtered_xyz) == 0:
            return

        # Determine colors based on selected mode
        mode = state['color_mode']
        if mode == "RGB (Photorealistic)":
            colors = filtered_rgb
        elif mode == "Semantic Segmentation (Object ID)":
            colors = np.zeros_like(filtered_xyz)
            for oid in np.unique(filtered_obj_id):
                c = palette.get(int(oid), np.array([0.7, 0.7, 0.7], dtype=np.float32))
                colors[filtered_obj_id == oid] = c
        elif mode == "Height Map (Z-axis)":
            z_vals = filtered_xyz[:, 2]
            z_min, z_max = np.percentile(z_vals, 2), np.percentile(z_vals, 98)
            z_norm = np.clip((z_vals - z_min) / max(1e-5, (z_max - z_min)), 0.0, 1.0)
            # Turbomap / Plasma approximation
            colors = np.stack([
                np.clip(1.5 * z_norm, 0.0, 1.0),
                np.clip(1.0 - np.abs(z_norm - 0.5) * 2.0, 0.0, 1.0),
                np.clip(1.5 * (1.0 - z_norm), 0.0, 1.0)
            ], axis=1)
        else:
            colors = np.ones_like(filtered_xyz) * 0.85

        # Render either as Gaussian Splats or Fast Point Cloud
        if state['render_mode'] == "Gaussian Splats (3DGS)":
            try:
                # Diagonal covariance from scales
                covariances = np.zeros((len(filtered_xyz), 3, 3), dtype=np.float32)
                covariances[:, 0, 0] = np.square(np.maximum(filtered_scale[:, 0], 1e-4))
                covariances[:, 1, 1] = np.square(np.maximum(filtered_scale[:, 1], 1e-4))
                covariances[:, 2, 2] = np.square(np.maximum(filtered_scale[:, 2], 1e-4))
                opacities = np.full((len(filtered_xyz), 1), 0.95, dtype=np.float32)

                server.scene.add_gaussian_splats(
                    name="/scene/gaussians",
                    centers=filtered_xyz,
                    covariances=covariances,
                    rgbs=colors,
                    opacities=opacities,
                    scale=state['point_size'] / 0.005
                )
            except Exception:
                server.scene.add_point_cloud(
                    name="/scene/gaussians",
                    points=filtered_xyz,
                    colors=colors,
                    point_size=state['point_size'],
                    point_shape="circle"
                )
        else:
            server.scene.add_point_cloud(
                name="/scene/gaussians",
                points=filtered_xyz,
                colors=colors,
                point_size=state['point_size'],
                point_shape="circle"
            )

        # Update info text
        if markdown_info is not None:
            unique_active_ids = np.unique(filtered_obj_id).tolist() if len(filtered_obj_id) > 0 else []
            markdown_info.content = (
                f"### 📍 Timestep `{t}` / `{timesteps[-1][0]}`\n"
                f"- **Gaussians Count**: `{len(filtered_xyz):,}` / `{N:,}`\n"
                f"- **Visible Objects**: `{unique_active_ids}`\n"
                f"- **Color Mode**: `{mode}`"
            )

    # -------------------------------------------------------------
    # GUI Layout
    # -------------------------------------------------------------
    with server.gui.add_folder("⏱️ Playback Controls"):
        t_indices = [t[0] for t in timesteps]
        timestep_slider = server.gui.add_slider(
            "Timestep",
            min=t_indices[0],
            max=t_indices[-1],
            step=1,
            initial_value=t_indices[0]
        )
        
        play_btn = server.gui.add_button("▶ Play / ⏸ Pause")
        btn_prev = server.gui.add_button("◀ Step Prev")
        btn_next = server.gui.add_button("Step Next ▶")
        
        fps_slider = server.gui.add_slider("Playback FPS", min=1, max=25, step=1, initial_value=state['fps'])
        loop_cb = server.gui.add_checkbox("Loop Sequence", initial_value=True)

    with server.gui.add_folder("🎨 Appearance & Color"):
        color_dropdown = server.gui.add_dropdown(
            "Color Mode",
            ("RGB (Photorealistic)", "Semantic Segmentation (Object ID)", "Height Map (Z-axis)", "Solid White"),
            initial_value=state['color_mode']
        )
        render_dropdown = server.gui.add_dropdown(
            "Render Type",
            ("Point Cloud (Fast)", "Gaussian Splats (3DGS)"),
            initial_value=state['render_mode']
        )
        point_size_slider = server.gui.add_slider(
            "Particle Radius",
            min=0.001,
            max=0.025,
            step=0.001,
            initial_value=state['point_size']
        )

    with server.gui.add_folder("🏷️ Semantic Object Filters"):
        obj_checkboxes = {}
        for oid in sorted_obj_ids:
            col_rgb = palette.get(oid, np.array([0.5, 0.5, 0.5]))
            col_hex = f"#{int(col_rgb[0]*255):02x}{int(col_rgb[1]*255):02x}{int(col_rgb[2]*255):02x}"
            label_name = f"Object ID {oid}" if oid != 0 else "Object 0 (Table/Background)"
            cb = server.gui.add_checkbox(f"{label_name}", initial_value=True)
            obj_checkboxes[oid] = cb

        btn_show_all = server.gui.add_button("Show All Objects")
        btn_hide_bg = server.gui.add_button("Isolate Target Objects (Hide ID 0)")

    with server.gui.add_folder("📐 Overlays & Camera Presets"):
        bounds_cb = server.gui.add_checkbox("Workspace Bounding Box", initial_value=True)
        mesh_cb = server.gui.add_checkbox("TSDF Marching Cubes Mesh", initial_value=False, disabled=(tsdf_mesh_obj is None))
        
        btn_iso = server.gui.add_button("📷 Isometric 3D View")
        btn_top = server.gui.add_button("📷 Top-Down View")
        btn_front = server.gui.add_button("📷 Front View")

    markdown_info = server.gui.add_markdown(
        f"### 📍 Timestep `{t_indices[0]}` / `{t_indices[-1]}`\n"
        f"- **Gaussians Count**: `0`\n"
        f"- **Dataset**: `{os.path.basename(data_dir)}`"
    )

    # -------------------------------------------------------------
    # Overlays in 3D Scene
    # -------------------------------------------------------------
    # Coordinate grid
    server.scene.add_grid("/environment/grid", width=2.0, height=2.0, plane="xy", cell_size=0.1)

    # Workspace bounding box [0.1, 0.65] x [-0.45, 0.45] x [0.005, 0.35]
    table_center = ((0.1 + 0.65) / 2.0, (-0.45 + 0.45) / 2.0, (0.005 + 0.35) / 2.0 + 0.75)
    table_dims = (0.65 - 0.1, 0.45 - (-0.45), 0.35 - 0.005)
    
    workspace_box_handle = server.scene.add_box(
        name="/environment/workspace_box",
        position=table_center,
        dimensions=table_dims,
        wireframe=True,
        color=(60, 140, 220),
        opacity=0.45,
        visible=state['show_workspace_bounds']
    )

    # TSDF Mesh overlay
    tsdf_mesh_handle = None
    if tsdf_mesh_obj is not None:
        tsdf_mesh_handle = server.scene.add_mesh_trimesh(
            name="/environment/tsdf_mesh",
            mesh=tsdf_mesh_obj,
            visible=state['show_tsdf_mesh']
        )

    # -------------------------------------------------------------
    # Callbacks & Event Handlers
    # -------------------------------------------------------------
    @timestep_slider.on_update
    def _(_) -> None:
        state['current_t'] = timestep_slider.value
        render_current_frame()

    @play_btn.on_click
    def _(_) -> None:
        state['is_playing'] = not state['is_playing']

    @btn_prev.on_click
    def _(_) -> None:
        curr = state['current_t']
        idx = t_indices.index(curr) if curr in t_indices else 0
        new_idx = max(0, idx - 1)
        timestep_slider.value = t_indices[new_idx]

    @btn_next.on_click
    def _(_) -> None:
        curr = state['current_t']
        idx = t_indices.index(curr) if curr in t_indices else 0
        new_idx = min(len(t_indices) - 1, idx + 1)
        timestep_slider.value = t_indices[new_idx]

    @fps_slider.on_update
    def _(_) -> None:
        state['fps'] = fps_slider.value

    @loop_cb.on_update
    def _(_) -> None:
        state['loop'] = loop_cb.value

    @color_dropdown.on_update
    def _(_) -> None:
        state['color_mode'] = color_dropdown.value
        render_current_frame()

    @render_dropdown.on_update
    def _(_) -> None:
        state['render_mode'] = render_dropdown.value
        render_current_frame()

    @point_size_slider.on_update
    def _(_) -> None:
        state['point_size'] = point_size_slider.value
        render_current_frame()

    @bounds_cb.on_update
    def _(_) -> None:
        workspace_box_handle.visible = bounds_cb.value

    @mesh_cb.on_update
    def _(_) -> None:
        if tsdf_mesh_handle is not None:
            tsdf_mesh_handle.visible = mesh_cb.value

    for oid, cb in obj_checkboxes.items():
        def make_cb_handler(target_oid=oid, target_cb=cb):
            @target_cb.on_update
            def _(_) -> None:
                state['active_objects'][target_oid] = target_cb.value
                render_current_frame()
        make_cb_handler(oid, cb)

    @btn_show_all.on_click
    def _(_) -> None:
        for oid, cb in obj_checkboxes.items():
            cb.value = True
            state['active_objects'][oid] = True
        render_current_frame()

    @btn_hide_bg.on_click
    def _(_) -> None:
        if 0 in obj_checkboxes:
            obj_checkboxes[0].value = False
            state['active_objects'][0] = False
        for oid, cb in obj_checkboxes.items():
            if oid != 0:
                cb.value = True
                state['active_objects'][oid] = True
        render_current_frame()

    # Camera Preset Handlers
    @btn_iso.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is not None:
            client.camera.position = (1.1, -1.1, 1.4)
            client.camera.look_at = (0.35, 0.0, 0.8)

    @btn_top.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is not None:
            client.camera.position = (0.35, 0.0, 2.0)
            client.camera.look_at = (0.35, 0.0, 0.8)

    @btn_front.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is not None:
            client.camera.position = (1.4, 0.0, 1.0)
            client.camera.look_at = (0.35, 0.0, 0.8)

    # Initial frame render
    render_current_frame()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        render_current_frame()

    # Background Playback Thread
    def playback_loop():
        while True:
            if state['is_playing']:
                curr = state['current_t']
                idx = t_indices.index(curr) if curr in t_indices else 0
                if idx < len(t_indices) - 1:
                    new_idx = idx + 1
                    timestep_slider.value = t_indices[new_idx]
                else:
                    if state['loop']:
                        timestep_slider.value = t_indices[0]
                    else:
                        state['is_playing'] = False
                time.sleep(1.0 / max(1, state['fps']))
            else:
                time.sleep(0.05)

    thread = threading.Thread(target=playback_loop, daemon=True)
    thread.start()

    print(f"\n✓ Interactive 3D Viser Visualizer running at: http://localhost:{port}")
    print("Press Ctrl+C in terminal or close process to stop.")

    if block:
        try:
            server.sleep_forever()
        except KeyboardInterrupt:
            print("\nShutting down Viser server...")
            server.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Launch interactive 3D Viser Web Visualizer for Dynamic Gaussian Splatting")
    parser.add_argument("--data_dir", type=str, default="./output_dynamic_mapping",
                        help="Path to directory containing gaussians/ folder or sequence_manifest.json")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host binding (default: 0.0.0.0)")
    parser.add_argument("--point_size", type=float, default=0.006, help="Initial particle radius in meters")
    parser.add_argument("--share", action="store_true", help="Request a public shareable URL from share.viser.studio")
    return parser.parse_args()


def main():
    args = parse_args()
    launch_viser_server(
        data_dir=args.data_dir,
        port=args.port,
        host=args.host,
        point_size=args.point_size,
        share=args.share,
        block=True
    )


if __name__ == "__main__":
    main()
