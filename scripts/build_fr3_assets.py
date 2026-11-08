#!/usr/bin/env python
"""
Builds assets/franka_fr3 from Drema2's FR3 models (robot-stack's fr3.urdf and the FR3 MJCF):

- meshes/: the FR3 collision meshes, the Franka Hand and its finger;
- fr3_twin.urdf: PyBullet twin model, link0 as root so the 7 arm joints come first (the twin maps PyBullet link i to
  arm link i+1); visuals are the collision meshes (PyBullet does not load the .dae visuals);
- fr3.xml: MuJoCo twin model, Drema2's MJCF with local mesh paths;
- fr3_mask.urdf: robot silhouette for the lab cameras, every visual replaced by the collision geometry.

    python scripts/build_fr3_assets.py <Drema2 checkout> [assets/franka_fr3]
"""

import copy
import os
import shutil
import sys
import xml.etree.ElementTree as ET


def local_mesh(path: str) -> str:
    rel = path.split("/franka_urdf/meshes/")[1]
    return f"meshes/{'hand' if 'robot_ee' in rel else 'arm'}_{os.path.basename(rel)}"


def urdf_with_collision_visuals(src: str, dst_dir: str, drop_world_base: bool) -> ET.ElementTree:
    """The URDF with local collision meshes, each visual replaced by the collision geometry."""
    tree = ET.parse(src)
    root = tree.getroot()
    if drop_world_base:
        root.remove(root.find("link[@name='base']"))
        root.remove(root.find("joint[@name='fr3_base_joint']"))
    for link in root.findall("link"):
        for vis in link.findall("visual"):
            link.remove(vis)
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is not None:
                rel = mesh.get("filename").split("/franka_urdf/meshes/")[1]
                name = local_mesh(mesh.get("filename"))
                shutil.copy(os.path.join(os.path.dirname(src), "meshes", rel), os.path.join(dst_dir, name))
                mesh.set("filename", name)
            vis = copy.deepcopy(col)
            vis.tag = "visual"
            vis.attrib.pop("name", None)
            link.insert(0, vis)
        if link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0.01")
            ET.SubElement(inertial, "inertia", ixx="1e-6", ixy="0", ixz="0", iyy="1e-6", iyz="0", izz="1e-6")
    return tree


def main(drema2: str, out: str) -> None:
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)
    urdf = os.path.join(drema2, "assets", "franka_urdf", "fr3.urdf")
    urdf_with_collision_visuals(urdf, out, drop_world_base=False).write(
        os.path.join(out, "fr3_mask.urdf"), xml_declaration=True, encoding="utf-8")
    urdf_with_collision_visuals(urdf, out, drop_world_base=True).write(
        os.path.join(out, "fr3_twin.urdf"), xml_declaration=True, encoding="utf-8")

    mjcf = os.path.join(drema2, "assets", "mujoco_fr3", "fr3_gripper.xml")
    tree = ET.parse(mjcf)
    for mesh in tree.getroot().find("asset").findall("mesh"):
        name = os.path.basename(mesh.get("file"))
        local = {"hand.stl": "hand_hand.stl"}.get(name, f"arm_{name}" if name.startswith("link") else name)
        if not os.path.exists(os.path.join(out, "meshes", local)):
            shutil.copy(os.path.realpath(os.path.join(os.path.dirname(mjcf), "meshes", name)),
                        os.path.join(out, "meshes", local))
        mesh.set("file", local)
    tree.write(os.path.join(out, "fr3.xml"), encoding="utf-8")
    print(f"FR3 assets written to {out}: {sorted(os.listdir(out))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "assets/franka_fr3")
