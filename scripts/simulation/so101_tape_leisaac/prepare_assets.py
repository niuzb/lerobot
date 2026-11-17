"""Convert the prior SO101 robot and author physically hollow task objects."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET  # nosec B405 - Only trusted local robot MJCF assets are parsed.
from pathlib import Path

import numpy as np
from common import FACES, lattice_vertices, ring_vertices, sha256, write_json


def materialize_defaults(root):
    """Expand MJCF defaults because the USD importer handles fewer class forms."""
    defaults = {}

    def register(node, inherited):
        values = {key: dict(value) for key, value in inherited.items()}
        for child in node:
            if child.tag != "default":
                values.setdefault(child.tag, {}).update(child.attrib)
        defaults[node.get("class", "main")] = values
        for child in node.findall("default"):
            register(child, values)

    register(root.find("default"), {})

    def visit(node, inherited="main"):
        active = node.get("childclass", inherited) if node.tag == "body" else inherited
        selected = node.get("class", active)
        if node.tag not in {"default", "mujoco", "body"}:
            node.attrib.update({**defaults.get(selected, {}).get(node.tag, {}), **node.attrib})
        node.attrib.pop("class", None)
        node.attrib.pop("childclass", None)
        for child in node:
            if child.tag != "default":
                visit(child, active)

    visit(root)
    root.remove(root.find("default"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "enable_crashreporter": False,
            "limit_cpu_threads": 4,
            "multi_gpu": False,
            "active_gpu": 0,
            "anti_aliasing": 2,
            "extra_args": ["--allow-root", "--/rtx/verifyDriverVersion/enabled=false"],
        }
    )
    try:
        import omni.kit.commands
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        asset = Path(config["asset"])
        tree = ET.parse(asset)  # nosec B314 - Trusted local robot asset, not an uploaded XML document.
        root = tree.getroot()
        materialize_defaults(root)
        mesh_dir = args.output / "source_meshes"
        mesh_dir.mkdir()
        for source in (asset.parent / "assets").glob("*"):
            if source.is_file():
                shutil.copy2(source, mesh_dir / source.name)
        root.find("compiler").set("meshdir", "source_meshes")
        for parent in root.iter():
            for camera in list(parent.findall("camera")):
                parent.remove(camera)
            for geom in list(parent.findall("geom")):
                if geom.get("mesh") == "ArduCam_M12Lens":
                    parent.remove(geom)
        for material in root.findall("asset/material"):
            if material.get("rgba") == "1 0.82 0.12 1":
                material.set("rgba", "0.88 0.88 0.86 1")
        mjcf = args.output / "robot_source.xml"
        tree.write(mjcf, encoding="utf-8")
        enable_extension("isaacsim.asset.importer.mjcf")
        ok, importer = omni.kit.commands.execute("MJCFCreateImportConfig")
        if not ok:
            raise RuntimeError("Cannot create the official MJCF importer")
        importer.merge_fixed_joints = False
        importer.convex_decomp = False
        importer.import_inertia_tensor = True
        importer.fix_base = True
        importer.distance_scale = 1.0
        robot_path = args.output / "robot.usd"
        ok, imported = omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=str(mjcf),
            import_config=importer,
            prim_path="/Robot",
            dest_path=str(robot_path),
        )
        if not ok:
            raise RuntimeError("Official MJCF import failed")
        robot_stage = Usd.Stage.Open(str(robot_path))
        robot_stage.Reload()
        robot_stage.Load()
        for prim in list(robot_stage.TraverseAll()):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in list(robot_stage.TraverseAll()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                prim.RemoveAppliedSchema("PhysxArticulationAPI")
        UsdPhysics.ArticulationRootAPI.Apply(robot_stage.GetDefaultPrim())
        PhysxSchema.PhysxArticulationAPI.Apply(robot_stage.GetDefaultPrim())
        robot_colliders = []
        for prim in robot_stage.TraverseAll():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.0)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(0.00015)
                collision.CreateRestOffsetAttr(0.0)
                robot_colliders.append(str(prim.GetPath()))
        robot_stage.GetRootLayer().Save()

        def create_object(name, mass):
            stage = Usd.Stage.CreateNew(str(args.output / f"{name}.usd"))
            body = UsdGeom.Xform.Define(stage, "/Object")
            stage.SetDefaultPrim(body.GetPrim())
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass_api.CreateMassAttr(mass)
            rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
            rigid.CreateSolverPositionIterationCountAttr(16)
            rigid.CreateSolverVelocityIterationCountAttr(4)
            rigid.CreateMaxDepenetrationVelocityAttr(0.1)
            rigid.CreateEnableCCDAttr(True)
            rigid.CreateSleepThresholdAttr(0.0)
            PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.0)
            material = UsdShade.Material.Define(stage, "/Object/Material")
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr(1.0)
            physics.CreateDynamicFrictionAttr(1.0)
            physics.CreateRestitutionAttr(0.0)
            return stage, body, mass_api, material

        def collision(prim, material):
            UsdPhysics.CollisionAPI.Apply(prim)
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(0.0001)
            api.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        def mesh(stage, path, vertices, material, color):
            geom = UsdGeom.Mesh.Define(stage, path)
            geom.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in vertices])
            geom.CreateFaceVertexCountsAttr([4] * 6)
            geom.CreateFaceVertexIndicesAttr(FACES)
            geom.CreateSubdivisionSchemeAttr("none")
            geom.CreateDoubleSidedAttr(True)
            geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            collision(geom.GetPrim(), material)
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")

        tape = config["tape"]
        stage, body, mass_api, material = create_object("tape", tape["mass_kg"])
        for index, vertices in enumerate(
            ring_vertices(tape["inner_radius_m"], tape["outer_radius_m"], tape["height_m"])
        ):
            mesh(stage, f"/Object/Wedge_{index:03d}", vertices, material, (0.08, 0.08, 0.08))
        axial = tape["mass_kg"] * (tape["outer_radius_m"] ** 2 + tape["inner_radius_m"] ** 2) / 2
        transverse = axial / 2 + tape["mass_kg"] * tape["height_m"] ** 2 / 12
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(transverse, transverse, axial))
        stage.GetRootLayer().Save()

        holder = config["holder"]
        stage, body, mass_api, material = create_object("holder", holder["mass_kg"])
        strand_count = 0
        for index, vertices in enumerate(lattice_vertices(holder)):
            mesh(stage, f"/Object/Strand_{index:04d}", vertices, material, (0.045, 0.05, 0.055))
            strand_count += 1
        rim_height = holder["mesh"]["rim_height_m"]
        for name, z in (
            ("Lower", holder["bottom_m"] + rim_height / 2),
            ("Upper", holder["height_m"] - rim_height / 2),
        ):
            for index, vertices in enumerate(
                ring_vertices(
                    holder["inner_radius_m"], holder["inner_radius_m"] + holder["wall_m"], rim_height
                )
            ):
                mesh(
                    stage,
                    f"/Object/{name}Rim_{index:02d}",
                    vertices + [0, 0, z],
                    material,
                    (0.045, 0.05, 0.055),
                )
        bottom = UsdGeom.Cylinder.Define(stage, "/Object/Bottom")
        bottom.CreateRadiusAttr(holder["inner_radius_m"] + holder["wall_m"])
        bottom.CreateHeightAttr(holder["bottom_m"])
        bottom.CreateAxisAttr("Z")
        bottom.AddTranslateOp().Set(Gf.Vec3d(0, 0, holder["bottom_m"] / 2))
        bottom.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.1, 0.1)])
        collision(bottom.GetPrim(), material)
        # Same estimated 60/20/20 lattice/rims/bottom mass split as the prior experiment.
        mass = holder["mass_kg"]
        centers = np.array(
            [
                (holder["bottom_m"] + holder["height_m"]) / 2,
                holder["bottom_m"] + rim_height / 2,
                holder["height_m"] - rim_height / 2,
                holder["bottom_m"] / 2,
            ]
        )
        weights = mass * np.array([0.6, 0.1, 0.1, 0.2])
        center = np.dot(centers, weights) / mass
        radius = holder["inner_radius_m"] + holder["wall_m"] / 2
        axial = np.dot(weights, [radius**2, radius**2, radius**2, (radius + holder["wall_m"] / 2) ** 2 / 2])
        heights = np.array(
            [holder["height_m"] - holder["bottom_m"], rim_height, rim_height, holder["bottom_m"]]
        )
        transverse = axial / 2 + np.dot(weights, heights**2 / 12 + (centers - center) ** 2)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, center))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(transverse, transverse, axial))
        stage.GetRootLayer().Save()
        write_json(
            args.output / "assets_manifest.json",
            {
                "source_robot": str(asset),
                "source_robot_sha256": sha256(asset),
                "imported_prim": str(imported),
                "robot_default_prim": str(robot_stage.GetDefaultPrim().GetPath()),
                "robot_collision_paths": robot_colliders,
                "tape_convex_wedges": 32,
                "holder_perforated_strands": strand_count,
                "holder_hidden_solid_wall": False,
                "contact_report_threshold_n": 0.0,
                "calibration_passed": False,
                "files_sha256": {
                    str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob("*") if p.is_file()
                },
            },
        )
        print("TASK_ASSETS_PREPARED", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
