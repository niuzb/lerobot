"""Reproducible per-recorded-frame texture changes, independent of physics."""

import colorsys
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from common import sha256, write_json

GROUPS = ("table", "background", "tape", "holder", "robot")
PATTERNS = ("checker", "wood_grain", "woven", "speckle", "stripes", "tiles")


def texture_pixels(seed, index, size=128):
    """Create actual spatial patterns, not uniform colour swatches."""
    if size < 16:
        raise ValueError("Texture dimensions must be at least 16 pixels")
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    y, x = np.mgrid[:size, :size] / size
    frequency = int(rng.integers(4, 13))
    phase = rng.uniform(0, 2 * np.pi)
    mode = index % len(PATTERNS)
    if mode == 0:
        value = (np.floor(x * frequency) + np.floor(y * frequency)) % 2
    elif mode == 1:
        value = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * x + 2 * np.sin(y * 15 + phase))
    elif mode == 2:
        value = 0.5 + 0.25 * np.sin(x * frequency * 2 * np.pi) + 0.25 * np.cos(y * frequency * 2 * np.pi)
    elif mode == 3:
        coarse = rng.uniform(0, 1, (16, 16))
        indices = np.minimum(np.arange(size) * 16 // size, 15)
        value = 0.7 * coarse[indices[:, None], indices[None, :]] + 0.3 * rng.uniform(0, 1, (size, size))
    elif mode == 4:
        value = 0.5 + 0.5 * np.sin((x + y * rng.uniform(-1, 1)) * frequency * 2 * np.pi + phase)
    else:
        value = np.minimum((x * frequency) % 1, (y * frequency) % 1) > 0.12
    hue = rng.uniform(0, 1)
    a = np.array(colorsys.hsv_to_rgb(hue, rng.uniform(0.2, 0.8), rng.uniform(0.25, 0.55)))
    b = np.array(
        colorsys.hsv_to_rgb((hue + rng.uniform(0.2, 0.6)) % 1, rng.uniform(0.2, 0.8), rng.uniform(0.65, 0.95))
    )
    pixels = np.clip((a + np.asarray(value)[..., None] * (b - a)) * 255, 0, 255).astype(np.uint8)
    if pixels.reshape(-1, 3).std(axis=0).max() < 5:
        raise ValueError("Procedural texture is insufficiently patterned")
    return pixels


def write_png(path, pixels):
    """Write lossless RGB PNG images without adding a simulation dependency."""
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Require uint8 RGB pixels")
    height, width, _ = pixels.shape

    def chunk(name, data):
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\0" + row.tobytes() for row in pixels)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def create_bank(folder, seed, count=16, size=128):
    if count < 2:
        raise ValueError("At least two textures are required for framewise switching")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    groups = {}
    for number, group in enumerate(GROUPS):
        target = folder / group
        target.mkdir()
        entries = []
        for index in range(count):
            pixels = texture_pixels(seed + 1000 * number, index, size)
            path = target / f"texture_{index:03d}.png"
            write_png(path, pixels)
            entries.append(
                {
                    "id": index,
                    "file": str(path.relative_to(folder)),
                    "sha256": sha256(path),
                    "pattern": PATTERNS[index % len(PATTERNS)],
                    "pixel_std": float(pixels.astype(float).std(axis=(0, 1)).max()),
                }
            )
        groups[group] = entries
    report = {
        "seed": seed,
        "count_per_group": count,
        "size": size,
        "groups": groups,
        "generator": "procedural RGB patterns; no downloaded images",
        "frequency": "per_recorded_frame",
    }
    write_json(folder / "manifest.json", report)
    return report


class TextureSchedule:
    """Every group changes at every frame; both cameras share the same row."""

    def __init__(self, seed, episode, count):
        if count < 2:
            raise ValueError("Framewise randomization requires at least two texture IDs")
        self.count = count
        self.rng = {
            group: np.random.default_rng(np.random.SeedSequence([seed, episode, i]))
            for i, group in enumerate(GROUPS)
        }
        self.previous = dict.fromkeys(GROUPS, -1)
        self.frame = 0

    def next(self, source_frame):
        values = {}
        for group in GROUPS:
            previous = self.previous[group]
            values[group] = (
                int(self.rng[group].integers(self.count))
                if previous < 0
                else (previous + 1 + int(self.rng[group].integers(self.count - 1))) % self.count
            )
        self.previous = values
        result = {"frame_index": self.frame, "source_frame_index": int(source_frame), "texture_ids": values}
        self.frame += 1
        return result


def validate_schedule(rows, frame_count, bank):
    if len(rows) != frame_count:
        raise ValueError("Appearance metadata is missing frames")
    previous = None
    previous_rgb = None
    for index, row in enumerate(rows):
        if row["frame_index"] != index or row["source_frame_index"] != index:
            raise ValueError("Appearance frame indices differ from full-rate physical records")
        ids = row["texture_ids"]
        if set(ids) != set(GROUPS):
            raise ValueError("An appearance group is missing")
        for group in GROUPS:
            if not isinstance(ids[group], int) or not 0 <= ids[group] < len(bank["groups"][group]):
                raise ValueError("Invalid texture ID")
            if previous is not None and ids[group] == previous[group]:
                raise ValueError(f"Texture did not change at frame {index}: {group}")
        previous = ids
        captured = row.get("capture_rgb_sha256")
        if captured is not None:
            if set(captured) != {"front", "side"}:
                raise ValueError("Require paired RGB capture hashes")
            if previous_rgb is not None and any(captured[name] == previous_rgb[name] for name in captured):
                raise ValueError("A camera repeated an identical image despite framewise texture changes")
            previous_rgb = captured


def verified_appearance(render, episode):
    """Read complete per-frame records and verify their immutable texture bank."""
    render = Path(render)
    manifest = json.loads((render / "render_manifest.json").read_text())
    profile = manifest.get("appearance")
    if profile is None:
        return None
    if profile["frequency"] != "per_recorded_frame" or not profile.get("probe", {}).get("passed"):
        raise ValueError("Require verified framewise texture rendering")
    if not profile["probe"].get("physics_properties_unchanged"):
        raise ValueError("Appearance rendering lacks physics-preservation evidence")
    folder = render / "texture_bank"
    if sha256(folder / "manifest.json") != profile["bank_manifest_sha256"]:
        raise ValueError("Texture bank manifest changed")
    bank = json.loads((folder / "manifest.json").read_text())
    for entries in bank["groups"].values():
        for entry in entries:
            path = (folder / entry["file"]).resolve()
            if folder.resolve() not in path.parents or sha256(path) != entry["sha256"]:
                raise ValueError("Texture image changed or left the texture-bank directory")
    path = render / f"attempt_{episode['attempt']:04d}" / "appearance.jsonl"
    if episode.get("appearance_frames") != episode["frames"] or sha256(path) != episode["files_sha256"].get(
        "appearance.jsonl"
    ):
        raise ValueError("Per-frame texture records changed or are incomplete")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    validate_schedule(rows, episode["frames"], bank)
    return rows


def physics_fingerprint(stage):
    """Hash authored collision/physical properties and physics-purpose bindings."""
    from pxr import UsdPhysics

    values = {}
    for prim in stage.TraverseAll():
        attributes = {
            attribute.GetName(): repr(attribute.Get())
            for attribute in prim.GetAttributes()
            if attribute.GetName().startswith(("physics:", "physx"))
        }
        relationships = {
            relationship.GetName(): [str(path) for path in relationship.GetTargets()]
            for relationship in prim.GetRelationships()
            if relationship.GetName().startswith(("physics:", "material:binding:physics"))
        }
        if attributes or relationships or prim.HasAPI(UsdPhysics.CollisionAPI):
            values[str(prim.GetPath())] = {
                "attributes": attributes,
                "relationships": relationships,
                "collider": prim.HasAPI(UsdPhysics.CollisionAPI),
            }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class FrameTextures:
    """Bind USD texture graphs without editing source assets or physical poses."""

    def __init__(self, stage, config, folder, seed, count=16):
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        self.stage, self.folder = stage, Path(folder)
        self.seed, self.count = seed, count
        self.before_physics = physics_fingerprint(stage)
        self.bank = create_bank(self.folder, seed, count)
        self.inputs = {}
        self.uv_meshes = {}
        root = "/World/envs/env_0"
        height = config.get("spatial", {}).get("table_top_height_m", 0.0)

        def panel(path, points, repeat):
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
            ).Set([Gf.Vec2f(0, 0), Gf.Vec2f(repeat, 0), Gf.Vec2f(repeat, repeat), Gf.Vec2f(0, repeat)])
            return mesh.GetPrim()

        x, y = config["table"]["half_x_m"], config["table"]["half_y_m"]
        top = height + 0.00002
        table = panel(root + "/TextureTableTop", [(-x, -y, top), (x, -y, top), (x, y, top), (-x, y, top)], 3)
        background = panel(
            root + "/TextureBackdrop",
            [(-0.16, -1, height), (-0.16, 1, height), (-0.16, 1, height + 0.9), (-0.16, -1, height + 0.9)],
            3,
        )
        targets = {
            "table": table,
            "background": background,
            "tape": stage.GetPrimAtPath(root + "/Tape"),
            "holder": stage.GetPrimAtPath(root + "/Holder"),
            "robot": stage.GetPrimAtPath(root + "/Robot"),
        }
        for group, target in targets.items():
            self.uv_meshes[group] = self._uvs(target, group, config)
            material = UsdShade.Material.Define(stage, f"/World/FrameTextures/{group}")
            surface = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Surface"))
            surface.CreateIdAttr("UsdPreviewSurface")
            surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
            surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            reader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("UV"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            texture = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Texture"))
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 0, 1, 1))
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            self.inputs[group] = texture.CreateInput("file", Sdf.ValueTypeNames.Asset)
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                texture.ConnectableAPI(), "rgb"
            )
            surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(target).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        self.apply(dict.fromkeys(GROUPS, 0))
        self.assert_physics_unchanged()

    def _uvs(self, root, group, config):
        from pxr import Gf, Sdf, Usd, UsdGeom

        count = 0
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            existing = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
            if existing and existing.GetAttr().HasAuthoredValueOpinion():
                count += 1
                continue
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            if not len(points):
                continue
            if group == "tape":
                uv = (points[:, :2] / (2 * config["tape"]["outer_radius_m"]) + 0.5) * 2
            elif group == "holder":
                uv = (
                    np.c_[
                        (np.arctan2(points[:, 1], points[:, 0]) + np.pi) / (2 * np.pi),
                        points[:, 2] / config["holder"]["height_m"],
                    ]
                    * 3
                )
            else:
                axes = np.argsort(np.ptp(points, axis=0))[-2:]
                uv = (
                    (points[:, axes] - points[:, axes].min(axis=0))
                    / np.maximum(np.ptp(points[:, axes], axis=0), 1e-8)
                    * 2
                )
            UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
            ).Set([Gf.Vec2f(float(u), float(v)) for u, v in uv])
            count += 1
        return count

    def apply(self, ids):
        from pxr import Sdf

        for group in GROUPS:
            path = self.folder / self.bank["groups"][group][ids[group]]["file"]
            self.inputs[group].Set(Sdf.AssetPath(str(path)))
            if self.inputs[group].Get().path != str(path):
                raise RuntimeError("Texture binding did not match the frame schedule")

    def assert_physics_unchanged(self):
        if physics_fingerprint(self.stage) != self.before_physics:
            raise RuntimeError("Appearance randomization modified physical properties")

    def describe(self):
        return {
            "frequency": "per_recorded_frame",
            "groups": list(GROUPS),
            "seed": self.seed,
            "bank_manifest_sha256": sha256(self.folder / "manifest.json"),
            "count_per_group": self.count,
            "uv_meshes": self.uv_meshes,
            "physics_properties_sha256": self.before_physics,
            "both_cameras_share_appearance": True,
            "changes_policy_state": False,
        }
