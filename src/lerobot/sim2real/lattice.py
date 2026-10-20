"""Rigid diamond lattice with open visual and collision holes."""

import math
import xml.etree.ElementTree as ET

import numpy as np


def validate_lattice(holder):
    """Bound geometry complexity and keep real openings between strands."""
    mesh = holder["mesh"]
    count, segments = mesh["strands_per_direction"], mesh["segments_per_strand"]
    if not isinstance(count, int) or not 8 <= count <= 96:
        raise ValueError("Mesh strands_per_direction must be an integer in 8..96")
    if not isinstance(segments, int) or not 4 <= segments <= 64 or count * segments > 4096:
        raise ValueError("Mesh segments_per_strand exceeds supported complexity")
    width, slope, rim = (mesh[k] for k in ("strand_width_m", "circumferential_slope", "rim_height_m"))
    if not np.isfinite([width, slope, rim]).all() or min(width, slope, rim) <= 0:
        raise ValueError("Mesh width, slope and rim height must be finite and positive")
    if width >= math.pi * holder["inner_radius_m"] / count:
        raise ValueError("Mesh strand width closes the intended holes")
    if slope > 2 or 2 * rim + holder["bottom_m"] >= holder["height_m"]:
        raise ValueError("Invalid lattice winding or rim dimensions")


def lattice_segments(holder):
    """Two crossing helical ribbon families, discretized into convex prisms."""
    validate_lattice(holder)
    mesh = holder["mesh"]
    radius = holder["inner_radius_m"] + holder["wall_m"] / 2
    low = holder["bottom_m"] + mesh["rim_height_m"]
    high = holder["height_m"] - mesh["rim_height_m"]
    for direction in (-1, 1):
        for start in range(mesh["strands_per_direction"]):
            phase = 2 * math.pi * start / mesh["strands_per_direction"]
            for z0, z1 in zip(
                np.linspace(low, high, mesh["segments_per_strand"] + 1)[:-1],
                np.linspace(low, high, mesh["segments_per_strand"] + 1)[1:],
                strict=True,
            ):
                theta = (
                    phase + direction * mesh["circumferential_slope"] * (np.array([z0, z1]) - low) / radius
                )
                middle = theta.mean()
                radial = np.array([math.cos(middle), math.sin(middle), 0])
                endpoints = np.c_[radius * np.cos(theta), radius * np.sin(theta), [z0, z1]]
                along = endpoints[1] - endpoints[0]
                across = np.cross(radial, along)
                across /= np.linalg.norm(across)
                vertices = [
                    point + r * radial * holder["wall_m"] / 2 + w * across * mesh["strand_width_m"] / 2
                    for point in endpoints
                    for r in (-1, 1)
                    for w in (-1, 1)
                ]
                yield np.asarray(vertices)


def add_lattice_wall(root, body, holder, mass, rgba="0.045 0.05 0.055 1"):
    """Use the same perforated ribbon geometry for rendering and collision."""
    mesh = holder["mesh"]
    count = 2 * mesh["strands_per_direction"] * mesh["segments_per_strand"]
    for i, vertices in enumerate(lattice_segments(holder)):
        name = f"holder_wall_strand_{i}"
        ET.SubElement(root.find("asset"), "mesh", name=name, vertex=" ".join(map(str, vertices.ravel())))
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            rgba=rgba,
            mass=str(mass / count),
            friction="1 0.005 0.0001",
            condim="4",
        )
