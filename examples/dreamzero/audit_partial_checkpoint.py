"""Read tensor metadata and byte extents without executing checkpoint globals or loading tensors."""

import argparse
import json
import pickle
import struct
from pathlib import Path


class Placeholder:
    pass


def rebuild_tensor(storage, offset, shape, stride, *args):
    return {"storage": storage["key"], "offset": offset, "shape": shape, "stride": stride}


def rebuild_type(function, tensor_type, args, state):
    return function(*args)


class MetadataReader(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.storages = {}

    def find_class(self, module, name):
        from collections import OrderedDict

        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
            ("torch._tensor", "_rebuild_from_type_v2"): rebuild_type,
            ("torch", "Tensor"): Placeholder,
            ("torch", "FloatStorage"): "float32",
            ("deepspeed.runtime.zero.config", "ZeroStageEnum"): int,
            ("deepspeed.runtime.fp16.loss_scaler", "LossScaler"): Placeholder,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported metadata global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, identifier):
        kind, dtype, key, location, elements = identifier
        if kind != "storage" or dtype != "float32":
            raise ValueError(f"Unsupported storage metadata: {kind}, {dtype}")
        self.storages[key] = {"key": key, "bytes": elements * 4, "location": location}
        return self.storages[key]


def read_header(stream):
    raw = stream.read(30)
    if len(raw) != 30:
        raise ValueError("Incomplete local ZIP header")
    header = struct.unpack("<IHHHHHIIIHH", raw)
    if header[0] != 0x04034B50 or header[3] != 0:
        raise ValueError("Expected an uncompressed PyTorch ZIP member")
    name = stream.read(header[-2]).decode()
    stream.seek(header[-1], 1)
    return name, stream.tell()


def audit(path):
    total = path.stat().st_size
    with path.open("rb") as stream:
        name, offset = read_header(stream)
        if not name.endswith("/data.pkl"):
            raise ValueError("First member must contain tensor metadata")
        reader = MetadataReader(stream)
        metadata = reader.load()
        metadata_size = stream.tell() - offset
        groups = metadata["optimizer_state_dict"]["fp32_flat_groups"]
        master_keys = {group["storage"] for group in groups}
        sizes = {"data.pkl": metadata_size, ".format_version": 1, ".storage_alignment": 2,
                 "byteorder": 6, "version": 2, ".data/serialization_id": 40}
        complete, partial = set(), []
        stream.seek(0)
        while stream.tell() + 30 <= total:
            name, offset = read_header(stream)
            relative = name.split("/", 1)[1]
            key = relative.removeprefix("data/") if relative.startswith("data/") else None
            size = reader.storages[key]["bytes"] if key is not None else sizes[relative]
            if offset + size > total:
                partial.append({"member": relative, "required_bytes": size,
                                "available_bytes": max(0, total - offset)})
                break
            if key is not None:
                complete.add(key)
            stream.seek(offset + size)
            after = stream.read(512)
            next_header = after.find(b"PK\x03\x04")
            if next_header < 0:
                break
            stream.seek(offset + size + next_header)
    missing = master_keys - complete
    return {"file": str(path), "file_bytes": total, "fp32_master_groups": len(master_keys),
            "complete_master_groups": len(master_keys & complete),
            "missing_or_partial_master_groups": len(missing),
            "missing_master_keys": sorted(missing, key=int), "partial_members": partial,
            "all_master_payloads_present": not missing, "payload_checksums_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for checkpoint_path in args.paths:
        print(json.dumps(audit(checkpoint_path)), flush=True)
