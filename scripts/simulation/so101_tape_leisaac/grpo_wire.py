"""Private local NumPy RPC between Isaac Python3.11 and policy Python3.12."""

import io
import json
import struct

import numpy as np


def receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("Local rollout RPC closed")
        chunks.extend(chunk)
    return bytes(chunks)


def send(connection, metadata, **arrays):
    buffer = io.BytesIO()
    np.savez(buffer, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8), **arrays)
    payload = buffer.getvalue()
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def receive(connection):
    size = struct.unpack("!Q", receive_exact(connection, 8))[0]
    if not 0 < size < 64 * 1024 * 1024:
        raise ValueError("Invalid local rollout packet size")
    with np.load(io.BytesIO(receive_exact(connection, size)), allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes().decode())
        arrays = {k: data[k].copy() for k in data.files if k != "metadata"}
    return metadata, arrays


def request(connection, metadata, **arrays):
    send(connection, metadata, **arrays)
    result, output = receive(connection)
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result, output
