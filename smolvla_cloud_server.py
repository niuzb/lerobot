"""Run the trained SmolVLA over the internal worker network and SSH relay."""

import argparse
import os
from concurrent import futures
from dataclasses import replace
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"
)

import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options

CHECKPOINT = (
    "/root/code/lerobot/outputs/train/smolvla_tape_curated_20261002/checkpoints/020000/pretrained_model"
)


class SmolVLAServer(PolicyServer):
    """Serve one explicitly selected SmolVLA checkpoint with native camera aspect ratio."""

    @property
    def policy_image_features(self):
        """Keep the two camera streams at their recorded resolution."""
        # Preserve the training images' aspect ratio. SmolVLA performs its own
        # resize-with-padding to 512x512; the generic server would stretch to 256.
        return {
            key: replace(feature, shape=(3, 480, 640))
            for key, feature in self.policy.config.image_features.items()
        }

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Accept only the model selected when this server was started."""
        import pickle

        specs = pickle.loads(request.data)
        if specs.policy_type != "smolvla" or specs.pretrained_name_or_path != CHECKPOINT:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unexpected model for this dedicated service")
        result = super().SendPolicyInstructions(request, context)
        self.policy.eval()
        self.policy.reset()
        return result

    @torch.inference_mode()
    def _predict_action_chunk(self, observation_t):
        return super()._predict_action_chunk(observation_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT,
        help="Exact native pretrained_model directory also sent by the client",
    )
    parser.add_argument("--port", type=int, default=50053)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not Path(args.checkpoint).is_dir():
        parser.error("--checkpoint must be an existing pretrained_model directory")
    CHECKPOINT = args.checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    service = SmolVLAServer(
        PolicyServerConfig(host="[::]", port=args.port, fps=30, inference_latency=0, obs_queue_timeout=5)
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=grpc_channel_options())
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    if not server.add_insecure_port(f"[::]:{args.port}"):
        raise RuntimeError(f"Port {args.port} unavailable")
    server.start()
    print(f"SMOLVLA_SERVER_READY port={args.port} checkpoint={CHECKPOINT}", flush=True)
    server.wait_for_termination()
