"""Run the trained SmolVLA over the internal worker network and SSH relay."""
import os
from concurrent import futures
from dataclasses import replace

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")

import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options

CHECKPOINT = "/root/code/lerobot/outputs/train/smolvla_tape_curated_20261002/checkpoints/020000/pretrained_model"


class SmolVLAServer(PolicyServer):
    @property
    def policy_image_features(self):
        # Preserve the training images' aspect ratio. SmolVLA performs its own
        # resize-with-padding to 512x512; the generic server would stretch to 256.
        return {key: replace(feature, shape=(3, 480, 640))
                for key, feature in self.policy.config.image_features.items()}

    def SendPolicyInstructions(self, request, context):
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    service = SmolVLAServer(PolicyServerConfig(host="[::]", port=50053, fps=30,
                                             inference_latency=0, obs_queue_timeout=5))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=grpc_channel_options())
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    if not server.add_insecure_port("[::]:50053"):
        raise RuntimeError("Port 50053 unavailable")
    server.start()
    print("SMOLVLA_SERVER_READY port=50053", flush=True)
    server.wait_for_termination()
