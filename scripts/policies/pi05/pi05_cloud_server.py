"""Dedicated final Pi05 server; camera preprocessing matches training."""
import os
from concurrent import futures
from dataclasses import replace
import pickle

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')

import grpc
import torch
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options

CHECKPOINT = '/root/code/lerobot/outputs/train/pi05_tape_curated_20261004/checkpoints/080000/pretrained_model'


class Pi05Server(PolicyServer):
    @property
    def policy_image_features(self):
        # Pi05 pads/resizes images internally; retain the camera aspect ratio.
        return {key: replace(feature, shape=(3, 480, 640))
                for key, feature in self.policy.config.image_features.items()
                if 'empty_camera' not in key}

    def SendPolicyInstructions(self, request, context):
        specs = pickle.loads(request.data)
        if specs.policy_type != 'pi05' or specs.pretrained_name_or_path != CHECKPOINT:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Unexpected model for dedicated Pi05 service')
        result = super().SendPolicyInstructions(request, context)
        self.policy.eval()
        self.policy.reset()
        return result

    @torch.inference_mode()
    def _predict_action_chunk(self, observation_t):
        return super()._predict_action_chunk(observation_t)


class WarmupContext:
    def peer(self):
        return 'local-model-preload'

    def abort(self, code, details):
        raise RuntimeError(details)


if __name__ == '__main__':
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    print('GPU', torch.cuda.get_device_name(0), flush=True)
    service = Pi05Server(PolicyServerConfig(host='[::]', port=50055, fps=30,
                                          inference_latency=0, obs_queue_timeout=5))
    specs = RemotePolicyConfig('pi05', CHECKPOINT, {}, 30, 'cuda')
    service.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(specs)), WarmupContext())
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=grpc_channel_options())
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    assert server.add_insecure_port('[::]:50055')
    server.start()
    print('PI05_SERVER_READY port=50055', flush=True)
    server.wait_for_termination()
