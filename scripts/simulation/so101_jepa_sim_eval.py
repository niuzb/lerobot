"""SO-101 MuJoCo closed-loop smoke evaluation. Never opens physical hardware."""
import argparse
from contextlib import nullcontext
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
INITIAL = np.array([8.08791256, -97.27472687, 96.43955994, 47.38461685, -0.92307693, 54.92537308])


def make_scene(source, target):
    tree = ET.parse(source)
    root = tree.getroot()
    root.find('compiler').set('meshdir', str((source.parent / 'assets').resolve()))
    root.find('option').set('timestep', str(1 / 300))
    visual = root.find('visual')
    ET.SubElement(visual, 'global', offwidth='640', offheight='480')
    world = root.find('worldbody')
    # The vendor lens mesh intersects its optical origin; omit only that visual mesh.
    mount = world.find(".//body[@name='camera_mount']")
    for geom in list(mount.findall('geom')):
        if geom.get('mesh') == 'ArduCam_M12Lens':
            mount.remove(geom)
    ET.SubElement(world, 'light', pos='0 -0.3 1.5', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(world, 'geom', name='table', type='plane', size='1 1 0.02', rgba='0.75 0.7 0.6 1')
    # Camera looks at the table from the front. Wrist camera comes with the robot asset.
    pos = np.array([0.45, -0.55, 0.42])
    z = pos - np.array([0, -0.18, 0.10]); z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, 'camera', name='side', pos=' '.join(map(str, pos)),
                  xyaxes=' '.join(map(str, np.r_[x, y])), fovy='55')
    obj = ET.SubElement(world, 'body', name='tape_proxy', pos='0 -0.23 0.015')
    ET.SubElement(obj, 'freejoint', name='object_free')
    ET.SubElement(obj, 'geom', name='tape', type='cylinder', size='0.025 0.012',
                  mass='0.025', rgba='0.8 0.65 0.22 1', friction='1 0.01 0.001')
    # Open square container approximates a pen holder; no attachment or scripted grasp.
    cup = ET.SubElement(world, 'body', name='holder', pos='0.13 -0.23 0')
    for i, (p, size) in enumerate([
        ('0 0 0.004', '0.046 0.046 0.004'),
        ('0.045 0 0.04', '0.004 0.049 0.04'),
        ('-0.045 0 0.04', '0.004 0.049 0.04'),
        ('0 0.045 0.04', '0.041 0.004 0.04'),
        ('0 -0.045 0.04', '0.041 0.004 0.04'),
    ]):
        ET.SubElement(cup, 'geom', name=f'holder_{i}', type='box', pos=p, size=size, rgba='0.2 0.35 0.6 1')
    tree.write(target, encoding='utf-8')


class Sim:
    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.qidx = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINTS])
        self.aidx = np.array([self.model.actuator(n).id for n in JOINTS])
        self.bounds = self.model.actuator_ctrlrange[self.aidx].copy()
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def to_sim(self, state):
        q = np.deg2rad(np.asarray(state, dtype=np.float64))
        q[5] = self.bounds[5, 0] + state[5] / 100 * np.diff(self.bounds[5])[0]
        return q

    def state(self):
        q = self.data.qpos[self.qidx].copy()
        s = np.rad2deg(q)
        s[5] = (q[5] - self.bounds[5, 0]) / np.diff(self.bounds[5])[0] * 100
        return s.astype(np.float32)

    def reset(self, seed):
        mujoco.mj_resetData(self.model, self.data)
        q = np.clip(self.to_sim(INITIAL), self.bounds[:, 0], self.bounds[:, 1])
        self.data.qpos[self.qidx] = q
        self.data.ctrl[self.aidx] = q
        idx = self.model.jnt_qposadr[self.model.joint('object_free').id]
        self.data.qpos[idx:idx+2] += np.random.default_rng(seed).uniform(-0.012, 0.012, 2)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(150):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def images(self):
        frames = {}
        for name, camera in [('front', 'wrist_cam'), ('side', 'side')]:
            self.renderer.update_scene(self.data, camera=camera)
            frames[name] = self.renderer.render().copy()
        return frames

    def step(self, action):
        raw = self.to_sim(action)
        bounded = np.clip(raw, self.bounds[:, 0], self.bounds[:, 1])
        self.data.ctrl[self.aidx] = bounded
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        assert np.isfinite(self.data.qpos).all(), 'Non-finite simulation state'
        return int(np.count_nonzero(raw != bounded))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--episodes', type=int, default=3)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.output / 'scene.xml'
    make_scene(args.asset, scene)
    sim = Sim(scene)
    sim.reset(0)
    # Analytic unit round trip, independent of dynamics.
    q = sim.to_sim(INITIAL)
    s = np.rad2deg(q); s[5] = (q[5] - sim.bounds[5, 0]) / np.diff(sim.bounds[5])[0] * 100
    np.testing.assert_allclose(s, INITIAL, atol=1e-6)
    for name, frame in sim.images().items():
        imageio.imwrite(args.output / f'initial_{name}.png', frame)
    policy = None
    if args.checkpoint:
        import torch
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
        from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
        from lerobot.policies import make_pre_post_processors
        if args.device == 'cuda':
            assert torch.cuda.is_available(), 'CUDA unavailable'
        torch.set_num_threads(8)
        torch.manual_seed(1000)
        config = VLAJEPAConfig.from_pretrained(str(args.checkpoint))
        config.device = args.device
        if args.device == 'cpu':
            config.torch_dtype = 'float32'
        policy = VLAJEPAPolicy.from_pretrained(str(args.checkpoint), config=config).to(args.device).eval()
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(args.checkpoint),
                                            preprocessor_overrides={'device_processor': {'device': args.device}})
    report = {'mode': 'trained_policy_closed_loop' if policy else 'physics_only_smoke',
              'checkpoint': str(args.checkpoint), 'device': args.device, 'fps': 30,
              'limitations': ['Approximate camera placement and visual domain',
                              'Nominal degree-to-radian mapping; physical calibration offsets not fitted',
                              'Solid cylinder tape proxy and square holder, not a real-scene success benchmark'],
              'episodes': []}
    for ep in range(args.episodes):
        sim.reset(ep)
        if policy:
            policy.reset(); pre.reset(); post.reset()
        rows = []; clipped = 0; heights = []; contacts = 0; start = time.monotonic()
        objid = sim.model.body('tape_proxy').id
        tapeid = sim.model.geom('tape').id
        with imageio.get_writer(str(args.output / f'episode_{ep:02d}.mp4'), fps=30) as video:
            for step in range(args.steps):
                frames = sim.images(); state = sim.state()
                if policy:
                    observation = {'observation.state': torch.from_numpy(state), 'task': 'Put the tape into the pen holder'}
                    for name, frame in frames.items():
                        observation['observation.images.' + name] = torch.from_numpy(frame).permute(2, 0, 1).float() / 255
                    with torch.inference_mode(), (torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()):
                        action = post(policy.select_action(pre(observation))).detach().cpu().numpy().reshape(-1)
                    assert action.shape == (6,) and np.isfinite(action).all()
                else:
                    action = INITIAL.copy()
                clipped += sim.step(action)
                position = sim.data.xpos[objid].copy(); heights.append(float(position[2]))
                for contact in sim.data.contact:
                    if tapeid in (contact.geom1, contact.geom2):
                        other = contact.geom2 if contact.geom1 == tapeid else contact.geom1
                        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ''
                        contacts += int('jaw' in name)
                rows.append({'step': step, 'state': state.tolist(), 'action': action.tolist(), 'object_pos': position.tolist()})
                video.append_data(np.concatenate([frames['front'], frames['side']], axis=1))
                if step % 30 == 0:
                    print(json.dumps({'episode': ep, 'step': step, 'object_z': float(position[2])}), flush=True)
        summary = {'episode': ep, 'steps': len(rows), 'max_object_z_m': max(heights),
                   'lifted_above_5cm': max(heights) > 0.05, 'jaw_contact_samples': contacts,
                   'clipped_joint_commands': clipped, 'wall_seconds': time.monotonic() - start,
                   'final_object_pos': position.tolist(),
                   'placed_in_holder': bool(abs(position[0] - .13) < .035 and abs(position[1] + .23) < .035 and position[2] < .06)}
        (args.output / f'episode_{ep:02d}.json').write_text(json.dumps(rows))
        report['episodes'].append(summary)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print('EPISODE_COMPLETE', json.dumps(summary), flush=True)
    sim.renderer.close()


if __name__ == '__main__':
    main()
