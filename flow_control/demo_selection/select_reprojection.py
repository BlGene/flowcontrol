import numpy as np

from sklearn.preprocessing import minmax_scale

from flow_control.servoing.playback_env_servo import PlaybackEnvServo
from flow_control.servoing.module import ServoingModule


def similarity_from_reprojection(servo_module, live_rgb, demo_rgb, demo_mask, return_images=False):
    # evaluate the similarity via flow reprojection error
    flow = servo_module.flow_module.step(demo_rgb, live_rgb)
    warped = servo_module.flow_module.warp_image(live_rgb / 255.0, flow)
    error = np.linalg.norm((warped - (demo_rgb / 255.0)), axis=2) * demo_mask
    error = error.sum() / demo_mask.sum()
    mean_flow = np.linalg.norm(flow[demo_mask], axis=1).mean()
    if return_images:
        return error, mean_flow, flow, warped
    return error, mean_flow


def normalize_errors(sim_scores, mean_flows):
    sim_l = sim_scores
    mean_flows_l = mean_flows
    w = .5
    print("debug: normalizing", np.max(sim_l), np.max(mean_flows_l))
    sim_scores_norm = np.mean((1 * minmax_scale(sim_l), w * minmax_scale(mean_flows_l)), axis=0) / (1 + w)
    return sim_scores_norm


def select_recording_reprojection(recordings, state, part_info, part_name):
    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    servo_module = ServoingModule(recordings[0], control_config=control_config, start_paused=False, plot=False,
                                  flow_module="RAFT")
    current_rgb = state['rgb_gripper']
    errors = np.ones((len(recordings)))
    mean_flows = np.zeros((len(recordings)))
    for demo_idx, rec in enumerate(recordings):
        playback = PlaybackEnvServo(rec, load='keep')
        frame_index = part_info[rec.name][part_name][0]
        demo_rgb = playback[frame_index].cam.get_image()[0]
        demo_mask = playback.fg_masks[frame_index]

        error, mean_flow = similarity_from_reprojection(servo_module, current_rgb,
                                                        demo_rgb, demo_mask, return_images=False)
        errors[demo_idx] = error
        mean_flows[demo_idx] = mean_flow

    errors_norm = normalize_errors(errors, mean_flows)
    best_idx = np.argmin(errors_norm)
    rec_kfs = part_info[recordings[best_idx].name][part_name]
    return best_idx, rec_kfs
