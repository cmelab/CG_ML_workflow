import torch

def calculate_error(prediction, target, target_stats, device):
    force_std = torch.from_numpy(target_stats['force_std']).type(torch.FloatTensor).to(device)
    predicted_force = prediction[:, :3] * force_std
    target_force = target[:, :3] * force_std
    force_error = (predicted_force - target_force).abs().mean()

    torque_std = torch.from_numpy(target_stats['torque_std']).type(torch.FloatTensor).to(device)
    predicted_torque = prediction[:, 3:] * torque_std
    target_torque = target[:, 3:] * torque_std

    torque_error = (predicted_torque - target_torque).abs().mean()

    return force_error.item(), torque_error.item()


def syn_calculate_error(prediction, target, target_stats, device):
    force_std = torch.from_numpy(target_stats['force_std']).type(torch.FloatTensor).to(device)
    predicted_force = prediction[:, :3] * force_std
    target_force = target[:, :3] * force_std
    force_error = (predicted_force - target_force).abs().mean()

    torque_std = torch.from_numpy(target_stats['torque_std']).type(torch.FloatTensor).to(device)
    predicted_torque = prediction[:, 3:] * torque_std
    target_torque = target[:, 3:] * torque_std

    torque_error = (predicted_torque - target_torque).abs().mean()

    return force_error.item(), torque_error.item()