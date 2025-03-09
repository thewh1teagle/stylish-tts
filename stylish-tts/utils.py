from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import matplotlib.pyplot as plt
from munch import Munch
import os
import subprocess


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(path):
    result = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            result = f.readlines()
    return result


def length_to_mask(lengths) -> torch.Tensor:
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def get_git_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError as e:
        print("Error obtaining git commit hash:", e)
        return "unknown"


def get_git_diff():
    try:
        # Run the git diff command
        diff_output = subprocess.check_output(["git", "diff"]).decode("utf-8")
        return diff_output
    except subprocess.CalledProcessError as e:
        print("Error obtaining git diff:", e)
        return ""


def save_git_diff(out_dir):
    hash = get_git_commit_hash()
    diff = get_git_diff()
    diff_file = os.path.join(out_dir, "git_state.txt")
    with open(diff_file, "w") as f:
        f.write(f"Git commit hash: {hash}\n\n")
        f.write(diff)
    print(f"Git diff saved to {diff_file}")


def clamped_exp(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x, device=x.device)
    torch.clamp(x, -35, 35, out=result)
    return torch.exp(result)


def leaky_clamp(
    x_in: torch.Tensor, min_f: float, max_f: float, slope: float = 0.001
) -> torch.Tensor:
    x = x_in
    min_t = torch.full_like(x, min_f, device=x.device)
    max_t = torch.full_like(x, max_f, device=x.device)
    x = torch.maximum(x, min_t)  # + slope * (x - min_t))
    x = torch.minimum(x, max_t)  # + slope * (x - max_t))
    return x


class DecoderPrediction:
    def __init__(
        self,
        audio=None,
        log_amplitude=None,
        phase=None,
        real=None,
        imaginary=None,
        magnitude=None,
    ):
        self.audio = audio
        self.log_amplitude = log_amplitude
        self.phase = phase
        self.real = real
        self.imaginary = imaginary
        self.magnitude = magnitude
