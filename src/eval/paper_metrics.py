from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PaperMetricState:
    count: int
    radial_spectrum_sum: torch.Tensor
    hf_ratio_sum: torch.Tensor
    sobel_mean_sum: torch.Tensor
    sobel_std_sum: torch.Tensor
    laplacian_abs_sum: torch.Tensor


def image_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    images = ((images.float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    if images.shape[1] == 1:
        return images
    weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def radial_bin_indices(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.arange(height, device=device) - height // 2
    x = torch.arange(width, device=device) - width // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.sqrt(yy.float().square() + xx.float().square()).floor().long()


def radial_energy(images: torch.Tensor, bin_indices: torch.Tensor) -> torch.Tensor:
    grayscale = image_to_grayscale(images).squeeze(1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(grayscale, norm="ortho"), dim=(-2, -1))
    energy = spectrum.abs().square()
    batch_size = int(images.shape[0])
    flat_bins = bin_indices.flatten().unsqueeze(0).expand(batch_size, -1)
    flat_energy = energy.flatten(start_dim=1)
    num_bins = int(bin_indices.max().item()) + 1
    binned = torch.zeros(batch_size, num_bins, device=images.device, dtype=flat_energy.dtype)
    binned.scatter_add_(dim=1, index=flat_bins, src=flat_energy)
    return binned


def radial_spectrum(images: torch.Tensor, bin_indices: torch.Tensor) -> torch.Tensor:
    binned_energy = radial_energy(images, bin_indices)
    counts = torch.bincount(bin_indices.flatten(), minlength=binned_energy.shape[1]).to(
        device=images.device,
        dtype=binned_energy.dtype,
    )
    return binned_energy / counts.clamp_min(1.0)


def high_frequency_ratio(images: torch.Tensor, bin_indices: torch.Tensor, threshold: float) -> torch.Tensor:
    binned_energy = radial_energy(images, bin_indices)
    max_radius = float(bin_indices.max().item())
    high_frequency_mask = torch.arange(binned_energy.shape[1], device=images.device, dtype=torch.float32) > (
        threshold * max_radius
    )
    high_energy = binned_energy[:, high_frequency_mask].sum(dim=1)
    total_energy = binned_energy.sum(dim=1).clamp_min(1e-12)
    return high_energy / total_energy


def edge_statistics(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grayscale = image_to_grayscale(images)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=grayscale.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=images.device,
        dtype=grayscale.dtype,
    ).view(1, 1, 3, 3)
    laplacian = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=images.device,
        dtype=grayscale.dtype,
    ).view(1, 1, 3, 3)
    gradient_x = F.conv2d(grayscale, sobel_x, padding=1)
    gradient_y = F.conv2d(grayscale, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)
    sobel_mean = gradient_magnitude.flatten(start_dim=1).mean(dim=1)
    sobel_std = gradient_magnitude.flatten(start_dim=1).std(dim=1, unbiased=False)
    laplacian_abs = F.conv2d(grayscale, laplacian, padding=1).abs().flatten(start_dim=1).mean(dim=1)
    return sobel_mean, sobel_std, laplacian_abs


def initialize_state(image_size: int, device: torch.device) -> PaperMetricState:
    bin_indices = radial_bin_indices(image_size, image_size, device)
    num_bins = int(bin_indices.max().item()) + 1
    scalar = torch.zeros((), device=device)
    return PaperMetricState(
        count=0,
        radial_spectrum_sum=torch.zeros(num_bins, device=device),
        hf_ratio_sum=scalar.clone(),
        sobel_mean_sum=scalar.clone(),
        sobel_std_sum=scalar.clone(),
        laplacian_abs_sum=scalar.clone(),
    )


@torch.no_grad()
def update_state(state: PaperMetricState, images: torch.Tensor, hf_threshold: float) -> None:
    batch_size = int(images.shape[0])
    bin_indices = radial_bin_indices(int(images.shape[-2]), int(images.shape[-1]), images.device)
    state.radial_spectrum_sum += radial_spectrum(images, bin_indices).sum(dim=0)
    state.hf_ratio_sum += high_frequency_ratio(images, bin_indices, hf_threshold).sum()
    sobel_mean, sobel_std, laplacian_abs = edge_statistics(images)
    state.sobel_mean_sum += sobel_mean.sum()
    state.sobel_std_sum += sobel_std.sum()
    state.laplacian_abs_sum += laplacian_abs.sum()
    state.count += batch_size


def finalize_state(state: PaperMetricState) -> dict[str, torch.Tensor]:
    count = float(state.count)
    return {
        "radial_spectrum": state.radial_spectrum_sum / count,
        "hf_ratio": state.hf_ratio_sum / count,
        "sobel_mean": state.sobel_mean_sum / count,
        "sobel_std": state.sobel_std_sum / count,
        "laplacian_abs": state.laplacian_abs_sum / count,
    }


def compare_paper_metrics(
    real_state: PaperMetricState,
    generated_state: PaperMetricState,
    eps: float = 1e-12,
) -> dict[str, float]:
    real = finalize_state(real_state)
    generated = finalize_state(generated_state)
    spectrum_distance = (
        torch.log(real["radial_spectrum"].clamp_min(eps)) - torch.log(generated["radial_spectrum"].clamp_min(eps))
    ).abs().mean()
    return {
        "radial_fourier_spectrum_distance": float(spectrum_distance.item()),
        "real_high_frequency_energy_ratio": float(real["hf_ratio"].item()),
        "generated_high_frequency_energy_ratio": float(generated["hf_ratio"].item()),
        "high_frequency_energy_ratio_gap": float((real["hf_ratio"] - generated["hf_ratio"]).abs().item()),
        "real_sobel_mean": float(real["sobel_mean"].item()),
        "generated_sobel_mean": float(generated["sobel_mean"].item()),
        "sobel_mean_gap": float((real["sobel_mean"] - generated["sobel_mean"]).abs().item()),
        "real_sobel_std": float(real["sobel_std"].item()),
        "generated_sobel_std": float(generated["sobel_std"].item()),
        "sobel_std_gap": float((real["sobel_std"] - generated["sobel_std"]).abs().item()),
        "real_laplacian_abs": float(real["laplacian_abs"].item()),
        "generated_laplacian_abs": float(generated["laplacian_abs"].item()),
        "laplacian_abs_gap": float((real["laplacian_abs"] - generated["laplacian_abs"]).abs().item()),
    }
