from skimage.transform import resize
import torch
import torch.nn.functional as F


def gradcam_pp(model, layer_name, image, idx=None):
    """Visualize CNN decisions with Grad-CAM++.

    Args:
        model (nn.Module): CNN.
        layer_name (str): The layer whose output is used to obtain a Grad-Cam++
                          heat-map.
        image (Tensor): Input image.
        idx (int): Class index. If `idx` is None, the index of the maximum
                   output will be used.
    Returns:
        numpy.ndarray: Heat-map with the same size as the input image.
    """
    if model.training:
        raise RuntimeError('Model is not in eval mode.'
                           'Call model.eval() before using this function.')

    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() == 4 and image.shape[0] != 1:
        raise ValueError('`gradcam_pp` function does not support batch input.')

    image_size = (image.shape[2], image.shape[3])

    def forward_hook(module, _, output):
        module.feature_maps = output

    def backward_hook(module, _, grad_output):
        module.feature_maps_grad = grad_output[0]

    layer = getattr(model, layer_name)
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    s = model(image).squeeze(0)
    if idx is None:
        idx = torch.argmax(s).item()
    model.zero_grad()
    s[idx].backward()

    with torch.no_grad():
        feature = layer.feature_maps.squeeze(0)
        feature_grad = layer.feature_maps_grad.squeeze(0)

        feature_sum = torch.sum(feature, dim=(1, 2), keepdim=True)
        alpha_denom =\
            2 * feature_grad.pow(2) + feature_sum * feature_grad.pow(3)
        alpha_denom = torch.where(alpha_denom != 0,
                                  alpha_denom,
                                  torch.ones_like(alpha_denom))
        alpha = feature_grad.pow(2) / alpha_denom
        weight = torch.sum(alpha * F.relu(torch.exp(s[idx]) * feature_grad),
                           dim=(1, 2), keepdim=True)
        cam = F.relu((weight * feature).sum(0))
        # Rescale to [0, 1]
        cam = cam / torch.max(cam)
        cam = resize(cam.cpu().numpy(), image_size)

    del layer.feature_maps
    del layer.feature_maps_grad
    forward_handle.remove()
    backward_handle.remove()

    return cam
