from skimage.transform import resize
import torch
import torch.nn.functional as F


def gradcam(model, layer_name, image, idx=None):
    """Visualize CNN decisions with Grad-CAM.

    Args:
        model (nn.Module): CNN.
        layer_name (str): The layer whose output is used to obtain a Grad-Cam
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
        raise ValueError('`gradcam` function does not support batch input.')

    image_size = (image.shape[2], image.shape[3])

    def forward_hook(module, _, output):
        module.feature_maps = output

    def backward_hook(module, _, grad_output):
        module.feature_maps_grad = grad_output[0]

    layer = getattr(model, layer_name)
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    y = model(image).squeeze(0)
    if idx is None:
        idx = torch.argmax(y).item()
    model.zero_grad()
    y[idx].backward()

    with torch.no_grad():
        feature_maps = layer.feature_maps.squeeze(0)
        feature_maps_grad = layer.feature_maps_grad.squeeze(0)
        alpha = feature_maps_grad.mean(2, keepdim=True).mean(1, keepdim=True)
        cam = F.relu((alpha * feature_maps).sum(0))
        # Rescale to [0, 1]
        cam = cam / torch.max(cam)
        cam = resize(cam.cpu().numpy(), image_size)

    del layer.feature_maps
    del layer.feature_maps_grad
    forward_handle.remove()
    backward_handle.remove()

    return cam
