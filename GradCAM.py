import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        self.model.eval()

        output = self.model(x)

        self.model.zero_grad()

        # 取指定疾病類別的分數
        score = output[:, class_idx].sum()
        score.backward()

        # gradients:   [B, C, H, W]
        # activations: [B, C, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # resize 回原圖大小
        cam = F.interpolate(
            cam,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu()

        # normalize 到 0~1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def show_gradcam(image_tensor, cam, title="Grad-CAM"):
    """
    image_tensor: [1, H, W]
    cam: [H, W]
    """

    image = image_tensor.squeeze().detach().cpu()

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title(title)
    plt.axis("off")
    plt.show()