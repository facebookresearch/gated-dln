# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch

from src.model.third_party import resnet as resnet_family


class ResNetModel(torch.nn.Module):
    def __init__(self, name: str, should_use_pretrained: bool, path_to_load: str):
        super().__init__()
        self.name = name
        if self.name == "resnet20":
            self.resnet_model = resnet_family.resnet20_for_feature_extraction()  # type: ignore[attr-defined]
            # error: Module has no attribute "resnet20_for_feature_extraction"  [attr-defined]
        elif self.name == "resnet110":
            self.resnet_model = resnet_family.resnet110_for_feature_extraction()  # type: ignore[attr-defined]
            # error: Module has no attribute "resnet110_for_feature_extraction"  [attr-defined]
        else:
            raise NotImplementedError(f"name = {name} is not supported.")
        if should_use_pretrained:
            self.resnet_model.load_state_dict(torch.load(path_to_load))
            print(f"Using a pretrained model from {path_to_load}")
        for param in self.resnet_model.parameters():
            param.requires_grad = False

    def forward(self, img_target_list):
        img_list, target_list = zip(*img_target_list)
        img_tensor = torch.cat([img.unsqueeze(0) for img in img_list], dim=0)
        with torch.inference_mode():
            features = self.resnet_model(img_tensor)
        return features, torch.tensor(target_list)
