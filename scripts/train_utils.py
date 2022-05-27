import sys
import os

parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(parent_dir, 'models'))

import minkunet

def build_model(model_type):
    if model_type == "14A":
        return minkunet.MinkUNet14A(in_channels=3, out_channels=23, D=3)
    elif model_type == "14B":
        return minkunet.MinkUNet14B(in_channels=3, out_channels=23, D=3)
    elif model_type == "14C":
        return minkunet.MinkUNet14C(in_channels=3, out_channels=23, D=3)
    elif model_type == "14D":
        return minkunet.MinkUNet14D(in_channels=3, out_channels=23, D=3)
    elif model_type == "18A":
        return minkunet.MinkUNet18A(in_channels=3, out_channels=23, D=3)
    elif model_type == "18B":
        return minkunet.MinkUNet18B(in_channels=3, out_channels=23, D=3)
    elif model_type == "18D":
        return minkunet.MinkUNet18D(in_channels=3, out_channels=23, D=3)
    elif model_type == "34A":
        return minkunet.MinkUNet34A(in_channels=3, out_channels=23, D=3)
    elif model_type == "34B":
        return minkunet.MinkUNet34B(in_channels=3, out_channels=23, D=3)
    elif model_type == "34C":
        return minkunet.MinkUNet34C(in_channels=3, out_channels=23, D=3)
    else:
        raise NameError(f"No such model found: {model_type}")