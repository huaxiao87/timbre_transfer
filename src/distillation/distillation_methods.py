import torch
import torch.nn as nn
from src.models.synths.hpn_synth import exp_sigmoid

class DistillMethod(nn.Module):
    """Base class for distillation methods."""
    def __init__(self, loss, w):
        super(DistillMethod, self).__init__()
        self.loss = loss
        self.w = w
        
    def init_distillation(self, teacher, student):
        pass
        
    def forward(self, y_t, y_s):
        loss = self.loss(y_t, y_s)
        return self.w * loss, loss
    
    

class DistillAudio(DistillMethod):
    """Response-based distillation: audio."""
        
    def __init__(self, loss, w):
        super(DistillAudio, self).__init__(loss, w)
        
    def init_distillation(self, teacher, student):
        assert teacher.get_sr() == student.get_sr(), "Sample rates must be the same for teacher and student models"
        
    def forward(self, y_t, y_s):
        loss = {}
        loss['audio'] = self.loss(y_t["synth_audio"], y_s["synth_audio"])
        return self.w * loss['audio'], loss
    
    
class DistillControl(DistillMethod):
    """Response-based distillation (only for DDSP models): parameters."""
    def __init__(self, params, loss, w, mode='sum', scale_fn=exp_sigmoid):
        super(DistillControl, self).__init__(loss, w)
        assert mode in ['mean', 'sum'], "mode must be either 'mean' or 'sum'"
        self.mode = mode
        self.params = params
        self.scale_fn = scale_fn

    def init_distillation(self, teacher, student):
        assert type(teacher).__name__ == 'DDSP_Decoder' and type(student).__name__ == "DDSP_Decoder", f"For Control Distillation teacher model and student model must be a DDSP_Decoder, but got teacher: {type(teacher).__name__} and student: {type(student).__name__}"
        assert teacher.get_params() == student.get_params(), f"Output keys must be the same for teacher {teacher.get_params()} and student models {student.get_params()}"
        assert self.params == teacher.get_params(), f"Distillation params {self.params} must correspond to the output keys of the DDSP models {teacher.get_params()}"

    def forward(self, y_t, y_s):
        loss = {}
        c_loss = 0.
        for param in self.params:
            loss[param] = self.loss(self.scale_fn(y_t[param]), self.scale_fn(y_s[param]))

        if self.mode == 'mean':
            c_loss = torch.stack(list(loss.values())).mean()
        else:
            c_loss = torch.stack(list(loss.values())).sum()

        return self.w * c_loss, loss
    
class DistillFeatures(DistillMethod):
    """Feature-based distillation."""
    def __init__(self, hook_layers, align_features, loss, w, mode='mean'):
        super(DistillFeatures, self).__init__(loss, w)
        assert mode in ['mean', 'sum'], "mode must be either 'mean' or 'sum'"
        self.mode = mode
        self.teacher_features = {}
        self.student_features = {}
        self.hook_layers = hook_layers
        self.fmap = {}
        for layer, (teacher_transform, student_transform) in zip(hook_layers, align_features):
            self.fmap[layer] = []
            self.fmap[layer].append(self._init_features_align(teacher_transform))
            self.fmap[layer].append(self._init_features_align(student_transform))


    def init_distillation(self, teacher, student):
        for layer in self.hook_layers:
            assert self.has_layer(teacher, layer), f"Layer {layer} not found in teacher model"
            assert self.has_layer(student, layer), f"Layer {layer} not found in student model"

        for layer in self.hook_layers:
            self.find_layer(teacher, layer).register_forward_hook(self._extract_teacher_feature(layer))
            self.find_layer(student, layer).register_forward_hook(self._extract_student_feature(layer))
             
        # Got problem: check if the device is the same for teacher, student and fmap   
        device = next(teacher.parameters()).device
        assert device == next(student.parameters()).device, "Teacher and student models must be on the same device"
        
        for fmap in self.fmap.values():
            for f in fmap:
                f.to(device)
            

    def _init_features_align(self, features_transform):
        if type(features_transform) is list:
            in_features, out_features = features_transform 
            fmap = nn.Sequential(
            #nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            #nn.Conv1d(out_channels, out_channels, kernel_size=1)
            nn.Linear(out_features, out_features)
        )
        elif features_transform == -1: 
            fmap = nn.Identity()
        else:
            raise ValueError(f"Invalid features transform {features_transform}. Must be -1 (Identity transformation) or a tuple of (in_features, out_features) for linear transformation.")
        return fmap

    def _extract_teacher_feature(self, name):
            def hook(model, input, output):
                    self.teacher_features[name] = output[0] if type(output)==tuple else output

            return hook

    def _extract_student_feature(self, name):
            def hook(model, input, output):
                self.student_features[name] = output[0] if type(output)==tuple else output

            return hook

    def has_layer(self, model, name):
        """
        Checks if a PyTorch model has a submodule specified by dot notation.

        Args:
            model (nn.Module): The PyTorch model to check.
            dot_notation (str): The dot notation string representing the path to the submodule.

        Returns:
            bool: True if the submodule exists, False otherwise.
        """
        attributes = name.split('.')
        current_module = model
        for attr in attributes:
            if hasattr(current_module, attr):
                current_module = getattr(current_module, attr)
            else:
                return False
        return True

    def find_layer(self, model, name):
        attributes = name.split('.')
        current_module = model
        for attr in attributes:
            if hasattr(current_module, attr):
                current_module = getattr(current_module, attr)
            else:
                return None
        return current_module

    def forward(self, y_t, y_s):
        loss = {}
        f_loss = 0.
        for layer in self.hook_layers:
            t_features = self.fmap[layer][0](self.teacher_features[layer])
            s_features = self.fmap[layer][1](self.student_features[layer])
            loss[layer] = self.loss(t_features, s_features)
        if self.mode == 'mean':
            f_loss = torch.stack(list(loss.values())).mean()
        else:
            f_loss = torch.stack(list(loss.values())).sum()
        return self.w * f_loss, loss
    
# class DistillFeatures(DistillMethod):
#     """Feature-based distillation."""
#     def __init__(self, hook_layers, loss, w, mode='mean'):
#         super(DistillFeatures, self).__init__(loss, w)
#         assert mode in ['mean', 'sum'], "mode must be either 'mean' or 'sum'"
#         self.mode = mode
#         self.teacher_features = {}
#         self.student_features = {}
#         self.hook_layers = hook_layers

#     def init_distillation(self, teacher, student):
#         for layer in self.hook_layers:
#             assert self.has_layer(teacher, layer), f"Layer {layer} not found in teacher model"
#             assert self.has_layer(student, layer), f"Layer {layer} not found in student model"

#         for layer in self.hook_layers:
    #         self.find_layer(teacher, layer).register_forward_hook(self._extract_teacher_feature(layer))
    #         self.find_layer(student, layer).register_forward_hook(self._extract_student_feature(layer))

    # def _extract_teacher_feature(self, name):
    #       def hook(model, input, output):
    #         self.teacher_features[name] = output[0] if type(output)==tuple else output

    #       return hook

    # def _extract_student_feature(self, name):
    #       def hook(model, input, output):
    #         self.student_features[name] = output[0] if type(output)==tuple else output

    #       return hook

    # def has_layer(self, model, name):
    #     """
    #     Checks if a PyTorch model has a submodule specified by dot notation (example: 'encoder.conv1')

    #     Args:
    #       model (nn.Module): The PyTorch model to check.
    #       dot_notation (str): The dot notation string representing the path to the submodule.

    #     Returns:
    #       bool: True if the submodule exists, False otherwise.
    #     """
    #     attributes = name.split('.')
    #     current_module = model
    #     for attr in attributes:
    #       if hasattr(current_module, attr):
    #           current_module = getattr(current_module, attr)
    #       else:
    #           return False
    #     return True

    # def find_layer(self, model, name):
    #     attributes = name.split('.')
    #     current_module = model
    #     for attr in attributes:
    #       if hasattr(current_module, attr):
    #           current_module = getattr(current_module, attr)
    #       else:
    #           return None
    #     return current_module

    # def forward(self, y_t, y_s):
    #     loss = {}
    #     f_loss = 0.
    #     for layer in self.hook_layers:
    #         loss[layer] = self.loss(self.teacher_features[layer], self.student_features[layer])
    #     if self.mode == 'mean':
    #         f_loss = torch.stack(list(loss.values())).mean()
    #     else:
    #         f_loss = torch.stack(list(loss.values())).sum()
    #     return self.w * f_loss, loss
        