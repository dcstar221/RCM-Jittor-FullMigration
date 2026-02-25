
import numpy as np
import jittor as jt

class LiDARInstance3DBoxes:
    """
    3D boxes in LiDAR coordinates.
    Jittor version of mmdet3d.core.bbox.structures.LiDARInstance3DBoxes.
    
    Args:
        tensor (jt.Var or np.ndarray): Box tensor with shape (N, box_dim).
        box_dim (int): Dimension of box. Default: 7.
        origin (tuple[float]): origin of the box. Default: (0.5, 0.5, 0).
    """
    def __init__(self, tensor, box_dim=7, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, (list, tuple)):
            tensor = np.array(tensor)
        
        self.box_dim = box_dim
        self.origin = origin
        
        if isinstance(tensor, np.ndarray):
             # Ensure float32
             tensor = tensor.astype(np.float32)
             self.tensor = jt.array(tensor)
        else:
             self.tensor = tensor
             
        if self.tensor.shape[-1] != box_dim:
             # padding or trimming might be needed, but usually we match
             pass

    @property
    def gravity_center(self):
        """jt.Var: Calculated gravity center of all boxes."""
        # LiDAR box: [x, y, z, dx, dy, dz, rot, ...]
        # Origin (0.5, 0.5, 0) means (x,y,z) is bottom center.
        # Gravity center (0.5, 0.5, 0.5) means z + dz/2.
        bottom_center = self.tensor[:, :3]
        dims = self.tensor[:, 3:6]
        
        return bottom_center + jt.stack([
            jt.zeros(bottom_center.shape[0]),
            jt.zeros(bottom_center.shape[0]),
            dims[:, 2] * (0.5 - self.origin[2])
        ], dim=-1)

    @property
    def dims(self):
        """jt.Var: Dimensions of boxes."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """jt.Var: Yaw angle of boxes."""
        return self.tensor[:, 6]

    def numpy(self):
        """Convert to numpy array."""
        return self.tensor.numpy()
        
    def __len__(self):
        return self.tensor.shape[0]

    def to(self, device):
        # Jittor handles device automatically usually, but keep API valid
        return self
        
    def clone(self):
        return LiDARInstance3DBoxes(self.tensor.clone(), self.box_dim, self.origin)

    def convert_to(self, dst, rt_mat=None):
        # Minimal implementation, mainly for type checking
        return self
