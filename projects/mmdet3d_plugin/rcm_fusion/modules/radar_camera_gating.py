
import jittor as jt
from jittor import nn

def auto_fp16(apply_to=None):
    def decorator(func):
        return func
    return decorator

class RadarCameraGating(nn.Module):
    def __init__(self,
                 in_channels=256):
        super(RadarCameraGating, self).__init__()
        self.in_channels = in_channels

        self.cam_atten_weight = nn.Sequential(
            nn.Conv1d(in_channels,in_channels,kernel_size=7,padding=3),
        )
        self.rad_atten_weight = nn.Sequential(
            nn.Conv1d(in_channels,in_channels,kernel_size=7,padding=3),
        )
        
    @auto_fp16(apply_to=('query_c', 'query_r'))
    def execute(self, query_c, query_r):
        query_rc = (query_c + query_r).permute(0,2,1)
        cam_weight = self.cam_atten_weight(query_rc).sigmoid().permute(0,2,1)
        rad_weight = self.rad_atten_weight(query_rc).sigmoid().permute(0,2,1)
        query_rc = query_c * cam_weight + query_r * rad_weight
        
        return query_rc
