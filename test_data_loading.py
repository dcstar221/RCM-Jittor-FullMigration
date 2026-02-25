
import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.map_expansion.map_api import NuScenesMap

def check_data():
    print("Checking NuScenes data loading...")
    
    # 1. Check NuScenes initialization
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
        print("✅ NuScenes initialized successfully.")
    except Exception as e:
        print(f"❌ NuScenes initialization failed: {e}")
        return

    # 2. Check Samples
    try:
        first_sample = nusc.sample[0]
        print(f"✅ Found {len(nusc.sample)} samples.")
        print(f"   First sample token: {first_sample['token']}")
        
        # Check sensor file existence
        cam_token = first_sample['data']['CAM_FRONT']
        cam_path = nusc.get_sample_data_path(cam_token)
        if os.path.exists(cam_path):
            print(f"✅ CAM_FRONT file exists: {cam_path}")
        else:
            print(f"❌ CAM_FRONT file missing: {cam_path}")
            
        lidar_token = first_sample['data']['LIDAR_TOP']
        lidar_path = nusc.get_sample_data_path(lidar_token)
        if os.path.exists(lidar_path):
            print(f"✅ LIDAR_TOP file exists: {lidar_path}")
        else:
            print(f"❌ LIDAR_TOP file missing: {lidar_path}")
            
    except Exception as e:
        print(f"❌ Sample check failed: {e}")

    # 3. Check Maps
    try:
        # Check if expansion maps are accessible
        # NuScenesMap expects map_name to be one of: singapore-onenorth, boston-seaport, etc.
        # We'll try loading one that we saw in the expansion folder
        map_name = 'boston-seaport'
        nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name=map_name)
        print(f"✅ Map '{map_name}' loaded successfully.")
    except Exception as e:
        print(f"❌ Map loading failed: {e}")
        print("   (This might be okay if we only need basemaps, but RCM-Fusion usually needs vector maps)")

    # 4. Check CAN Bus
    try:
        nusc_can = NuScenesCanBus(dataroot='data/nuscenes')
        # Try to get can bus data for a scene
        scene_name = nusc.scene[0]['name']
        messages = nusc_can.get_messages(scene_name, 'pose')
        if messages:
            print(f"✅ CAN Bus data found for scene {scene_name} ({len(messages)} messages).")
        else:
            print(f"⚠️ CAN Bus data empty for scene {scene_name}.")
    except Exception as e:
        print(f"❌ CAN Bus check failed: {e}")

if __name__ == "__main__":
    check_data()
