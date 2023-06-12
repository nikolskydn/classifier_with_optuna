import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is not None:
    devices = cuda_visible_devices.split(',')
    device_count = len(devices)
    print("CUDA_VISIBLE_DEVICES:", devices)
    print("Device Count:", device_count)
else:
    print("CUDA_VISIBLE_DEVICES not found.")
