import torch
import rawpy as raw
import numpy as np
import cv2
from PIL import Image

if __name__ == "__main__":

    iphone_images = "images/iPhone16/Lightroom/"
    image_path = f"{iphone_images}hw2_slanted/Nov_08/set3/center_501mm/APC_0658.dng"

    # Problem 1
    problem = 1
    print(f"*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    print(f"PyTorch Version: {torch.__version__}")

    # Hardcode platform for now
    '''
    TODO over winter break:
    import platform
    import subprocess

    Use platform.system() to get the OS.  Based on the OS, create a cmd to pass to
    subprocess.check_output() querying the OS for its details.  Depending on the OS,
    ask either WMI/CIM (Windows), the HW identifier (Mac) or the kernel (Linux).
    '''
    print(f"Running on a MacBook Air M1, 2020 with 16GB of memory.")

    test_tensor = torch.zeros(1,1)
    print(f"The test tensor is stored on the [{test_tensor.device}] device.")

    # Problem 2
    problem = 2
    print(f"*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # TODO turn this into a function that takes an image path and returns
    # the tensor and/or np representation for use

    # Assignment: "load an image ... as a NumPy array in the usual height by width by channel layout, and
    # convert it to a float 32 tensor in PyTorch's batch, channel, height, width layout."
    
    # Note: OpenCV would give BRG instead and would need the channel dimension reordered also
    img = np.array(Image.open(image_path))  # (H, W, C) uint8
    print(f"NumPy\t   shape\t        [{img.shape}] and type [{img.dtype}].")

    t = torch.from_numpy(img)   # (H, W, C)
    print(f"from_numpy shape    [{t.shape}] and type [{t.dtype}].")

    t = t.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    print(f"permute    shape    [{t.shape}] and type [{t.dtype}].")

    # Since images are normally 0 - 255 DN, convert to float and divide
    # by 255 to scale to [0,1].
    t = t.float() / 255.0
    print(f"float32    shape    [{t.shape}] and type [{t.dtype}].")

    # Add a dimension of length 1 to represent the batch.
    tensor = t.unsqueeze(0) # (1, C, H, W) float32
    print(f"unsqueeze  shape [{tensor.shape}] and type [{tensor.dtype}].")


    # Problem 3
    problem = 3
    print(f"*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # Convert tensor back to a NumPy array and assert it equals img
    tensor.squeeze_(0)  # (C, H, W)
    print(f"re-squeezed\t   shape [{tensor.shape}] and type [{tensor.dtype}].")

    t = tensor.permute(1, 2, 0) # (H, W, C)
    print(f"re-permuted\t   shape [{t.shape}] and type [{t.dtype}].")

    # Rescale back to 0.0 - 255.0 then round back to uint8
    t = (t * 255.0).round().to(torch.uint8)
    print(f"Scaled back to int shape [{t.shape}] and type [{t.dtype}].")

    numpy_arr = t.numpy()   # (H, W, C) uint8
    print(f"NumPy\t\t   shape [{numpy_arr.shape}]\t     and type [{numpy_arr.dtype}].")

    print(f"The original NumPy array {'equals' if np.array_equal(numpy_arr, img) else 'does not equal'} the restored Numpy array.")
    assert np.array_equal(numpy_arr, img), "The original and the restored NumPy arrays are different!"