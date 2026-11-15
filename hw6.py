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
    print(f"\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
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
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
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
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
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


    # Problem 4
    problem = 4
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")

    # Evaluate the derivative at a point in a one-variable scalar function using autograd
    # and compare the difference
    '''
    Function:
    f(x) = 9x^3 + 8x^2 + 4x + 1
    Evaluate the gradient at x = 2.

    d/dx = 27x^2 + 16x + 4

    d/dx(2) = 27(2)^2 + 16(2) + 4 = 144
    '''
    at = 2.0
    x = torch.tensor(at, requires_grad=True)

    y = 9*x**3 + 8*x**2 +4*x + 1

    # Compute the derivative using backpropagation
    y.backward()

    hand_derived = 27*at**2 + 16*at + 4
    torch_gradient = x.grad.item()

    print(f"f({at}) = {hand_derived} by hand.")
    # print(f"The same function using backpropagation gave {y.item()}.")
    print(f"f({at}) = {torch_gradient} by PyTorch gradient.")

    diff = abs(torch_gradient - hand_derived)
    print(f"Difference: {diff}")
    assert diff < 1e-6


    # Problem 5
    problem = 5
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")

    slope = 2.0
    bias = 3.0
    print(f"Reality:\t\tslope [{slope:.3f}] and intercept [{bias:.3f}].")
    rnd = np.random.default_rng(55)
    noise = rnd.normal(loc=0, scale=1.0, size=20)

    x_vals = np.arange(0, 20, dtype=float)
    clean_line = x_vals*slope + bias
    noisy_line = clean_line + noise

    # print(f"clean_line shape: {clean_line.shape}, max {clean_line.max():.3f}, mean {clean_line.mean():.3f}")
    # print(f"noisy_line shape: {noisy_line.shape}, max {noisy_line.max():.3f}, mean {noisy_line.mean():.3f}")

    ## PyTorch loop
    # y = mx + b
    x = torch.tensor(x_vals, dtype=torch.float32)
    y = torch.tensor(noisy_line, dtype=torch.float32)

    # Initial guesses for slope and intercept
    m = torch.tensor(0.0, requires_grad=True) # slope
    b = torch.tensor(0.0, requires_grad=True) # intercept

    # the PyTorch optimizer takes an iterable list/array of the tensors (or dicts) to optimize over
    optimizer = torch.optim.SGD([m, b], lr=0.001)

    losses = []
    # Loop for a while; no stop criteria
    for iteration in range(5000):
        optimizer.zero_grad()   # clear old gradients that accumulate across backward() calls
        prediction = m*x + b    # the forward pass
        loss = ((prediction - y)**2).mean() # MSE here, could be another loss function
        losses.append(loss)
        loss.backward()         # calculate gradients for this step
        optimizer.step()        # Perform optimization and update the variables we are optimizing over, i.e. m and b

    print(f"PyTorch gradient decent slope [{m.item():.3f}] and intercept [{b.item():.3f}].")

    ## Least Squares
    # y = A * x
    A = np.column_stack([x_vals, np.ones_like(x_vals)])
    solution, residuals, rank, sv = np.linalg.lstsq(A, noisy_line, rcond=None)
    print(f"The Least Squares found slope [{solution[0]:.3f}] and intercept [{solution[1]:.3f}].")

    slope_diff = m.item() - solution[0]
    intercept_diff = b.item() - solution[1]
    print(f"Difference in slope:\t  {slope_diff:.3f}\nDifference in intercept: {intercept_diff:.3f}")