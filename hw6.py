import torch
import rawpy as raw
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

INPUT_PATH = "images/iPhone16/Lightroom/"
OUTPUT_PATH = "output/hw6/"

SHOW = False

def get_torch_afine_gradient(x, y, learning_rate=0.01, iterations=5000):
    # Initial guesses for slope and intercept
    m = torch.tensor(0.0, requires_grad=True) # slope
    b = torch.tensor(0.0, requires_grad=True) # intercept

    # the PyTorch optimizer takes an iterable list/array of the tensors (or dicts) to optimize over
    optimizer = torch.optim.SGD([m, b], lr=learning_rate)

    losses = []
    # Loop for a while; no stop criteria
    for iteration in range(5000):
        optimizer.zero_grad()   # clear old gradients that accumulate across backward() calls
        prediction = m*x + b    # the forward pass
        loss = ((prediction - y)**2).mean() # MSE here, could be another loss function
        losses.append(loss.item())
        # if iteration % 500 == 0:
        #     print(f"Iteration [{iteration:4d}] loss [{loss.item():.3f}] slope [{m.item():.3f}] intercept [{b.item():.3f}].")
        loss.backward()         # calculate gradients for this step
        optimizer.step()        # Perform optimization and update the variables we are optimizing over, i.e. m and b
    losses = np.array(losses)
    return m, b, losses


def plot_losses(losses, problem, rate):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.loglog()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss (lr={rate:.3f})")
    if SHOW:
        plt.show()
    fig.savefig(f"{OUTPUT_PATH}p{problem:2f}_training_loss_lr_{rate:.3f}.png")
    plt.close(fig)


if __name__ == "__main__":

    image_path = f"{INPUT_PATH}hw2_slanted/Nov_08/set3/center_501mm/APC_0658.dng"

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
    #make a copy of tensor to use later
    tensor_copy = tensor.clone()

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

    m, b, losses = get_torch_afine_gradient(x, y, learning_rate=0.001, iterations=5000)
    print(f"PyTorch gradient decent slope [{m.item():.3f}] and intercept [{b.item():.3f}].")

    ## Least Squares
    # y = A * x
    A = np.column_stack([x_vals, np.ones_like(x_vals)])
    solution, residuals, rank, sv = np.linalg.lstsq(A, noisy_line, rcond=None)
    print(f"The Least Squares found slope [{solution[0]:.3f}] and intercept [{solution[1]:.3f}].")

    slope_diff = m.item() - solution[0]
    intercept_diff = b.item() - solution[1]
    print(f"Difference in slope:\t  {slope_diff:.3f}\nDifference in intercept: {intercept_diff:.3f}")


    # Problem 6
    problem = 6
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # Plot the loss against iteration for that fit.
    plot_losses(losses, problem, 0.001)


    # Problem 7
    problem = 7
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # Raise the learning rate until the fit stops converging, and say what the loss curve does when it fails.
    learning_rates = np.linspace(0.001, 0.01, num=10)
    loss_per_rate = []
    for rate in learning_rates:
        m, b, losses = get_torch_afine_gradient(x, y, learning_rate=rate, iterations=5000)
        plot_losses(losses, problem, rate)
        print(f"Learning rate [{rate:.3f}] final slope [{m.item():.3f}] and intercept [{b.item():.3f}].")
        loss_per_rate.append(losses[-1])

    # plot loss curve vs learning rate
    fig, ax = plt.subplots()
    ax.plot(learning_rates, loss_per_rate)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Final Loss")
    ax.set_title(f"Final Loss vs Learning Rate")
    if SHOW:
        plt.show()
    plt.close(fig)
    '''
    It broke around a learning rate of 0.0008.  Up to that point, the estimated slope and intercept were getting more accurate, but then the loss curve went up exponentially and both the slope and the intercept went to NaN.
    '''
    print(f"It broke around a learning rate of 0.0008.  Up to that point, the estimated slope and intercept were getting more accurate, but then the loss curve went up exponentially and both the slope and the intercept went to NaN.  The initial loss starts high, drops as the learning rate increases, and then climbs again as the learning rate gets too high.")


    # Problem 8
    problem = 8
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    sigma = 1.0
    kernel_size = 5
    half = kernel_size // 2
    kernel_axis = np.arange(-half, half + 1)                # [-2, -1, 0, 1, 2]
    gaussian_1D = np.exp(-kernel_axis**2 / (2 * sigma**2))  # 1D Gaussian
    kernel = np.outer(gaussian_1D, gaussian_1D)             # 2D is outer product of 1D
    kernel /= kernel.sum()                                  # Normalize to maintain overall brightness
    print(f"Kernel shape [{kernel.shape}] and total weight [{kernel.sum()}].")

    conv = torch.nn.Conv2d(in_channels=3,
                           out_channels=3,
                           kernel_size=kernel_size,
                           padding=half,
                           padding_mode='reflect',
                           groups=3,
                           bias=False)

    k = torch.tensor(kernel, dtype=torch.float32)
    # Question: How do I know when something needs to be turned into a tensor?
    # Answer: When you need to perform operations that are specific to PyTorch tensors, such as matrix multiplication, gradient computation, or using PyTorch's built-in functions.
    # Question: I guess I'm not quite following how PyTorch organizes the problem space.

    with torch.no_grad():
        # Without copy_(), the weights would be random and we would be using an unitialized kernel.
        # With copy_(), we are explicitly setting the weights to our Gaussian kernel.
        conv.weight.copy_(k.expand(3, 1, kernel_size, kernel_size))  # (out_channels, in_channels/groups, H, W), same kernel for each channel
    conv.weight.requires_grad_(False)  # We don't want to learn the kernel, just use it as is
    print(f"Conv2d weight shape [{conv.weight.shape}] and requires_grad [{conv.weight.requires_grad}].")

    with torch.no_grad():
        filtered_tensor = conv(tensor_copy)  # (1, C, H, W)
    print(f"Filtered tensor shape [{filtered_tensor.shape}].")

    out_tensor = filtered_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (1, C, H, W) -> (C, H, W) -> (H, W, C)

    fig, ax = plt.subplots()
    ax.imshow(out_tensor)
    ax.set_title("PyTorch Filtered Image")
    if SHOW:
        plt.show()
    fig.savefig(f"{OUTPUT_PATH}p{problem:2f}_guassian_filtered_image.png")
    plt.close(fig)

    # img from problem 2 as float32 scaled to 0-1
    img_float32 = img.astype(np.float32) / 255.0
    out_scipy = ndimage.correlate(img_float32.astype(np.float64),
                                  kernel[:, :, None],   # (5, 5, 1): don't mix across channels
                                  mode='mirror')        # matches the reflect padding used in PyTorch Conv2d

    print(f"Scipy filtered image shape [{out_scipy.shape}] and type [{out_scipy.dtype}].")

    # Plot the Scipy filtered image
    fig, ax = plt.subplots()
    ax.imshow(out_scipy)
    ax.set_title("Scipy Filtered Image")
    if SHOW:
        plt.show()
    fig.savefig(f"{OUTPUT_PATH}p{problem:2f}_scipy_gaussian_filtered_image.png")
    plt.close(fig)

    max_diff = np.abs(out_tensor - out_scipy).max()
    print(f"Max difference between PyTorch and Scipy filtered images: {max_diff:.12f}")
    print(f"float32 eps: {np.finfo(np.float32).eps:.6e}, float64 eps: {np.finfo(np.float64).eps:.6e}")