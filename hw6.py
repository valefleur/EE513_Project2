from pyclbr import Class

import torch
import torch.nn as nn
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
    fig.savefig(f"{OUTPUT_PATH}p{problem:02d}_training_loss_lr_{rate:.3f}.png")
    plt.close(fig)


def num_params(module):
    '''
    Returns the number of parameters in the PyTorch module.
    Code from Professor McNames 11_deep_learning_on_ramp_a.ipynb.
    This is useful for checking the size of a model and comparing it to other models.
    '''
    return sum(t.numel() for t in module.parameters())


def receptive_field(model, n=96, tries=8):
    # The input pixels where the gradient of the center pixel of the predicted noise y - f(y)
    # is not zero, for any of a few random inputs; and how far they reach.
    # Borrowed from Professor McNames 11_deep_learning_on_ramp_a.ipynb.
    hit = np.zeros((n, n), bool)
    for _ in range(tries):
        u = torch.rand(1, 1, n, n, requires_grad=True)
        (u - model.eval()(u))[0, 0, n // 2, n // 2].backward()
        hit |= u.grad[0, 0].numpy() != 0
    rows, cols = np.nonzero(hit)
    return hit, rows.max() - rows.min() + 1, cols.max() - cols.min() + 1


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
    # print problem with a leading zero if less than 10, and no decimal places
    fig.savefig(f"{OUTPUT_PATH}p{problem:02d}_guassian_filtered_image.png")
    plt.close(fig)

    # img from problem 2 as float32 scaled to 0-1
    img_float32 = img.astype(np.float32) / 255.0
    out_scipy = ndimage.correlate(img_float32.astype(np.float64),
                                  kernel[:, :, None],   # (5, 5, 1): don't mix across channels
                                  mode='mirror')        # 'mirror' matches the 'reflect' padding used in PyTorch Conv2d

    print(f"Scipy filtered image shape [{out_scipy.shape}] and type [{out_scipy.dtype}].")

    # Plot the Scipy filtered image
    fig, ax = plt.subplots()
    ax.imshow(out_scipy)
    ax.set_title("Scipy Filtered Image")
    if SHOW:
        plt.show()
    fig.savefig(f"{OUTPUT_PATH}p{problem:02d}_scipy_gaussian_filtered_image.png")
    plt.close(fig)

    max_diff = np.abs(out_tensor - out_scipy).max()
    print(f"Max difference between PyTorch and Scipy filtered images: {max_diff:.12f}")
    print(f"float32 eps: {np.finfo(np.float32).eps:.6e}, float64 eps: {np.finfo(np.float64).eps:.6e}")
    '''
    Rule of thumb:
    A difference of 1 to 10 eps is rounding.
    A difference of 1e-4 or larger (or anything that doesn't shrink when you check) points to a bug
    such as mismatched borders.
    '''


    # Problem 9
    problem = 9
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")
    # Double check by seperating border and interior regions
    d = np.abs(out_tensor - out_scipy)

    interior = d[half:-half, half:-half, :]
    print(f"Interior max: [{interior.max():.12f}] or [{interior.max():.3e}]")

    mask = np.ones(d.shape[:2], dtype=bool)
    mask[half:-half, half:-half] = False
    border = d[mask]    # the 2-pixel frame around the image
    print(f"Border   max: [{border.max():.12f}] or [{border.max():.3e}]")

    '''
    If both maxima are around 1e-7 to 3e-7, then the difference is likely due to rounding errors and there is no disagreement.
    If the border max is much larger than the interior max, then there is likely a bug in how the borders are handled.
    If both are large (say > 1e-2), then there is likely a bug in the weights or scaling of the kernel.
    '''
    print(f"There was no disagreement between the PyTorch and SciPy approaches beyond the float32 rounding.  The max difference of {max_diff:.1e} (about {np.finfo(np.float32).eps:.1e} eps) is accumulated rounding error from float32 arithmetic compared to float64 arithmetic.  The interior max is {interior.max():.1e} and the border max is {border.max():.1e}, so there is no disagreement in the border handling either.")


    # Problem 10
    problem = 10
    print(f"\n\n*~*~*~*~*~*~*~*~*~*~ Problem {problem} ~*~*~*~*~*~*~*~*~*~*")


    class SmallDenoiser(nn.Module):
        # Structured after Professor McNames' SmallDnCNN class in 11_deep_learning_on_ramp_a.ipynb, cell 12
        def __init__(self, maps=16):
            super().__init__()
            k = 3 # kernel size
            # Padding keeps the output the same size as the input.
            # This pads with zeros by default, but we could also use 'reflect' or 'replicate'
            # to avoid introducing new values.
            pad = (k-1) // 2
            '''
            In this CNN, the first convolution turns a gray map into 16 feature maps, then the
            middle layers combine those 16 feature maps into 16 new feature maps, and the last
            layer collapses them back into one map (or three for color) so that the output is
            the same size as the input.  The output is the noise, so it can be subtracted
            from the input to get the denoised image.
            Question: Why is the output noise and not something else?
            Question: Why would collapsing down to three maps give color instead of something else?
            Generally, why does this math and ML give us the results we are hoping for instead of
            something that might mean something to the computer but not to humans?  Why does this work?

            '''
            self.body = nn.Sequential(
                # C_out(C_in * k^2 + 1) parameters per layer, where C_out = C_in = maps for the hidden layers, and C_out = 1 for the last layer
                nn.Conv2d(1, maps, k, padding=pad),       # 16(1*9 + 1) = 160 parameters
                nn.ReLU(),
                nn.Conv2d(maps, maps, k, padding=pad),    # 16(16*9 + 1) = 2320 parameters
                nn.ReLU(),
                nn.Conv2d(maps, maps, k, padding=pad),    # 16(16*9 + 1) = 2320 parameters
                nn.ReLU(),
                nn.Conv2d(maps, 1, k, padding=pad)        # 1(16*9 + 1) = 145 parameters
                # no ReLu after last layer; this allows for negative values in the output
            )
            print(f"Number of parameters in SmallDenoiser: {num_params(self)}")

        def forward(self, y):
            return y - self.body(y)  # residual learning: output is the noise, so subtract it from the input to get the denoised image
            '''
            Question: So forward is the goal, the function to optimize over?
            Answer: Yes, the forward method defines how the input data flows through the network
            layers to produce the output. During training, PyTorch uses this method to compute the
            output and then calculates the loss based on the difference between the predicted
            output and the actual target. The optimizer then updates the model's parameters based
            on this loss.
            Question: Which parameters?  Just the elements of the kernels since C_out and C_in are
            calculated?
            Answer: Yes, the parameters are the weights and biases of the convolutional layers,
            which are the elements of the kernels.
            Question: Where are the biases represented in the code above?
            Answer: The biases are part of the convolutional layers in PyTorch. By default,
            nn.Conv2d includes a bias term for each output channel. In the code above, the biases
            are automatically included in each Conv2d layer unless you explicitly set bias=False
            when defining the layer.
            Question: So the kernel weights are analagus to the slope from a least squares fit and the
            bias is analagus to the intercept?
            Answer: Yes, in a way. The kernel weights in a convolutional layer determine how the input
            is transformed to produce the output, similar to how the slope in a linear equation determines
            the rate of change. The bias term allows for an additional degree of freedom, similar to the
            intercept in a linear equation, allowing the model to shift the output up or down. However,
            convolutional layers are more complex than simple linear equations, as they involve multiple
            weights and biases that interact with the input data in a spatial manner.
            Question: Why is it called "forward"?  What is the context there?
            Answer: The term "forward" refers to the forward pass of the neural network, where the
            input data is passed through the network layers to produce an output. This is in
            contrast to the "backward" pass, where gradients are computed and parameters are updated
            based on the loss. The forward method is a standard convention in PyTorch for defining
            how data flows through the model.
            Question: What is self.body()?  What value(s) does it hold and what do they represent?
            Answer: self.body is an instance of nn.Sequential, which is a container that holds a
            sequence of layers (in this case, convolutional and ReLU layers). It represents the
            architecture of the neural network. When you call self.body(y), it passes the input y
            through all the layers defined in the nn.Sequential container, producing the output of
            the network.
            '''

    first_cnn_denoiser = SmallDenoiser()
    for i, layer in enumerate(first_cnn_denoiser.body):
        print(f"Layer {i}: {type(layer).__name__} has {num_params(layer)} parameters with shapes: {[tuple(p.shape) for p in layer.parameters()]}")

    # Run receptive_Filed on my small denoiser to see how far this neural network can affect per pixel
    hit, rows, cols = receptive_field(first_cnn_denoiser)
    print(f"Receptive field hit shape: {hit.shape}, rows: {rows}, cols: {cols}")
    plt.imshow(hit, cmap='gray')
    plt.title(f"Receptive Field Hit Map\nRows: {rows}, Cols: {cols}")
    if SHOW:
        plt.show()
    plt.savefig(f"{OUTPUT_PATH}p{problem:02d}_receptive_field_hit_map.png")
    plt.close()