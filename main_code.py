import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Tuple

class Convolution_Correlation:
    @staticmethod
    def correlation(img: NDArray, mask:NDArray, same_pad: bool = True) -> NDArray:
        h_in, w_in = img.shape
        h_mask, w_mask = mask.shape

        if same_pad:
            pad_h = h_mask // 2
            pad_w = w_mask // 2
            img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)))
            h_out, w_out = h_in, w_in
        else:
            h_out = h_in - h_mask + 1
            w_out = w_in - w_mask + 1

        res = np.empty([h_out, w_out], dtype=np.float32)

        flatten_mask = mask.flatten()
        # for each position
        for y in range(h_out):
            for x in range(w_out):
                # weighed sum
                res[y, x] = img[y: y + h_mask, x: x + w_mask].flatten().dot(flatten_mask)
        return res

    @staticmethod
    def convolution(img: NDArray, mask:NDArray, same_pad: bool = True) -> NDArray:
        # rotate 180 degree mask
        r_mask = mask[::-1, ::-1]
        return Convolution_Correlation.correlation(img, r_mask, same_pad)

class GaussianMask:
    @staticmethod
    def _gauss_function(x:int, y:int, x_0:int, y_0:int, sigma: float, dtype = np.float32) -> float:
        x_ = (x - x_0) ** 2 / 2 / sigma / sigma
        y_ = (y - y_0) ** 2 / 2 / sigma / sigma 
        return np.exp(- (x_ + y_))

    @staticmethod
    def mask_gauss(sigma: float, size:int, dtype = np.float32) -> NDArray:
        if size % 2 == 0:
            raise ValueError()
        
        offset = size //2
        res = np.empty([size, size], dtype= dtype)
        for y in range(size):
            for x in range(size):
                res[y, x] = GaussianMask._gauss_function(x, y, offset, offset, sigma, dtype)
        
        return res/res.sum()
    
class FilterImage:
    @staticmethod
    def create_img(size:int = 100, value = 1, dtype=np.uint8) -> NDArray:
        if size % 2 == 0:
            raise ValueError()
        
        res = np.zeros([size, size], dtype=dtype)
        middle = (size + 1) // 2
        res[middle, middle] = value
        return res

    @staticmethod
    def img_conv_kernel(img: NDArray, kernel: NDArray) -> NDArray:
        return Convolution_Correlation.convolution(img, kernel)
    
    @staticmethod
    def img_corr_kernel(img: NDArray, kernel: NDArray) -> NDArray:
        return Convolution_Correlation.correlation(img, kernel)
    
    @staticmethod
    def kernel_conv_img(img: NDArray, kernel: NDArray) -> NDArray:
        return Convolution_Correlation.convolution(kernel, img)
    
    @staticmethod
    def kernel_corr_img(img: NDArray, kernel: NDArray) -> NDArray:
        return Convolution_Correlation.correlation(kernel, img)
    
class GaussianPyramid:
    @staticmethod
    def _one_loop(img: NDArray, kernel:NDArray) -> NDArray:
        res = Convolution_Correlation.convolution(img, kernel)
        return res[::2, ::2]

    @staticmethod
    def gaussian_pyramid(img: NDArray, size: int, sigma:float, time: int) -> Sequence[NDArray]:
        kernel = GaussianMask.mask_gauss(sigma, size)
        res = [img]
        last = img
        for i in range(time):
            last = GaussianPyramid._one_loop(last, kernel)
            res.append(last)
        return res

class FourierGaussianFilter:
    @staticmethod
    def gaussian_fft(img: NDArray, kernel: NDArray) -> NDArray:
        h, w = img.shape
        kh, kw = kernel.shape

        padded_kernel = np.zeros_like(img)
        padded_kernel[:kh, :kw] = kernel
        padded_kernel = np.roll(padded_kernel, -kh//2, axis=0)
        padded_kernel = np.roll(padded_kernel, -kw//2, axis=1)

        f_img = np.fft.fft2(img)
        f_kernel = np.fft.fft2(padded_kernel)

        return np.real(np.fft.ifft2(f_img * f_kernel))


    @staticmethod
    def test():
        img = FilterImage.create_img(size=101, value=1).astype(np.float32)
        kernel = GaussianMask.mask_gauss(sigma=5, size=21)

        # spatial convolution
        spatial_conv = FilterImage.img_conv_kernel(img, kernel)

        # spatial correlation
        spatial_corr = FilterImage.img_corr_kernel(img, kernel)

        # FFT filtering
        fft_result = FourierGaussianFilter.gaussian_fft(img, kernel)

        # crop FFT result to match spatial output size
        h, w = spatial_conv.shape
        fft_cropped = fft_result[:h, :w]

        # compare
        diff_conv_fft = np.mean(np.abs(spatial_conv - fft_cropped))
        diff_corr_fft = np.mean(np.abs(spatial_corr - fft_cropped))
        diff_conv_corr = np.mean(np.abs(spatial_conv - spatial_corr))

        print("Mean |Conv - FFT| :", diff_conv_fft)
        print("Mean |Corr - FFT| :", diff_corr_fft)
        print("Mean |Conv - Corr|:", diff_conv_corr)
    
    @staticmethod
    def high_pass_gaussian(sigma: float, size: int) -> NDArray:
        gauss = GaussianMask.mask_gauss(sigma, size)
        delta = np.zeros_like(gauss)
        center = size // 2
        delta[center, center] = 1.0
        return delta - gauss

    @staticmethod
    def band_pass_gaussian(sigma_low: float, sigma_high: float, size: int) -> NDArray:
        g_low = GaussianMask.mask_gauss(sigma_low, size)
        g_high = GaussianMask.mask_gauss(sigma_high, size)
        return g_low - g_high

class SteerabkeGaussian:
    @staticmethod
    def _gaussian_derivative_basis(size: int, sigma: float) -> Tuple[NDArray, NDArray]:
        assert size % 2 == 1
        gauss = GaussianMask.mask_gauss(sigma, size)
        
        h, w = gauss.shape
        gauss_x = np.empty_like(gauss)
        gauss_y = np.empty_like(gauss)

        for y in range(h):
            for x in range(w):
                cx = cy = size // 2
                gauss_x[y, x] = -(x - cx) / sigma**2 * gauss[y, x]
                gauss_y[y, x] = -(y - cy) / sigma**2 * gauss[y, x]

        return gauss_x, gauss_y

    @staticmethod
    def steerable(image: NDArray, theta: float,
              size: int = 21, sigma: float = 3.0) -> Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]]:

        Gx, Gy = SteerabkeGaussian._gaussian_derivative_basis(size, sigma)
        G_theta = np.cos(theta) * Gx + np.sin(theta) * Gy
        residual = Convolution_Correlation.correlation(image, G_theta)
        basis = (Gx, Gy)
        return residual, G_theta, basis
