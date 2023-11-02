import numpy as np

from skimage.color import rgb2gray
import matplotlib
from matplotlib import pyplot
import imageio
import copy


RGB = 2
GRAYSCALE = 1
RGB_DIM = 3
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    matrix_picture = imageio.imread(filename).astype('float64')
    if representation == RGB:
        return np.divide(matrix_picture, 255)

    return np.divide(rgb2gray(matrix_picture), 255)


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    picture_matrix = read_image(filename, representation)
    pyplot.imshow(picture_matrix, cmap=matplotlib.cm.gray,vmin=0,vmax=1)
    pyplot.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """

    im_size= imRGB.shape
    YIQ = np.zeros(im_size)
    red = imRGB[:, :, 0]
    green = imRGB[:, :, 1]
    blue = imRGB[:, :, 2]
    channels = np.arange(3)
    for channel in channels:
        YIQ[:, :, channel] = red * RGB_YIQ_TRANSFORMATION_MATRIX[channel][0] + \
                             green * RGB_YIQ_TRANSFORMATION_MATRIX[channel][
                                 1] + \
                             blue * RGB_YIQ_TRANSFORMATION_MATRIX[channel][2]

    return YIQ.astype('float64')



def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """

    inv_mat = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)

    im_size = imYIQ.shape
    RGB = np.zeros(im_size)
    red = imYIQ[:, :, 0]
    green = imYIQ[:, :, 1]
    blue = imYIQ[:, :, 2]
    channels = np.arange(3)
    for channel in channels:
        RGB[:, :, channel] = red * inv_mat[channel][0] + \
                             green * inv_mat[channel][1] + \
                             blue * inv_mat[channel][2]

    return RGB.astype('float64')

def rgb2y(im_orig):
    """
    Transform a RGB color space into the Y  channel
    :param im_orig: the original image
    :return:  YIQ  image and the T  channel
    """
    yiq_mat = rgb2yiq(im_orig)
    pic = np.multiply(yiq_mat[:, :, 0], 255)
    return  yiq_mat, pic

def y2rgb(im_orig,im_eq_y,yiq_mat):
    """
    Transform a Y channel into the  RGB color space
    :param im_orig: the original image
    :param im_eq_y: equalaized Y channel
    :param yiq_mat: the original YIQ image
    :return:equalized image
    """
    im_size = im_orig.shape
    new_yiq = np.zeros(im_size)
    new_yiq[:, :, 0] = im_eq_y
    new_yiq[:, :, 1] = yiq_mat[:, :, 1]
    new_yiq[:, :, 2] = yiq_mat[:, :, 2]
    im_eq = yiq2rgb(new_yiq)
    return im_eq



def transition_array(n_quant,q,z):
    """
    This function makes an array, which contains between [z[i],z[i+1]]
    indexes, the corresponding q[i]
    :param n_quant: number of quants
    :param q: q array
    :param z: z array
    :return:  array with all ranges
    """
    t_array = np.zeros(256)
    z = z.astype(int)
    for i in range(n_quant):
        intensity = q[i]
        first_bound = z[i]
        second_bound = z[i + 1]
        t_array[first_bound + 1: second_bound + 1] = intensity
    return t_array

def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """


    pic = np.multiply(im_orig, 255)
    if im_orig.ndim == RGB_DIM:
        yiq_mat,pic = rgb2y(im_orig)


    hist, bin_edges = np.histogram(pic.astype(int), bins=256,range=(0,255))


    comulative_array = np.cumsum(hist)


    pixels_num = comulative_array[255]
    temp_array = comulative_array[comulative_array != 0]
    c_m = temp_array.min()
    normalized_hist = 255*((comulative_array - c_m) / (pixels_num - c_m))

    t_array = np.round(normalized_hist)

    equalized_image = t_array[pic.astype(int)]
    im_eq = np.divide(equalized_image, 255)
    if im_orig.ndim == RGB_DIM:
        im_eq = y2rgb(im_orig,im_eq,yiq_mat)

    hist_eq, bin_hist = np.histogram(equalized_image, bins=256,range=(0,255))

    return [im_eq.astype('float64'), hist, hist_eq]



def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """

    pic = np.multiply(im_orig, 255)
    if im_orig.ndim == RGB_DIM:
        yiq_mat, pic = rgb2y(im_orig)
    error =[]
    pixels_num = pic.shape[0]*pic.shape[1]
    pixels_in_segment = pixels_num/n_quant

    hist, bounds = np.histogram(pic.astype(int), bins=256,range=(0,255))
    cumsum_array = np.cumsum(hist)

    z = np.zeros(n_quant+1)
    for j in range(1,n_quant):
        ndarray = np.where(cumsum_array>=(pixels_in_segment*j))
        z[j] =(ndarray[0][0])


    z[0] = -1
    z[n_quant] = 255
    q = ((z[:-1] + z[1:])/2)

    for  iter in range(n_iter):

        err =0
        temp_z_array = copy.deepcopy(z)
        for quant in range(1,n_quant+2):
            if quant!= n_quant+1:
                sigma_arange = np.arange(np.floor(z[quant-1])+1,np.floor(z[quant])+1).astype(int)
                numerator = np.sum(np.multiply(sigma_arange,hist[sigma_arange]))
                divider = np.sum(hist[sigma_arange])
                q[quant-1] =numerator/divider
            if (quant-1!=0) and (quant!= n_quant+1) :
                z[quant-1] = (q[quant-2] + q[quant-1]) / 2
            if quant != n_quant + 1:
                err += np.sum(np.multiply(np.power((q[quant-1]-sigma_arange),2),hist[sigma_arange]))


        if np.array_equal(temp_z_array ,z):
            break
        error.append(err)


    t_array = transition_array(n_quant,q,z)
    quantized_image = t_array[pic.astype(int)]
    im_quant = np.divide(quantized_image, 255)
    if im_orig.ndim == RGB_DIM:
        im_quant = y2rgb(im_orig, im_quant, yiq_mat)

    im_quant = im_quant.astype('float64')

    return[im_quant, np.array(error)]










from sklearn.cluster import KMeans

def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    im_orig = np.multiply(im_orig, 255)
    size = im_orig.shape
    flatten_img = im_orig.reshape(size[0] * size[1], size[2])
    kmeans = KMeans(n_clusters=n_quant)
    clusters = kmeans.fit_predict(flatten_img)
    centroid = kmeans.cluster_centers_
    dimensions = (size[0], size[1], size[2])
    return np.divide((np.reshape(centroid.astype(int)[clusters], dimensions)), 255)








