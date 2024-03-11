import numpy as np
import scipy
import os
import scipy.optimize


from PIL import Image, ImageDraw, ImageEnhance

def get_centroids_from_image(image, sigma=2, image_th=None, crop=None, downsample=None,
                             filtsize=25, bg_sub_mode='local_mean', sigma_mode='global_root_square',
                             binary_open=True, centroid_window=None, max_area=100, min_area=5,
                             max_sum=None, min_sum=None, max_axis_ratio=None, max_returned=None,
                             return_moments=False, return_images=True):
    """Extract spot centroids from an image and calculate statistics.

    This is a versatile function for finding spots (e.g. stars or satellites) in an image and
    calculating/filtering their positions (centroids) and statistics (e.g. sum, area, shape).

    The coordinates start at the top/left edge of the pixel, i.e. x=y=0.5 is the centre of the
    top-left pixel. To convert the results to integer pixel indices use the floor operator.

    To aid in finding optimal settings pass `return_images=True` to get back a dictionary with
    partial extraction results and tweak the parameters accordingly. The dictionary entry
    `binary_mask` shows the result of the raw star detection and `final_centroids` labels the
    centroids in the original image (green for accepted, red for rejected).

    Technically, the best extraction is attained with `bg_sub_mode='local_median'` and
    `sigma_mode='local_median_abs'` with a reasonable (e.g. 15) size filter and a very sharp image.
    However, this may be slow (especially for larger filter sizes) and requires that the camera
    readout bit-depth is sufficient to accurately capture the camera noise. A recommendable and
    much faster alternative is `bg_sub_mode='local_mean'` and `sigma_mode='global_root_square'`
    with a larger (e.g. 25 or more) sized filter, which is the default. You may elect to do
    background subtraction and image thresholding by your own methods, then pass `bg_sub_mode=None`
    and your threshold as `image_th` to bypass these extraction steps.

    The algorithm proceeds as follows:
        1. Convert image to 2D numpy.ndarray with type float32.
        2. Call :meth:`tetra3.crop_and_downsample_image` with the image and supplied arguments
           `crop` and `downsample`.
        3. Subtract the background if `bg_sub_mode` is not None. Four methods are available:

           - 'local_median': Create the background image using a median filter of
             size `filtsize` and subtract pixelwise.
           - 'local_mean' (the default): Create the background image using a mean filter of size `filtsize` and
             subtract pixelwise.
           - 'global_median': Subtract the median value of all pixels from each pixel.
           - 'global_mean': Subtract the mean value of all pixels from each pixel.

        4. Calculate the image threshold if image_th is None. If image_th is defined this value
           will be used to threshold the image. The threshold is determined by calculating the
           noise standard deviation with the method selected as `sigma_mode` and then scaling it by
           `sigma` (default 3). The available methods are:

           - 'local_median_abs': For each pixel, calculate the standard deviation as
             the median of the absolute values in a region of size `filtsize` and scale by 1.48.
           - 'local_root_square': For each pixel, calculate the standard deviation as the square
             root of the mean of the square values in a region of size `filtsize`.
           - 'global_median_abs': Use the median of the absolute value of all pixels scaled by 1.48
             as the standard deviation.
           - 'global_root_square' (the default): Use the square root of the mean of the square of
             all pixels as the standard deviation.

        5. Create a binary mask using the image threshold. If `binary_open=True` (the default)
           apply a binary opening operation with a 3x3 cross as structuring element to clean up the
           mask.
        6. Label all regions (spots) in the binary mask.
        7. Calculate statistics on each region and reject it if it fails any of the max or min
           values passed. Calculated statistics are: area, sum, centroid (first moments) in x and
           y, second moments in xx, yy, and xy, major over minor axis ratio.
        8. Sort the regions, largest sum first, and keep at most `max_returned` if not None.
        9. If `centroid_window` is not None, recalculate the statistics using a square region of
           the supplied width (instead of the region from the binary mask).
        10. Undo the effects of cropping and downsampling by adding offsets/scaling the centroid
            positions to correspond to pixels in the original image.

    Args:
        image (PIL.Image): Image to find centroids in.
        sigma (float, optional): The number of noise standard deviations to threshold at.
            Default 2.
        image_th (float, optional): The value to threshold the image at. If supplied `sigma` and
            `simga_mode` will have no effect.
        crop (tuple, optional): Cropping to apply, see :meth:`tetra3.crop_and_downsample_image`.
        downsample (int, optional): Downsampling to apply, see
            :meth:`tetra3.crop_and_downsample_image`.
        filtsize (int, optional): Size of filter to use in local operations. Must be odd.
            Default 25.
        bg_sub_mode (str, optional): Background subtraction mode. Must be one of 'local_median',
            'local_mean' (the default), 'global_median', 'global_mean'.
        sigma_mode (str, optinal): Mode used to calculate noise standard deviation. Must be one of
            'local_median_abs', 'local_root_square', 'global_median_abs', or
            'global_root_square' (the default).
        binary_open (bool, optional): If True (the default), apply binary opening with 3x3 cross
           to thresholded binary mask.
        centroid_window (int, optional): If supplied, recalculate statistics using a square window
            of the supplied size.
        max_area (int, optional): Reject spots larger than this. Defaults to 100 pixels.
        min_area (int, optional): Reject spots smaller than this. Defaults to 5 pixels.
        max_sum (float, optional): Reject spots with a sum larger than this. Defaults to None.
        min_sum (float, optional): Reject spots with a sum smaller than this. Defaults to None.
        max_axis_ratio (float, optional): Reject spots with a ratio of major over minor axis larger
            than this. Defaults to None.
        max_returned (int, optional): Return at most this many spots. Defaults to None, which
            returns all spots. Will return in order of brightness (spot sum).
        return_moments (bool, optional): If set to True, return the calculated statistics (e.g.
            higher order moments, sum, area) together with the spot positions.
        return_images (bool, optional): If set to True, return a dictionary with partial results
            from the steps in the algorithm.

    Returns:
        numpy.ndarray or tuple: If `return_moments=False` and `return_images=False` (the defaults)
        an array of shape (N,2) is returned with centroid positions (y down, x right) of the
        found spots in order of brightness. If `return_moments=True` a tuple of numpy arrays
        is returned with: (N,2) centroid positions, N sum, N area, (N,3) xx yy and xy second
        moments, N major over minor axis ratio. If `return_images=True` a tuple is returned
        with the results as defined previously and a dictionary with images and data of partial
        results. The keys are: `converted_input`: The input after conversion to a mono float
        numpy array. `cropped_and_downsampled`: The image after cropping and downsampling.
        `removed_background`: The image after background subtraction. `binary_mask`: The
        thresholded image where raw stars are detected (after binary opening).
        `final_centroids`: The original image annotated with green circles for the extracted
        centroids, and red circles for any centroids that were rejected.
    """

    # 1. Ensure image is float np array and 2D:
    raw_image = image.copy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        assert image.shape[2] in (1, 3), 'Colour image must have 1 or 3 colour channels'
        if image.shape[2] == 3:
            # Convert to greyscale
            image = image[:, :, 0]*.299 + image[:, :, 1]*.587 + image[:, :, 2]*.114
        else:
            # Delete empty dimension
            image = image.squeeze(axis=2)
    else:
        assert image.ndim == 2, 'Image must be 2D or 3D array'
    if return_images:
        images_dict = {'converted_input': image.copy()}
    # 2 Crop and downsample
    (image, offs) = crop_and_downsample_image(image, crop=crop, downsample=downsample,
                                              return_offsets=True, sum_when_downsample=True)
    (height, width) = image.shape
    (offs_h, offs_w) = offs
    if return_images:
        images_dict['cropped_and_downsampled'] = image.copy()
    # 3. Subtract background:
    if bg_sub_mode is not None:
        if bg_sub_mode.lower() == 'local_median':
            assert filtsize is not None, \
                'Must define filter size for local median background subtraction'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            image = image - scipy.ndimage.filters.median_filter(image, size=filtsize,
                                                                output=image.dtype)
        elif bg_sub_mode.lower() == 'local_mean':
            assert filtsize is not None, \
                'Must define filter size for local median background subtraction'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            image = image - scipy.ndimage.uniform_filter(image, size=filtsize,
                                                                 output=image.dtype)
        elif bg_sub_mode.lower() == 'global_median':
            image = image - np.median(image)
        elif bg_sub_mode.lower() == 'global_mean':
            image = image - np.mean(image)
        else:
            raise AssertionError('bg_sub_mode must be string: local_median, local_mean,'
                                 + ' global_median, or global_mean')
    if return_images:
        images_dict['removed_background'] = image.copy()
    # 4. Find noise standard deviation to threshold unless a threshold is already defined!
    if image_th is None:
        assert sigma_mode is not None and isinstance(sigma_mode, str), \
            'Must define a sigma mode or image threshold'
        assert sigma is not None and isinstance(sigma, (int, float)), \
            'Must define sigma for thresholding (int or float)'
        if sigma_mode.lower() == 'local_median_abs':
            assert filtsize is not None, 'Must define filter size for local median sigma mode'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            img_std = scipy.ndimage.filters.median_filter(np.abs(image), size=filtsize,
                                                          output=image.dtype) * 1.48
        elif sigma_mode.lower() == 'local_root_square':
            assert filtsize is not None, 'Must define filter size for local median sigma mode'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            img_std = np.sqrt(scipy.ndimage.filters.uniform_filter(image**2, size=filtsize,
                                                                   output=image.dtype))
        elif sigma_mode.lower() == 'global_median_abs':
            img_std = np.median(np.abs(image)) * 1.48
        elif sigma_mode.lower() == 'global_root_square':
            img_std = np.sqrt(np.mean(image**2))
        else:
            raise AssertionError('sigma_mode must be string: local_median_abs, local_root_square,'
                                 + ' global_median_abs, or global_root_square')
        image_th = img_std * sigma
    if return_images:
       images_dict['image_threshold'] = image_th
    # 5. Threshold to find binary mask
    bin_mask = image > image_th
    if binary_open:
        bin_mask = scipy.ndimage.binary_opening(bin_mask)
    if return_images:
        images_dict['binary_mask'] = bin_mask
    # 6. Label each region in the binary mask
    (labels, num_labels) = scipy.ndimage.label(bin_mask)
    index = np.arange(1, num_labels + 1)
    if return_images:
        images_dict['labelled_regions'] = labels
    if num_labels < 1:
        # Found nothing in binary image, return empty.
        if return_moments and return_images:
            return ((np.empty((0, 2)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 3)),
                     np.empty((0, 1))), images_dict)
        elif return_moments:
            return (np.empty((0, 2)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 3)),
                    np.empty((0, 1)))
        elif return_images:
            return (np.empty((0, 2)), images_dict)
        else:
            return np.empty((0, 2))

    # 7. Get statistics and threshold
    def calc_stats(a, p):
        """Calculates statistics for each labelled region:
        - Sum (zeroth moment)
        - Centroid y, x (first moment)
        - Variance xx, yy, xy (second moment)
        - Area (pixels)
        - Major axis/minor axis ratio
        First variable will be NAN if failed any of the checks
        """
        (y, x) = (np.unravel_index(p, (height, width)))
        area = len(a)
        centroid = np.sum([a, x*a, y*a], axis=-1)
        m0 = centroid[0]
        centroid[1:] = centroid[1:] / m0
        m1_x = centroid[1]
        m1_y = centroid[2]
        # Check basic filtering
        if min_area and area < min_area:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        if max_area and area > max_area:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        if min_sum and m0 < min_sum:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        if max_sum and m0 > max_sum:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        # If higher order data is requested or used for filtering, calculate.
        if return_moments or max_axis_ratio is not None:
            # Need to calculate second order data about the regions, firstly the moments
            # then use that to get major/minor axes.
            m2_xx = max(0, np.sum((x - m1_x)**2 * a) / m0)
            m2_yy = max(0, np.sum((y - m1_y)**2 * a) / m0)
            m2_xy = np.sum((x - m1_x) * (y - m1_y) * a) / m0
            major = np.sqrt(2 * (m2_xx + m2_yy + np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))
            minor = np.sqrt(2 * max(0, m2_xx + m2_yy - np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))
            if max_axis_ratio and minor <= 0:
                return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
            axis_ratio = major / max(minor, .000000001)
            if max_axis_ratio and axis_ratio > max_axis_ratio:
                return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
            return (m0, m1_y+.5, m1_x+.5, m2_xx, m2_yy, m2_xy, area, axis_ratio)
        else:
            return (m0, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, area, np.nan)

    tmp = scipy.ndimage.labeled_comprehension(image, labels, index, calc_stats, '8f', None,
                                              pass_positions=True)
    valid = ~np.isnan(tmp[:, 0])
    extracted = tmp[valid, :]
    rejected = tmp[~valid, :]
    if return_images:
        # Convert 16-bit to 8-bit:
        if raw_image.mode == 'I;16':
            tmp = np.array(raw_image, dtype=np.uint16)
            tmp //= 256
            tmp = tmp.astype(np.uint8)
            raw_image = Image.fromarray(tmp)
        # Convert mono to RGB
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')
        # Draw green circles for kept centroids, red for rejected
        img_draw = ImageDraw.Draw(raw_image)
        def draw_circle(centre, radius, **kwargs):
            bbox = [centre[1] - radius,
                    centre[0] - radius,
                    centre[1] + radius,
                    centre[0] + radius]
            img_draw.ellipse(bbox, **kwargs)
        for entry in extracted:
            pos = entry[1:3].copy()
            size = .01*width
            if downsample is not None:
                pos *= downsample
                pos += [offs_h, offs_w]
                size *= downsample
            draw_circle(pos, size, outline='green')
        for entry in rejected:
            pos = entry[1:3].copy()
            size = .01*width
            if downsample is not None:
                pos *= downsample
                pos += [offs_h, offs_w]
                size *= downsample
            draw_circle(pos, size, outline='red')
        images_dict['final_centroids'] = raw_image

    # 8. Sort
    order = (-extracted[:, 0]).argsort()
    if max_returned:
        order = order[:max_returned]
    extracted = extracted[order, :]

    # 9. If desired, redo centroiding with traditional window
    if centroid_window is not None:
        if centroid_window > min(height, width):
            centroid_window = min(height, width)
        for i in range(extracted.shape[0]):
            c_x = int(np.floor(extracted[i, 2]))
            c_y = int(np.floor(extracted[i, 1]))
            offs_x = c_x - centroid_window // 2
            offs_y = c_y - centroid_window // 2
            if offs_y < 0:
                offs_y = 0
            if offs_y > height - centroid_window:
                offs_y = height - centroid_window
            if offs_x < 0:
                offs_x = 0
            if offs_x > width - centroid_window:
                offs_x = width - centroid_window
            img_cent = image[offs_y:offs_y + centroid_window, offs_x:offs_x + centroid_window]
            img_sum = np.sum(img_cent)
            (xx, yy) = np.meshgrid(np.arange(centroid_window) + .5,
                                   np.arange(centroid_window) + .5)
            xc = np.sum(img_cent * xx) / img_sum
            yc = np.sum(img_cent * yy) / img_sum
            extracted[i, 1:3] = np.array([yc, xc]) + [offs_y, offs_x]

    # 10. Revert effects of crop and downsample
    if downsample:
        extracted[:, 1:3] = extracted[:, 1:3] * downsample  # Scale centroid
    if crop:
        extracted[:, 1:3] = extracted[:, 1:3] + np.array([offs_h, offs_w])  # Offset centroid
    # Return results, default just the centroids 
    if not any((return_moments, return_images)):
        return extracted[:, 1:3]
    # Otherwise, build list of requested returned items
    result = [extracted[:, 1:3]]
    if return_moments:
        result.append([extracted[:, 0], extracted[:, 6], extracted[:, 3:6],
                extracted[:, 7]])
    if return_images:
        result.append(images_dict)
    return tuple(result)


def crop_and_downsample_image(image, crop=None, downsample=None, sum_when_downsample=True,
                              return_offsets=False):
    """Crop and/or downsample an image. Cropping is applied before downsampling.

    Args:
        image (numpy.ndarray): The image to crop and downsample. Must be 2D.
        crop (int or tuple, optional): Desired cropping of the image. May be defined in three ways:

            - Scalar: Image is cropped to given fraction (e.g. crop=2 gives 1/2 size image out).
            - 2-tuple: Image is cropped to centered region with size crop = (height, width).
            - 4-tuple: Image is cropped to region with size crop[0:2] = (height, width), offset
              from the centre by crop[2:4] = (offset_down, offset_right).

        downsample (int, optional): Downsampling factor, e.g. downsample=2 will combine 2x2 pixel
            regions into one. The image width and height must be divisible by this factor.
        sum_when_downsample (bool, optional): If True (the default) downsampled pixels are
            calculated by summing the original pixel values. If False the mean is used.
        return_offsets (bool, optional): If set to True, the applied cropping offset from the top
            left corner is returned.
    Returns:
        numpy.ndarray or tuple: If `return_offsets=False` (the default) a 2D array with the cropped
        and dowsampled image is returned. If `return_offsets=True` is passed a tuple containing
        the image and a tuple with the cropping offsets (top, left) is returned.
    """
    # Input must be 2-d numpy array
    # Crop can be either a scalar, 2-tuple, or 4-tuple:
    # Scalar: Image is cropped to given fraction (eg input crop=2 gives 1/2 size image out)
    # If 2-tuple: Image is cropped to center region with size crop = (height, width)
    # If 4-tuple: Image is cropped to ROI with size crop[0:1] = (height, width)
    #             offset from centre by crop[2:3] = (offset_down, offset_right)
    # Downsample is made by summing regions of downsample by downsample pixels.
    # To get the mean set sum_when_downsample=False.
    # Returned array is same type as input array!

    image = np.asarray(image)
    assert image.ndim == 2, 'Input must be 2D'
    # Do nothing if both are None
    if crop is None and downsample is None:
        if return_offsets is True:
            return (image, (0, 0))
        else:
            return image
    full_height, full_width = image.shape
    # Check if input is integer type (and therefore can overflow...)
    if np.issubdtype(image.dtype, np.integer):
        intype = image.dtype
    else:
        intype = None
    # Crop:
    if crop is not None:
        try:
            # Make crop into list of int
            crop = [int(x) for x in crop]
            if len(crop) == 2:
                crop = crop + [0, 0]
            elif len(crop) == 4:
                pass
            else:
                raise ValueError('Length of crop must be 2 or 4 if iterable, not '
                                 + str(len(crop)) + '.')
        except TypeError:
            # Could not make list (i.e. not iterable input), crop to portion
            crop = int(crop)
            assert crop > 0, 'Crop must be greater than zero if scalar.'
            assert full_height % crop == 0 and full_width % crop == 0,\
                'Crop must be divisor of image height and width if scalar.'
            crop = [full_height // crop, full_width // crop, 0, 0]
        # Calculate new height and width (making sure divisible with future downsampling)
        divisor = downsample if downsample is not None else 2
        height = int(np.ceil(crop[0]/divisor)*divisor)
        width = int(np.ceil(crop[1]/divisor)*divisor)
        # Clamp at original size
        if height > full_height:
            height = full_height
        if width > full_width:
            width = full_width
        # Calculate offsets from centre
        offs_h = int(round(crop[2] + (full_height - height)/2))
        offs_w = int(round(crop[3] + (full_width - width)/2))
        # Clamp to be inside original image
        if offs_h < 0:
            offs_h = 0
        if offs_h > full_height-height:
            offs_h = full_height-height
        if offs_w < 0:
            offs_w = 0
        if offs_w > full_width-width:
            offs_w = full_width-width
        # Do the cropping
        image = image[offs_h:offs_h+height, offs_w:offs_w+width]
    else:
        offs_h = 0
        offs_w = 0
        height = full_height
        width = full_width
    # Downsample:
    if downsample is not None:
        assert height % downsample == 0 and width % downsample == 0,\
            '(Cropped) image must be divisible by downsampling factor'
        if intype is not None:
            # Convert integer types into float for summing without overflow risk
            image = image.astype(np.float32)
        if sum_when_downsample is True:
            image = image.reshape((height//downsample, downsample, width//downsample,
                                   downsample)).sum(axis=-1).sum(axis=1)
        else:
            image = image.reshape((height//downsample, downsample, width//downsample,
                                   downsample)).mean(axis=-1).mean(axis=1)
        if intype is not None:
            # Convert back with clipping
            image = image.clip(np.iinfo(intype).min, np.iinfo(intype).max).astype(intype)
    # Return image and if desired the offset.

    if return_offsets is True:
        return (image, (offs_h, offs_w))
    else:
        return image
    
current_dir = os.getcwd()
image_path = os.path.join(current_dir, './plot.png')  # Join current directory with path
image = Image.open(image_path)
# image = image.convert("RGB")  # Discards the 4th channel
centr_data = get_centroids_from_image(image)
# print(centr_data)
if isinstance(centr_data, tuple):
            centroids = centr_data[0]
else:
            centroids = centr_data
print('Found ' + str(len(centroids)) + ' centroids.')
print(centroids)

labelled_regions = centr_data[1]['labelled_regions']  # Extract labelled regions

# Now you can use the labelled_regions array for further processing or visualization
print(labelled_regions)

def overlay_spots(original_image, labelled_regions, alpha=0.5):
  """
  Overlays the labelled_regions array on the original image, highlighting spots.

  Args:
      original_image: PIL Image object of the original image.
      labelled_regions: 2D boolean array indicating potential spots.
      alpha: Transparency level for the overlay (0.0 to 1.0).

  Returns:
      A PIL Image object with the overlay.
  """

  # Convert labelled_regions to a grayscale image (optional)
  # grayscale_image = labelled_regions * 255

  # Create a mask image from labelled_regions (optional)
  mask_image = Image.fromarray(labelled_regions * 255).convert('L')  # Convert to grayscale

  # Adjust contrast of mask for better highlighting (optional)
  mask_enhancer = ImageEnhance.Contrast(mask_image)
  mask_image = mask_enhancer.enhance(2.0)  # Adjust contrast as desired

  # Create a partially transparent overlay image
  overlay_image = mask_image.convert('RGBA')
  overlay_image.putalpha(int(alpha * 255))

  # Overlay the mask image on the original image with transparency
  original_image = original_image.convert('RGBA')
  return Image.alpha_composite(original_image, overlay_image)

# Assuming you have the original image (original_image) and labelled_regions array
overlayed_image = overlay_spots(image, labelled_regions)

save_path = "spots.png"
overlayed_image.save(save_path)
# Display the overlaid image
overlayed_image.show()


final_centroids = centr_data[1]['final_centroids']  # Extract final_centroids



# Overlay the image (assuming image has alpha channel for transparency)
final_image = Image.alpha_composite(overlayed_image, final_centroids)

# Display the overlaid image (optional)
final_image.show()

# Save the result
final_image.save("result.png")  # Replace with your desired filename