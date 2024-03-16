import cv2
import random
import torch
import numpy as np


def add_blur(image, x, y, hw, fog_coeff):
    overlay = image.copy()
    output = image.copy()
    alpha = 0.08 * fog_coeff
    rad = hw // 2
    point = (x + hw // 2, y + hw // 2)
    cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def generate_random_blur_coordinates(imshape, hw):
    blur_points = []
    midx = imshape[1] // 2 - 2 * hw
    midy = imshape[0] // 2 - hw
    index = 1
    while midx > -hw or midy > -hw:
        for i in range(hw // 10 * index):
            x = np.random.randint(midx, imshape[1] - midx - hw)
            y = np.random.randint(midy, imshape[0] - midy - hw)
            blur_points.append((x, y))
        midx -= 3 * hw * imshape[1] // sum(imshape)
        midy -= 3 * hw * imshape[0] // sum(imshape)
        index += 1
    return blur_points


def add_fog(image, fog_coeff=-1):
    assert fog_coeff == -1 or 0 < fog_coeff < 1, "Fog coeff can only be between 0 and 1"
    imshape = image.shape
    fog_coeff_t = random.uniform(0.3, 1) if fog_coeff == -1 else fog_coeff
    hw = int(imshape[1] / 3 * fog_coeff_t)
    haze_list = generate_random_blur_coordinates(imshape, hw)
    for haze_points in haze_list:
        image = add_blur(image, haze_points[0], haze_points[1], hw, fog_coeff_t)
    image = cv2.blur(image, (hw // 10, hw // 10))
    image_RGB = image

    return image_RGB


def generate_random_lines(imshape, slant, drop_length, rain_type):
    drops = []
    area = imshape[0] * imshape[1]
    no_of_drops = area // 600

    if rain_type.lower() == "drizzle":
        no_of_drops = area // 770
        drop_length = 10
    elif rain_type.lower() == "heavy":
        drop_length = 30
    elif rain_type.lower() == "torrential":
        no_of_drops = area // 500
        drop_length = 60

    for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops, drop_length


def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops):
    imshape = image.shape
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color, drop_width)
    image = cv2.blur(image_t, (7, 7))  ## rainy view are blurry
    brightness_coefficient = 0.7  ## rainy days are usually shady
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * brightness_coefficient  ## scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


##rain_type='drizzle','heavy','torrential'
def add_rain(image, slant=-1, drop_length=20, drop_width=1, drop_color=(200, 200, 200), rain_type="None"):  ## (200,200,200) a shade of gray
    assert slant == -1 or -20 <= slant <= 20
    assert 1 <= drop_width <= 5
    assert 0 <= drop_length <= 100

    imshape = image.shape
    slant_extreme = slant
    slant = np.random.randint(-10, 10) if slant == -1 else slant  # generate random slant if no slant value is given
    rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, rain_type)
    output = rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops)
    image_RGB = output
    return image_RGB


def snow_process(image, snow_coeff):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype=np.float64)
    brightness_coefficient = 2.5
    imshape = image.shape
    snow_point = snow_coeff  ## increase this for more snow
    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] = (
        image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] * brightness_coefficient
    )  ## scale pixel values up for channel 1(Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255  ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB


def add_snow(image, snow_coeff=-1):
    assert snow_coeff == -1 or 0 <= snow_coeff <= 1
    snow_coeff = random.uniform(0, 1) if snow_coeff == -1 else snow_coeff
    snow_coeff *= 255 / 2
    snow_coeff += 255 / 3
    output = snow_process(image, snow_coeff)
    image_RGB = output

    return image_RGB


def custom_noise_func(images):
    noise = []
    for image in torch.clip_(images.permute([0, 2, 3, 1]) * 127.5 + 127.5, 0, 255):
        cur_noise = image.numpy().astype("uint8")
        if random.random() > 0.6:
            cur_noise = add_fog(cur_noise)
        elif random.random() > 0.3:
            cur_noise = add_snow(cur_noise)
        else:
            cur_noise = add_rain(cur_noise)
        noise.append(torch.from_numpy(cur_noise))
    return torch.stack(noise).permute([0, 3, 1, 2]).float() / 127.5 - 1
