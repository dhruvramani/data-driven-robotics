import math
import numbers
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE : Source - RAD (https://github.com/MishaLaskin/rad)
# TODO : Will test later

def random_crop(images, out=192):
    '''
        + Arguments:
            - images: np.array shape (N,C,H,W)
            - out: output size (e.g. 84)
    '''
    n, c, h, w = images.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=images.dtype)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    
    return np.reshape(cropped, (n, out, out, c))

def grayscale(images):
    '''
        + Arguments:
            - images: np.array shape (B,C,H,W)
    '''
    images = torch.from_numpy(images)
    device = images.device
    b, c, h, w = images.shape
    frames = c // 3
    
    images = images.view([b, frames, 3, h, w])
    images = images[:, :, 0, ...] * 0.2989 + images[:, :, 1, ...] * 0.587 + images[:, :, 2, ...] * 0.114 
    
    images = images.type(torch.uint8).float()
    # assert len(images.shape) == 3, images.shape
    images = images[:, :, None, :, :]
    images = images.numpy() * np.ones([1, 1, 3, 1, 1])
    return np.reshape(images, (b, h, w, 1))

def random_grayscale(images, probab=0.3):
    '''
        + Arguments:
            - images: np.array shape (B,C,H,W)
            - probab 
    '''
    images = torch.from_numpy(images)
    device = images.device
    in_type = images.type()
    images = images * 255.
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= probab
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.
    return np.reshape(out.numpy(), (bs, h, w, -1)) 

def random_cutout(images, min_cut=10, max_cut=30):
    '''
        + Arguments:
            - images: np.array shape (B,C,H,W)
            - min / max cut: int, min / max size of cutout 
    '''

    n, c, h, w = images.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=images.dtype)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        cutouts[i] = cut_img
    return np.reshape(cutouts, (n, h, w, c))

def random_cutout_color(images, min_cut=10, max_cut=30):
    '''
        + Arguments:
            - images: np.array shape (N,C,H,W)
            - out: output size (e.g. 84)
    '''
    n, c, h, w = images.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=images.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cut_img = img.copy()
        
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
            rand_box[i].reshape(-1,1,1),                                                
            (1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])
        
        cutouts[i] = cut_img
    return np.reshape(cutouts, (n, h, w, c))

def random_flip(images, probab=0.2):
    '''
        + Arguments:
            - images: np.array shape (B,C,H,W)
            - probab 
    '''
    images = torch.from_numpy(images)
    device = images.device
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    flipped_images = images.flip([3])
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= probab
    mask = torch.from_numpy(mask)
    frames = images.shape[1] #// 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]
    
    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, h, w, -1])
    return out.numpy()

def random_rotation(images, probab=0.3):
    '''
        + Arguments:
            - images: np.array shape (B,C,H,W)
            - probab 
    '''
    images = torch.from_numpy(images)
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    rot90_images = images.rot90(1,[2,3])
    rot180_images = images.rot90(2,[2,3])
    rot270_images = images.rot90(3,[2,3])    
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= probab
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)
    
    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i,m in enumerate(masks):
        m[torch.where(mask==i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
        m = m[:,:,None,None]
        masks[i] = m
    
    
    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([bs, h, w, -1])
    return out.numpy()

def random_convolution(images):
    '''
        + Arguments:
            - images: np.array shape (B,C,H,W)
    '''
    images = torch.from_numpy(images)
    _device = images.device
    
    img_h, img_w = images.shape[2], images.shape[3]
    num_stack_channel = images.shape[1]
    num_batch = images.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)
    
    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
    
    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_images = images[trans_index*batch_size:(trans_index+1)*batch_size]
        temp_images = temp_images.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_images)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return np.reshape(total_out.numpy(), (num_batch, img_h, img_w, num_stack_channel))

def no_aug(images):
    return images

def random_color_jitter(images):
    '''
        + Arguments:
            - images: np.array shape (B,C,H,W)
    '''
    b,c,h,w = images.shape
    images = images.view(-1,3,h,w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5, p=1.0, batch_size=128))
    images = transform_module(images).view(b, h, w, c)
    return images

# ------------------------------------------------------------------------------------------------------------------------

def rgb2hsv(rgb, eps=1e-8):
    # Reference: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = rgb.device
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3])).to(_device)
    hue[Cmax== r] = (((g - b)/(delta + eps)) % 6)[Cmax == r]
    hue[Cmax == g] = ((b - r)/(delta + eps) + 2)[Cmax == g]
    hue[Cmax == b] = ((r - g)/(delta + eps) + 4)[Cmax == b]
    hue[Cmax == 0] = 0.0
    hue = hue / 6. # making hue range as [0, 1.0)
    hue = hue.unsqueeze(dim=1)

    saturation = (delta) / (Cmax + eps)
    saturation[Cmax == 0.] = 0.
    saturation = saturation.to(_device)
    saturation = saturation.unsqueeze(dim=1)

    value = Cmax
    value = value.to(_device)
    value = value.unsqueeze(dim=1)

    return torch.cat((hue, saturation, value), dim=1)#.type(torch.FloatTensor).to(_device)
    # return hue, saturation, value

def hsv2rgb(hsv):
    # Reference: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = hsv.device

    hsv = torch.clamp(hsv, 0, 1)
    hue = hsv[:, 0, :, :] * 360.
    saturation = hsv[:, 1, :, :]
    value = hsv[:, 2, :, :]

    c = value * saturation
    x = - c * (torch.abs((hue / 60.) % 2 - 1) - 1)
    m = (value - c).unsqueeze(dim=1)

    rgb_prime = torch.zeros_like(hsv).to(_device)

    inds = (hue < 60) * (hue >= 0)
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb_prime[:, 1, :, :][inds] = x[inds]

    inds = (hue < 120) * (hue >= 60)
    rgb_prime[:, 0, :, :][inds] = x[inds]
    rgb_prime[:, 1, :, :][inds] = c[inds]

    inds = (hue < 180) * (hue >= 120)
    rgb_prime[:, 1, :, :][inds] = c[inds]
    rgb_prime[:, 2, :, :][inds] = x[inds]

    inds = (hue < 240) * (hue >= 180)
    rgb_prime[:, 1, :, :][inds] = x[inds]
    rgb_prime[:, 2, :, :][inds] = c[inds]

    inds = (hue < 300) * (hue >= 240)
    rgb_prime[:, 2, :, :][inds] = c[inds]
    rgb_prime[:, 0, :, :][inds] = x[inds]

    inds = (hue < 360) * (hue >= 300)
    rgb_prime[:, 2, :, :][inds] = x[inds]
    rgb_prime[:, 0, :, :][inds] = c[inds]

    rgb = rgb_prime + torch.cat((m, m, m), dim=1)
    rgb = rgb.to(_device)

    return torch.clamp(rgb, 0, 1)
    
class ColorJitterLayer(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0, batch_size=128, stack_size=3):
        super(ColorJitterLayer, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.prob = p
        self.batch_size = batch_size
        self.stack_size = stack_size
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    
    def adjust_contrast(self, x):
        """
            Args:
                x: torch tensor img (rgb type)
            Factor: torch tensor with same length as x
                    0 gives gray solid image, 1 gives original image,
            Returns:
                torch tensor image: Brightness adjusted
        """
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.contrast)
        factor = factor.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        return torch.clamp((x - means)
                           * factor.view(len(x), 1, 1, 1) + means, 0, 1)
    
    def adjust_hue(self, x):
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.hue)
        factor = factor.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        h = x[:, 0, :, :]
        h += (factor.view(len(x), 1, 1) * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        return x
    
    def adjust_brightness(self, x):
        """
            Args:
                x: torch tensor img (hsv type)
            Factor:
                torch tensor with same length as x
                0 gives black image, 1 gives original image,
                2 gives the brightness factor of 2.
            Returns:
                torch tensor image: Brightness adjusted
        """
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.brightness)
        factor = factor.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        x[:, 2, :, :] = torch.clamp(x[:, 2, :, :]
                                     * factor.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)
    
    def adjust_saturate(self, x):
        """
            Args:
                x: torch tensor img (hsv type)
            Factor:
                torch tensor with same length as x
                0 gives black image and white, 1 gives original image,
                2 gives the brightness factor of 2.
            Returns:
                torch tensor image: Brightness adjusted
        """
        _device = x.device
        factor = torch.empty(self.batch_size, device=_device).uniform_(*self.saturation)
        factor = factor.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        x[:, 1, :, :] = torch.clamp(x[:, 1, :, :]
                                    * factor.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)
    
    def transform(self, inputs):
        hsv_transform_list = [rgb2hsv, self.adjust_brightness,
                              self.adjust_hue, self.adjust_saturate,
                              hsv2rgb]
        rgb_transform_list = [self.adjust_contrast]
        # Shuffle transform
        if random.uniform(0,1) >= 0.5:
            transform_list = rgb_transform_list + hsv_transform_list
        else:
            transform_list = hsv_transform_list + rgb_transform_list
        for t in transform_list:
            inputs = t(inputs)
        return inputs
    
    def forward(self, inputs):
        _device = inputs.device
        random_inds = np.random.choice(
            [True, False], len(inputs), p=[self.prob, 1 - self.prob])
        inds = torch.tensor(random_inds).to(_device)
        if random_inds.sum() > 0:
            inputs[inds] = self.transform(inputs[inds])
        return inputs


aug_to_func = {
    'crop': random_crop,
    'grayscale': random_grayscale,
    'cutout': random_cutout,
    'cutout_color': random_cutout_color,
    'flip': random_flip,
    'rotate': random_rotation,
    'rand_conv': random_convolution,
    'color_jitter': random_color_jitter,
    'no_aug': no_aug,
}

def apply_augs(images, config):
    for aug in config.augs:
        n, h, w, c = images.shape 
        images = np.reshape(images, (n, c, h, w))
        images = aug_to_func[aug](images)
    return images