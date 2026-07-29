import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator
from gelslim_shear.plot_utils.shear_plotter import cv_plot_scalar_field, cv_plot_vector_field, get_channel
import torch.nn.functional as F
import cv2
import numpy as np
import time
from skimage.feature import peak_local_max

def square_center_crop(image):
    # 3d image tensor with shape (C, H, W)
    height = image.shape[1]
    width = image.shape[2]
    if height > width:
        start = (height - width)//2
        return image[:, start:start+width, :]
    elif width > height:
        start = (width - height)//2
        return image[:, :, start:start+height]
    else:
        return image
    
def downsample(image, size):
    return F.interpolate(image.unsqueeze(0), size=size, mode='area').squeeze(0)

def center_crop(frame):
    # 2d frame with shape (H, W)
    h, w = frame.shape[:2]
    m = min(h, w)
    sy = (h - m) // 2
    sx = (w - m) // 2
    frame_cropped = frame[sy:sy+m, sx:sx+m]
    return frame_cropped

def process_shear(frame, frame_image, shgen, device, frame_period):
    frame_image = torch.from_numpy(frame_image).permute(2, 0, 1).float().to(device)
    frame_image = square_center_crop(frame_image)
    frame_image = downsample(frame_image, (200, 200))
    if frame == 0:
        shgen.update_base_tactile_image(frame_image)
    shgen.update_time(frame * frame_period)
    shgen.update_tactile_image(frame_image)
    shgen.update_shear()
    shear_field_tensor = shgen.get_shear_field()

    vf = get_channel(shear_field_tensor, [shgen.channels.index('u'), shgen.channels.index('v')])
    sf = get_channel(shear_field_tensor, shgen.channels.index('div'))
    diff_vf = get_channel(shear_field_tensor, [shgen.channels.index('du'), shgen.channels.index('dv')])

    shear      = cv_plot_vector_field(vf, ch_dim=0, color=(255, 0, 0))
    divergence = cv_plot_scalar_field(sf, max_magnitude=3, colormap=cv2.COLORMAP_JET)
    shear_diff = cv_plot_vector_field(diff_vf, ch_dim=0, color=(0, 0, 255))
    return shear, divergence, shear_diff, sf

def local_moments(window, y0, x0):
    """Estimate mean and covariance of a patch using weighted second moments."""
    H, W = window.shape
    y, x = np.indices((H, W))
    weights = window
    weights_sum = weights.sum()
    if weights_sum == 0:
        # fallback to center and identity covariance if patch is all zero
        return np.array([y0, x0]), np.eye(2)
    mu_y = np.sum(y * weights) / weights_sum
    mu_x = np.sum(x * weights) / weights_sum
    y_c = y - mu_y
    x_c = x - mu_x
    cov_yy = np.sum((y_c ** 2) * weights) / weights_sum
    cov_xx = np.sum((x_c ** 2) * weights) / weights_sum
    cov_xy = np.sum((y_c * x_c) * weights) / weights_sum
    cov = np.array([[cov_yy, cov_xy],
                    [cov_xy, cov_xx]])
    epslon = 1e-6
    cov += epslon * np.eye(2)  # ensure positive definiteness
    return np.array([mu_y + y0, mu_x + x0]), cov

def gaussian_2d(shape, mean, cov, amplitude=1.0):
    y, x = np.indices(shape)
    pos = np.stack([y - mean[0], x - mean[1]], axis=-1)
    inv_cov = np.linalg.inv(cov)
    exponent = np.einsum('...i,ij,...j->...', pos, inv_cov, pos)
    return amplitude * np.exp(-0.5 * exponent)

def main(camera_index=0, width=640, height=480):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device available:", device)
    frame_period = 1 / 30  # Adjust frame period as needed (FPS is assumed to be 30)
    
    shgen = ShearGenerator(method='2', channels=['u', 'v', 'div', 'du', 'dv'], 
                        Farneback_params=(0.5, 3, 45, 3, 5, 1.2, 0), output_size=(18, 18))
    
    # open the camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera #{camera_index}")
    else:
        print(f"Camera #{camera_index} opened successfully")

    # (optional) set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        frame_id = 0
        while True:
            tic = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # process shear field
            shear, divergence, shear_diff, sf = process_shear(frame_id, frame, shgen, device, frame_period)
            frame_id += 1
            
            frame_cropped = square_center_crop(frame)
            
            max_magnitude = 3.0
            sf = sf.cpu().numpy()
            clipped = np.clip(sf, -max_magnitude, max_magnitude)
            normed  = ((clipped + max_magnitude) / (2*max_magnitude) * 255).astype(np.float32)
            
            # TODO: tune force threshold in range [0, 255]
            coords = peak_local_max(normed, min_distance=2, threshold_abs=150)
            
            modeled = np.zeros_like(normed)
            window_size = 5  # use odd number
            half = window_size // 2
            for (y0, x0) in coords:
                # extract local window
                ymin = max(0, y0 - half)
                ymax = min(normed.shape[0], y0 + half + 1)
                xmin = max(0, x0 - half)
                xmax = min(normed.shape[1], x0 + half + 1)
                window = normed[ymin:ymax, xmin:xmax]
                # estimate mean/cov in window (relative to window top-left, adjust to global coords)
                mu_win, cov_win = local_moments(window, ymin, xmin)
                # TODO: uncomment for fix covariance
                # cov_win = np.array([[3, 0], [0, 3]])
                
                amplitude = window.max() * 0.8
                # generate Gaussian using estimated mean/cov and amplitude
                g = gaussian_2d(normed.shape, mean=mu_win, cov=cov_win, amplitude=amplitude)
                modeled += g
            
            modeled = cv_plot_scalar_field(modeled, max_magnitude=128, colormap=cv2.COLORMAP_JET)
            modeled = cv2.resize(modeled, (600, 600), interpolation=cv2.INTER_LINEAR)
            
            # resize images for display, can also use cv2.INTER_LINEAR for smoother results
            divergence = cv2.resize(divergence, (600, 600), interpolation=cv2.INTER_NEAREST)
            
            # put x markers of contact points on the images
            for (y0, x0) in coords:
                cv2.drawMarker(divergence, (int(x0*600/18), int(y0*600/18)), color=(255, 0, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=5)
                cv2.drawMarker(frame_cropped, (int(x0*600/18), int(y0*600/18)), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=5)
                cv2.drawMarker(shear, (int(x0*600/18), int(y0*600/18)), color=(0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=5)
                
            # cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Original', 600, 600)
            # cv2.moveWindow('Original', 0, 0)
            # cv2.imshow('Original', frame)
            
            cv2.namedWindow('Cropped Original', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Cropped Original', 600, 600)
            cv2.moveWindow('Cropped Original', 0, 0)
            cv2.imshow('Cropped Original', frame_cropped)

            cv2.namedWindow('Shear', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Shear', 600, 600)
            cv2.moveWindow('Shear', 0, 700)
            cv2.imshow('Shear', shear)

            cv2.namedWindow('Divergence', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Divergence', 600, 600)
            cv2.moveWindow('Divergence', 700, 0)
            cv2.imshow('Divergence', divergence)

            cv2.namedWindow('Shear Difference', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Shear Difference', 600, 600)
            cv2.moveWindow('Shear Difference', 700, 700)
            cv2.imshow('Shear Difference', shear_diff)

            cv2.namedWindow('Modeled Divergence', cv2.WINDOW_NORMAL)            
            cv2.resizeWindow('Modeled Divergence', 600, 600)
            cv2.moveWindow('Modeled Divergence', 1350, 0)
            cv2.imshow('Modeled Divergence', modeled)
            
            # cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('debug', 600, 600)
            # cv2.moveWindow('debug', 1400, 700)
            # cv2.imshow('debug', cv2.resize(debug, (600, 600), interpolation=cv2.INTER_LINEAR))
            
            toc = time.time()
            print(f"Processed in {toc - tic:.3f} seconds")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(camera_index=4) #8 if using dock, 4 if detached
