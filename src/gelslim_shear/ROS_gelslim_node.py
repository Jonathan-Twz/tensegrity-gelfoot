#!/usr/bin/env python3
"""
ROS Node for GelSlim Shear Force Visualization and Processing

This node captures camera feed, processes tactile images to compute shear forces,
detects contact points, and publishes visualization images to ROS topics.
"""

import os
import time
from typing import Tuple, List, Optional

# Set environment variable to avoid OpenMP library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header, MultiArrayDimension, MultiArrayLayout, Float32MultiArray

# https://github.com/UMich-HDRLab/tensegrity_msgs
from tensegrity_msgs.msg import Float32MultiArrayStamped

from cv_bridge import CvBridge
from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator
from gelslim_shear.plot_utils.shear_plotter import (
    cv_plot_scalar_field,
    cv_plot_vector_field,
    get_channel,
)

class Config:
    """Configuration constants for the GelSlim node."""
    # Camera settings
    DEFAULT_CAMERA_INDEX = 6
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    FRAME_RATE = 30
    
    # Image processing
    PROCESSED_IMAGE_SIZE = (200, 200)
    SHEAR_OUTPUT_SIZE = (30, 30)
    DISPLAY_SIZE = (600, 600)
    
    # Contact detection
    MAX_SHEAR_MAGNITUDE = 3.0
    CONTACT_MIN_DISTANCE = 2
    CONTACT_THRESHOLD = 150
    CONTACT_WINDOW_SIZE = 5
    GAUSSIAN_AMPLITUDE_FACTOR = 0.8
    
    # Visualization
    MARKER_SIZE = 20
    MARKER_THICKNESS = 5
    SHEAR_COLOR = (255, 0, 0)  # Red
    SHEAR_DIFF_COLOR = (0, 0, 255)  # Blue
    CONTACT_COLOR = (0, 0, 255)  # Blue
    FRAME_ID = "grasp_frame"
    
    # ShearGenerator parameters
    SHEAR_CHANNELS = ['u', 'v', 'div', 'du', 'dv']
    FARNEBACK_PARAMS = (0.5, 3, 45, 3, 5, 1.2, 0)
    
    # ROS image topics
    TOPIC_CROPPED = "gelslim/image/cropped"
    TOPIC_SHEAR = "gelslim/image/shear"
    TOPIC_DIVERGENCE = "gelslim/image/divergence"
    TOPIC_SHEAR_DIFF = "gelslim/image/shear_diff"
    TOPIC_MODELED = "gelslim/image/modeled"
    
    # ROS array topics
    TOPIC_SHEAR_VECTOR = "gelslim/array/shear_vector"
    # TOPIC_DIVERGENCE_VECTOR = "gelslim/array/divergence_vector"

# ================================
# UTILITY FUNCTIONS
# ================================

def square_center_crop(image: torch.Tensor) -> torch.Tensor:
    """
    Crop a 3D tensor to a square by center-cropping the larger dimension.
    
    Args:
        image: 3D tensor with shape (C, H, W)
        
    Returns:
        Square-cropped tensor
    """
    height = image.shape[1]
    width = image.shape[2]
    
    if height > width:
        start = (height - width) // 2
        return image[:, start:start+width, :]
    elif width > height:
        start = (width - height) // 2
        return image[:, :, start:start+height]
    else:
        return image


def downsample(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Downsample an image tensor to the specified size.
    
    Args:
        image: Input tensor
        size: Target size (height, width)
        
    Returns:
        Downsampled tensor
    """
    return F.interpolate(image.unsqueeze(0), size=size, mode='area').squeeze(0)


def center_crop(frame: np.ndarray) -> np.ndarray:
    """
    Crop a 2D frame to a square by center-cropping the larger dimension.
    
    Args:
        frame: 2D frame with shape (H, W) or 3D with (H, W, C)
        
    Returns:
        Square-cropped frame
    """
    h, w = frame.shape[:2]
    m = min(h, w)
    sy = (h - m) // 2
    sx = (w - m) // 2
    return frame[sy:sy+m, sx:sx+m]

def local_moments(window: np.ndarray, y0: int, x0: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate mean and covariance of a patch using weighted second moments.
    
    Args:
        window: 2D array representing the intensity window
        y0, x0: Offset coordinates for the window position
        
    Returns:
        Tuple of (mean_position, covariance_matrix)
    """
    H, W = window.shape
    y, x = np.indices((H, W))
    weights = window
    weights_sum = weights.sum()
    
    if weights_sum == 0:
        # Fallback to center and identity covariance if patch is all zero
        return np.array([y0, x0]), np.eye(2)
    
    # Calculate weighted mean
    mu_y = np.sum(y * weights) / weights_sum
    mu_x = np.sum(x * weights) / weights_sum
    
    # Calculate centered coordinates
    y_c = y - mu_y
    x_c = x - mu_x
    
    # Calculate covariance components
    cov_yy = np.sum((y_c ** 2) * weights) / weights_sum
    cov_xx = np.sum((x_c ** 2) * weights) / weights_sum
    cov_xy = np.sum((y_c * x_c) * weights) / weights_sum
    
    # Construct covariance matrix
    cov = np.array([[cov_yy, cov_xy],
                    [cov_xy, cov_xx]])
    
    # Ensure positive definiteness
    epsilon = 1e-6
    cov += epsilon * np.eye(2)
    
    return np.array([mu_y + y0, mu_x + x0]), cov


def gaussian_2d(shape: Tuple[int, int], mean: np.ndarray, cov: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """
    Generate a 2D Gaussian distribution over the specified shape.
    
    Args:
        shape: Shape of the output array (height, width)
        mean: Mean position [y, x]
        cov: 2x2 covariance matrix
        amplitude: Maximum amplitude of the Gaussian
        
    Returns:
        2D array containing the Gaussian distribution
    """
    y, x = np.indices(shape)
    pos = np.stack([y - mean[0], x - mean[1]], axis=-1)
    inv_cov = np.linalg.inv(cov)
    exponent = np.einsum('...i,ij,...j->...', pos, inv_cov, pos)
    return amplitude * np.exp(-0.5 * exponent)

# ================================
# MAIN CLASS
# ================================

class GelSlimShearNode:
    """
    ROS node for processing GelSlim tactile sensor data and computing shear forces.
    
    This class handles camera capture, shear force computation, contact detection,
    and visualization publishing to ROS topics.
    """
    
    def __init__(self, camera_index: int = Config.DEFAULT_CAMERA_INDEX, 
                 width: int = Config.DEFAULT_WIDTH, 
                 height: int = Config.DEFAULT_HEIGHT):
        """
        Initialize the GelSlim shear processing node.
        
        Args:
            camera_index: Index of the camera device
            width: Camera frame width
            height: Camera frame height
        """
        # Store configuration
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.frame_period = 1.0 / Config.FRAME_RATE
        
        # Initialize ROS components
        self.bridge = CvBridge()
        self._setup_publishers()
        
        # Initialize processing components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        self._setup_shear_generator()
        self._setup_camera()
        
        # Performance tracking
        self.frame_id = 0
        self.time_total = 0.0
    
    def _setup_publishers(self) -> None:
        """Setup ROS publishers for different visualization outputs."""
        # Images
        self.pub_cropped = rospy.Publisher(Config.TOPIC_CROPPED, Image, queue_size=1)
        self.pub_shear = rospy.Publisher(Config.TOPIC_SHEAR, Image, queue_size=1)
        self.pub_divergence = rospy.Publisher(Config.TOPIC_DIVERGENCE, Image, queue_size=1)
        self.pub_shear_diff = rospy.Publisher(Config.TOPIC_SHEAR_DIFF, Image, queue_size=1)
        self.pub_modeled = rospy.Publisher(Config.TOPIC_MODELED, Image, queue_size=1)
        
        # Arrays
        self.pub_shear_vector = rospy.Publisher(Config.TOPIC_SHEAR_VECTOR, Float32MultiArrayStamped, queue_size=10)
        # self.pub_divergence_vector = rospy.Publisher(Config.TOPIC_DIVERGENCE_VECTOR, Float32MultiArrayStamped, queue_size=1)
        
        rospy.loginfo(f"Publishers initialized:")
        rospy.loginfo(f"  - Images: cropped, shear, divergence, shear_diff, modeled")
        rospy.loginfo(f"  - Shear vector field: {Config.TOPIC_SHEAR_VECTOR} (Float32MultiArray)")
        # rospy.loginfo(f"  - Divergence vector field: {Config.TOPIC_DIVERGENCE_VECTOR} (Float32MultiArray)")
    
    def _setup_shear_generator(self) -> None:
        """Initialize the shear force computation generator."""
        self.shgen = ShearGenerator(
            method='2',
            channels=Config.SHEAR_CHANNELS,
            Farneback_params=Config.FARNEBACK_PARAMS,
            output_size=Config.SHEAR_OUTPUT_SIZE
        )
        
        # Load baseline image
        no_shear_image_path = os.path.join(os.path.dirname(__file__), 'no_shear_image.png')
        no_shear_image = cv2.imread(no_shear_image_path, cv2.IMREAD_COLOR)
        if no_shear_image is None:
            raise RuntimeError(f"No shear image not found at: {no_shear_image_path}")
        
        self.shgen.update_base_tactile_image(no_shear_image)
        rospy.loginfo("Shear generator initialized successfully")
    
    def _setup_camera(self) -> None:
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera #{self.camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        rospy.loginfo(f"Camera #{self.camera_index} opened successfully")
    
    def process_shear(self, frame_image: np.ndarray) -> Tuple[dict, torch.Tensor]:
        """
        Process a frame to compute shear forces and generate visualization images.
        
        Args:
            frame_image: Input camera frame
            
        Returns:
            Tuple of (images_dict, shear_vector_field)
        """
        # Convert and preprocess image
        frame_tensor = torch.from_numpy(frame_image).permute(2, 0, 1).float().to(self.device)
        frame_tensor = square_center_crop(frame_tensor)
        frame_tensor = downsample(frame_tensor, Config.PROCESSED_IMAGE_SIZE)
        
        # Update shear generator
        if self.frame_id == 0:
            self.shgen.update_base_tactile_image(frame_tensor)
        
        self.shgen.update_time(self.frame_id * self.frame_period)
        self.shgen.update_tactile_image(frame_tensor)
        self.shgen.update_shear()
        
        # Extract shear field components
        shear_field_tensor = self.shgen.get_shear_field()
        
        # Get velocity field (u, v components)
        vf = get_channel(shear_field_tensor, [self.shgen.channels.index('u'), 
                                             self.shgen.channels.index('v')])
        
        # Get scalar divergence field
        sf = get_channel(shear_field_tensor, self.shgen.channels.index('div'))

        # Difference vector field for visualization
        diff_vf = get_channel(shear_field_tensor, [self.shgen.channels.index('du'),
                                                  self.shgen.channels.index('dv')])

        # Build visualization images
        shear_img = cv_plot_vector_field(
            vf,
            ch_dim=0,
            image_size=Config.DISPLAY_SIZE,
            color=Config.SHEAR_COLOR,
        )

        divergence_small = cv_plot_scalar_field(
            sf,
            max_magnitude=Config.MAX_SHEAR_MAGNITUDE,
            colormap=cv2.COLORMAP_JET,
        )
        divergence_img = cv2.resize(
            divergence_small,
            Config.DISPLAY_SIZE,
            interpolation=cv2.INTER_NEAREST,
        )

        shear_diff_img = cv_plot_vector_field(
            diff_vf,
            ch_dim=0,
            image_size=Config.DISPLAY_SIZE,
            color=Config.SHEAR_DIFF_COLOR,
        )

        frame_cropped = center_crop(frame_image)
        frame_cropped = cv2.resize(
            frame_cropped,
            Config.DISPLAY_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

        # Contact modeling based on divergence magnitude
        sf_numpy = sf.cpu().numpy()
        clipped = np.clip(
            sf_numpy,
            -Config.MAX_SHEAR_MAGNITUDE,
            Config.MAX_SHEAR_MAGNITUDE,
        )
        normed = ((clipped + Config.MAX_SHEAR_MAGNITUDE) /
                  (2 * Config.MAX_SHEAR_MAGNITUDE) * 255).astype(np.float32)

        coords = self.detect_contacts(sf)
        modeled_field = self.generate_contact_model(normed, coords)

        modeled_small = cv_plot_scalar_field(
            modeled_field,
            max_magnitude=128,
            colormap=cv2.COLORMAP_JET,
        )
        modeled_img = cv2.resize(
            modeled_small,
            Config.DISPLAY_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

        # Overlay contact markers on key images
        self.add_contact_markers(
            [frame_cropped, shear_img, divergence_img],
            coords,
        )

        images = {
            'cropped': frame_cropped,
            'shear': shear_img,
            'divergence': divergence_img,
            'shear_diff': shear_diff_img,
            'modeled': modeled_img,
        }
        
        return images, vf
    
    def detect_contacts(self, shear_field: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Detect contact points from the shear field.
        
        Args:
            shear_field: Shear field tensor
            
        Returns:
            List of (y, x) coordinates of detected contact points
        """
        # Convert to numpy and normalize
        sf = shear_field.cpu().numpy()
        clipped = np.clip(sf, -Config.MAX_SHEAR_MAGNITUDE, Config.MAX_SHEAR_MAGNITUDE)
        normed = ((clipped + Config.MAX_SHEAR_MAGNITUDE) / 
                 (2 * Config.MAX_SHEAR_MAGNITUDE) * 255).astype(np.float32)
        
        # Find local maxima as contact points
        coords = peak_local_max(normed, 
                              min_distance=Config.CONTACT_MIN_DISTANCE,
                              threshold_abs=Config.CONTACT_THRESHOLD)
        
        return [(int(y), int(x)) for y, x in coords]
    
    def generate_contact_model(self, normed: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
        """
        Generate a Gaussian mixture model of detected contacts.
        
        Args:
            normed: Normalized shear field
            coords: List of contact point coordinates
            
        Returns:
            Modeled contact field as 2D array
        """
        modeled = np.zeros_like(normed)
        half_window = Config.CONTACT_WINDOW_SIZE // 2
        
        for y0, x0 in coords:
            # Define window around contact point
            ymin = max(0, y0 - half_window)
            ymax = min(normed.shape[0], y0 + half_window + 1)
            xmin = max(0, x0 - half_window)
            xmax = min(normed.shape[1], x0 + half_window + 1)
            
            # Extract window and compute local statistics
            window = normed[ymin:ymax, xmin:xmax]
            mu_win, cov_win = local_moments(window, ymin, xmin)
            
            # Generate Gaussian for this contact
            amplitude = window.max() * Config.GAUSSIAN_AMPLITUDE_FACTOR
            g = gaussian_2d(normed.shape, mean=mu_win, cov=cov_win, amplitude=amplitude)
            modeled += g
        
        return modeled
    
    def add_contact_markers(self, images: List[np.ndarray], coords: List[Tuple[int, int]]) -> None:
        """
        Add contact point markers to visualization images.
        
        Args:
            images: List of images to add markers to
            coords: List of contact point coordinates
        """
        scale_factor = Config.DISPLAY_SIZE[0] / Config.SHEAR_OUTPUT_SIZE[0]
        
        for y0, x0 in coords:
            marker_x = int(x0 * scale_factor)
            marker_y = int(y0 * scale_factor)
            
            for img in images:
                cv2.drawMarker(img, (marker_x, marker_y), 
                             color=Config.CONTACT_COLOR,
                             markerType=cv2.MARKER_CROSS, 
                             markerSize=Config.MARKER_SIZE, 
                             thickness=Config.MARKER_THICKNESS)
    
    # def create_vector_field_message(self, vf: torch.Tensor) -> Float32MultiArray:
    #     # Convert tensor to numpy and ensure it's on CPU
    #     vf_numpy = vf.cpu().numpy().astype(np.float32)
        
    #     # Create the message
    #     msg = Float32MultiArray()
        
    #     # Set up the layout - vector field has 3 dimensions: [channels, height, width]
    #     msg.layout.dim = [
    #         MultiArrayDimension(label="channels", size=vf_numpy.shape[0], stride=vf_numpy.size),
    #         MultiArrayDimension(label="height", size=vf_numpy.shape[1], stride=vf_numpy.shape[1] * vf_numpy.shape[2]),
    #         MultiArrayDimension(label="width", size=vf_numpy.shape[2], stride=vf_numpy.shape[2])
    #     ]
    #     msg.layout.data_offset = 0
    #     # Flatten the data and convert to list
    #     msg.data = vf_numpy.flatten().tolist()
    #     return msg

    def create_vector_field_message_stamped(self, vf: torch.Tensor, frame_id: Optional[str] = None, stamp: Optional[rospy.Time] = None) -> Float32MultiArrayStamped:
        # Convert tensor to numpy and ensure it's on CPU
        vf_numpy = vf.cpu().numpy().astype(np.float32)

        # Create the stamped message
        msg = Float32MultiArrayStamped()
        # Header with ROS time
        msg.header = Header()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        if frame_id is not None:
            msg.header.frame_id = frame_id

        # Layout matches Float32MultiArray message
        msg.layout.dim = [
            MultiArrayDimension(label="channels", size=vf_numpy.shape[0], stride=vf_numpy.size),
            MultiArrayDimension(label="height", size=vf_numpy.shape[1], stride=vf_numpy.shape[1] * vf_numpy.shape[2]),
            MultiArrayDimension(label="width", size=vf_numpy.shape[2], stride=vf_numpy.shape[2])
        ]
        msg.layout.data_offset = 0
        msg.data = vf_numpy.flatten().tolist()
        return msg

    def publish_images(self, images: dict, vf: Optional[torch.Tensor] = None) -> None:
        """
        Publish processed images and vector field to ROS topics.
        
        Args:
            images: Dictionary mapping topic names to image arrays
            vf: Optional vector field tensor to publish
        """
        try:
            now = rospy.Time.now()

            # cropped
            msg_cropped = self.bridge.cv2_to_imgmsg(images['cropped'], encoding='bgr8')
            msg_cropped.header.stamp = now
            msg_cropped.header.frame_id = Config.FRAME_ID
            self.pub_cropped.publish(msg_cropped)

            # shear
            msg_shear = self.bridge.cv2_to_imgmsg(images['shear'], encoding='bgr8')
            msg_shear.header.stamp = now
            msg_shear.header.frame_id = Config.FRAME_ID
            self.pub_shear.publish(msg_shear)

            # divergence
            msg_div = self.bridge.cv2_to_imgmsg(images['divergence'], encoding='bgr8')
            msg_div.header.stamp = now
            msg_div.header.frame_id = Config.FRAME_ID
            self.pub_divergence.publish(msg_div)

            # shear diff
            msg_diff = self.bridge.cv2_to_imgmsg(images['shear_diff'], encoding='bgr8')
            msg_diff.header.stamp = now
            msg_diff.header.frame_id = Config.FRAME_ID
            self.pub_shear_diff.publish(msg_diff)

            # modeled
            msg_modeled = self.bridge.cv2_to_imgmsg(images['modeled'], encoding='bgr8')
            msg_modeled.header.stamp = now
            msg_modeled.header.frame_id = Config.FRAME_ID
            self.pub_modeled.publish(msg_modeled)

            # Publish vector field if provided (use the same timestamp for sync)
            if vf is not None:
                vf_msg_stamped = self.create_vector_field_message_stamped(vf, frame_id="gelslim_vf", stamp=now)
                self.pub_shear_vector.publish(vf_msg_stamped)
                rospy.logdebug(f"Published vector field with shape: {vf.shape}")

        except Exception as e:
            rospy.logerr(f"ROS message publish error: {e}")
            raise
    
    def run(self) -> None:
        """Main processing loop."""
        rospy.loginfo("Starting GelSlim shear processing loop")
        
        try:
            while not rospy.is_shutdown():
                tic = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    rospy.logwarn("Failed to grab frame")
                    break
                
                # Process shear and publish results
                images, vf = self.process_shear(frame)
                self.publish_images(images, vf)
                
                # Update performance metrics
                self.frame_id += 1
                dt = time.time() - tic
                self.time_total += dt
                
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        except Exception as e:
            rospy.logerr(f"Error in processing loop: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.frame_id > 0 and self.time_total > 0:
            fps = self.frame_id / self.time_total
            rospy.loginfo(f"Shutting down ROS node, average FPS: {fps:.2f}")


# ================================
# MAIN FUNCTION
# ================================

def main(camera_index: int = Config.DEFAULT_CAMERA_INDEX, 
         width: int = Config.DEFAULT_WIDTH, 
         height: int = Config.DEFAULT_HEIGHT) -> None:
    """
    Main function to create and run the GelSlim shear processing node.
    
    Args:
        camera_index: Index of the camera device to use
        width: Camera frame width
        height: Camera frame height
    """
    try:
        node = GelSlimShearNode(camera_index, width, height)
        node.run()
    except Exception as e:
        rospy.logerr(f"Failed to initialize or run GelSlim node: {e}")
        raise

if __name__ == "__main__":
    """Entry point for the ROS node."""
    rospy.init_node("gelslim_shear_viz", anonymous=True)
    
    # You can override the default camera index here if needed
    camera_index = rospy.get_param('~camera_index', Config.DEFAULT_CAMERA_INDEX)
    width = rospy.get_param('~width', Config.DEFAULT_WIDTH)
    height = rospy.get_param('~height', Config.DEFAULT_HEIGHT)
    
    rospy.loginfo(f"Starting GelSlim node with camera {camera_index}, resolution {width}x{height}")
    
    try:
        main(camera_index=camera_index, width=width, height=height)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down due to keyboard interrupt")
    except Exception as e:
        rospy.logerr(f"Node failed with error: {e}")
        raise
