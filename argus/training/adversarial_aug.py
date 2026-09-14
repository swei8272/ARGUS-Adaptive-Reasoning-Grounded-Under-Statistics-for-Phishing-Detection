"""
Adversarial Augmentation Module
Visual manipulation for robustness training
"""
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
from typing import Tuple, Optional


class AdversarialAugmentation:
    """
    Adversarial data augmentation for visual robustness

    Applies 6 types of manipulations:
    1. Gaussian noise
    2. JPEG compression
    3. Blur
    4. Color shift
    5. Geometric transform
    6. Resolution reduction
    """

    def __init__(self, manipulation_types: Optional[list] = None):
        """
        Args:
            manipulation_types: List of manipulation types to use
        """
        if manipulation_types is None:
            manipulation_types = [
                'gaussian_noise',
                'jpeg_compression',
                'blur',
                'color_shift',
                'geometric_transform',
                'resolution_reduction'
            ]

        self.manipulation_types = manipulation_types

    def __call__(self,
                 image: Image.Image,
                 p: float = 0.5) -> Tuple[Image.Image, bool]:
        """
        Apply random augmentation

        Args:
            image: PIL Image
            p: Probability of applying manipulation

        Returns:
            augmented_image: Augmented image
            is_manipulated: Whether manipulation was applied
        """
        if random.random() > p:
            return image, False

        # Randomly select one manipulation
        manipulation = random.choice(self.manipulation_types)

        if manipulation == 'gaussian_noise':
            return self.add_gaussian_noise(image), True
        elif manipulation == 'jpeg_compression':
            return self.reduce_quality(image), True
        elif manipulation == 'blur':
            return self.apply_blur(image), True
        elif manipulation == 'color_shift':
            return self.color_shift(image), True
        elif manipulation == 'geometric_transform':
            return self.geometric_transform(image), True
        elif manipulation == 'resolution_reduction':
            return self.reduce_resolution(image), True
        else:
            return image, False

    def add_gaussian_noise(self,
                          image: Image.Image,
                          std: int = 25) -> Image.Image:
        """
        Add Gaussian noise

        Args:
            image: Input image
            std: Standard deviation of noise

        Returns:
            noisy_image: Image with noise
        """
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def reduce_quality(self,
                      image: Image.Image,
                      quality: int = 30) -> Image.Image:
        """
        Reduce JPEG quality

        Args:
            image: Input image
            quality: JPEG quality (1-100)

        Returns:
            compressed_image: Compressed image
        """
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def apply_blur(self,
                  image: Image.Image,
                  radius: int = 3) -> Image.Image:
        """
        Apply Gaussian blur

        Args:
            image: Input image
            radius: Blur radius

        Returns:
            blurred_image: Blurred image
        """
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def color_shift(self, image: Image.Image) -> Image.Image:
        """
        Apply random color shifts

        Args:
            image: Input image

        Returns:
            shifted_image: Color-shifted image
        """
        # Random factors
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3)
        saturation_factor = random.uniform(0.7, 1.3)

        # Apply enhancements
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)

        return image

    def geometric_transform(self, image: Image.Image) -> Image.Image:
        """
        Apply random geometric transformations

        Args:
            image: Input image

        Returns:
            transformed_image: Transformed image
        """
        # Random rotation
        angle = random.uniform(-15, 15)
        image = image.rotate(angle, fillcolor=(255, 255, 255))

        # Random scaling
        scale = random.uniform(0.8, 1.2)
        w, h = image.size
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Crop or pad back to original size
        if scale > 1:
            # Center crop
            left = (new_size[0] - w) // 2
            top = (new_size[1] - h) // 2
            image = image.crop((left, top, left + w, top + h))
        else:
            # Pad with white
            new_img = Image.new('RGB', (w, h), (255, 255, 255))
            paste_x = (w - new_size[0]) // 2
            paste_y = (h - new_size[1]) // 2
            new_img.paste(image, (paste_x, paste_y))
            image = new_img

        return image

    def reduce_resolution(self,
                         image: Image.Image,
                         scale: float = 0.5) -> Image.Image:
        """
        Reduce resolution

        Args:
            image: Input image
            scale: Downscaling factor

        Returns:
            low_res_image: Lower resolution image
        """
        w, h = image.size
        small_size = (int(w * scale), int(h * scale))

        # Downscale
        small = image.resize(small_size, Image.Resampling.LANCZOS)

        # Upscale back
        return small.resize((w, h), Image.Resampling.LANCZOS)


class VisualManipulationDetector:
    """
    Detect visual manipulations in images
    Used for training the confidence estimator
    """

    def __init__(self):
        self.augmentor = AdversarialAugmentation()

    def create_training_pair(self,
                            image: Image.Image) -> Tuple[Image.Image, Image.Image, int]:
        """
        Create a training pair: clean + manipulated

        Args:
            image: Clean image

        Returns:
            clean_image: Original image
            manipulated_image: Manipulated version
            label: 1 if manipulated, 0 if clean
        """
        # 50% chance of manipulation
        manipulated, is_manipulated = self.augmentor(image, p=0.5)

        if is_manipulated:
            return image, manipulated, 1
        else:
            return image, image, 0

    def create_batch(self,
                    images: list,
                    batch_size: int = 32) -> dict:
        """
        Create a batch of training pairs

        Args:
            images: List of PIL Images
            batch_size: Batch size

        Returns:
            batch: Dictionary with clean, manipulated, and labels
        """
        clean_batch = []
        manipulated_batch = []
        labels = []

        for _ in range(batch_size):
            image = random.choice(images)
            clean, manipulated, label = self.create_training_pair(image)

            clean_batch.append(clean)
            manipulated_batch.append(manipulated)
            labels.append(label)

        return {
            'clean': clean_batch,
            'manipulated': manipulated_batch,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


def test_augmentation():
    """Test the augmentation pipeline"""
    print("=" * 80)
    print("Testing Adversarial Augmentation")
    print("=" * 80)

    # Create a test image
    test_image = Image.new('RGB', (224, 224), color=(128, 128, 255))

    augmentor = AdversarialAugmentation()

    print("\nTesting each manipulation type:")

    for manip_type in augmentor.manipulation_types:
        augmentor.manipulation_types = [manip_type]
        augmented, was_manipulated = augmentor(test_image, p=1.0)

        print(f"  ✅ {manip_type}: {'Applied' if was_manipulated else 'Not applied'}")
        print(f"     Size: {augmented.size}, Mode: {augmented.mode}")

    print("\n" + "=" * 80)
    print("All augmentation tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_augmentation()