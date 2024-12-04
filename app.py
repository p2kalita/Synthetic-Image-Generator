import json
import logging
import warnings
from pathlib import Path
import random
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
import albumentations as A
from joblib import Parallel, delayed
from typing import List, Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SyntheticImageGenerator:
    def __init__(self, input_dir: str, output_dir: str, image_number: int, max_objects_per_image: int,
                 image_width: int, image_height: int, augmentation_path: str,
                 scale_foreground_by_background_size: bool, scaling_factors: List[float],
                 avoid_collisions: bool, parallelize: bool):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_number = image_number
        self.max_objects_per_image = max_objects_per_image
        self.image_width = image_width
        self.image_height = image_height
        self.zero_padding = 8
        self.augmentation_path = Path(augmentation_path)
        self.scale_foreground_by_background_size = scale_foreground_by_background_size
        self.scaling_factors = scaling_factors
        self.avoid_collisions = avoid_collisions
        self.parallelize = parallelize

        self._validate_input_directory()
        self._validate_output_directory()
        self._load_augmentation_pipeline()

    def _validate_input_directory(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f'Input directory does not exist: {self.input_dir}')

        self.foregrounds_dir = self.input_dir / 'foregrounds'
        self.backgrounds_dir = self.input_dir / 'backgrounds'

        if not self.foregrounds_dir.is_dir():
            raise FileNotFoundError(f"'foregrounds' sub-directory not found in {self.input_dir}")
        if not self.backgrounds_dir.is_dir():
            raise FileNotFoundError(f"'backgrounds' sub-directory not found in {self.input_dir}")

        self._process_foregrounds()
        self._process_backgrounds()

    def _process_foregrounds(self) -> None:
        self.foregrounds_dict: Dict[str, List[Path]] = {}
        for category in self.foregrounds_dir.iterdir():
            if category.is_dir():
                images = list(category.glob('*.png'))
                if images:
                    logging.info(f"Found category: {category.name}, images: {[str(img) for img in images]}")
                    self.foregrounds_dict[category.name] = images
        if not self.foregrounds_dict:
            raise ValueError(f"No valid foreground images found in {self.foregrounds_dir}")

    def _process_backgrounds(self) -> None:
        self.background_images: List[Path] = list(self.backgrounds_dir.glob('*.png')) + \
                                             list(self.backgrounds_dir.glob('*.jpg')) + \
                                             list(self.backgrounds_dir.glob('*.jpeg'))
        if not self.background_images:
            raise ValueError(f"No valid background images found in {self.backgrounds_dir}")

    def _validate_output_directory(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        elif any(self.output_dir.iterdir()):
            raise FileExistsError(f'Output directory is not empty: {self.output_dir}')

    def _load_augmentation_pipeline(self) -> None:
        if self.augmentation_path.is_file() and self.augmentation_path.suffix == '.yml':
            self.transforms = A.load(str(self.augmentation_path), data_format='yaml')
        else:
            self.transforms = None
            warnings.warn(f'{self.augmentation_path} is not a valid augmentation file. No augmentations will be applied.')

    def _generate_image(self, image_number: int) -> None:
        try:
            background_image_path = random.choice(self.background_images)
            num_foreground_images = random.randint(1, self.max_objects_per_image)
            foregrounds = [
                {
                    'category': random.choice(list(self.foregrounds_dict.keys())),
                    'image_path': random.choice(self.foregrounds_dict[random.choice(list(self.foregrounds_dict.keys()))])
                }
                for _ in range(num_foreground_images)
            ]

            composite, annotations = self._compose_images(foregrounds, background_image_path)
            save_filename = f'{image_number:0{self.zero_padding}}'
            composite_path = self.output_dir / f'{save_filename}.jpg'

            # Convert to RGB and save as JPEG
            composite.convert('RGB').save(composite_path, format='JPEG')

            annotations['imagePath'] = composite_path.name
            with open(self.output_dir / f'{save_filename}.json', 'w') as f:
                json.dump(annotations, f)
        except Exception as e:
            logging.error(f"Error generating image {image_number}: {e}")

    def _compose_images(self, foregrounds: List[Dict], background_image_path: Path) -> Tuple[Image.Image, Dict]:
        background = Image.open(background_image_path).convert('RGBA')
        background = background.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        composite = background.copy()

        annotations = {
            'shapes': [],
            'imageWidth': self.image_width,
            'imageHeight': self.image_height
        }

        for fg in foregrounds:
            fg_image = Image.open(fg['image_path']).convert('RGBA')
            scale = random.uniform(*self.scaling_factors)
            new_size = (int(fg_image.width * scale), int(fg_image.height * scale))
            fg_image = fg_image.resize(new_size, Image.Resampling.LANCZOS)

            if self.transforms:
                fg_image = Image.fromarray(self.transforms(image=np.array(fg_image))['image'])

            x = random.randint(0, max(0, self.image_width - fg_image.width))
            y = random.randint(0, max(0, self.image_height - fg_image.height))

            mask = fg_image.getchannel('A')
            composite.paste(fg_image, (x, y), mask)
            annotations['shapes'].append({
                'label': fg['category'],
                'points': [[x, y], [x + fg_image.width, y + fg_image.height]],
                'shape_type': 'rectangle'
            })

        return composite, annotations

    def generate_images(self) -> None:
        if self.parallelize:
            Parallel(n_jobs=-1)(
                delayed(self._generate_image)(i) for i in tqdm(range(1, self.image_number + 1))
            )
        else:
            for i in tqdm(range(1, self.image_number + 1), total=self.image_number):
                self._generate_image(i)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Synthetic Image Generator')
    parser.add_argument('--input_dir', required=True, help='Path to input directory')
    parser.add_argument('--output_dir', required=True, help='Path to output directory')
    parser.add_argument('--image_number', type=int, required=True, help='Number of images to generate')
    parser.add_argument('--max_objects_per_image', type=int, default=3, help='Max objects per image')
    parser.add_argument('--image_width', type=int, default=640, help='Image width')
    parser.add_argument('--image_height', type=int, default=480, help='Image height')
    parser.add_argument('--augmentation_path', type=str, default='transform.yml', help='Path to augmentation file')
    parser.add_argument('--scale_foreground_by_background_size', action='store_true', help='Scale foreground by background size')
    parser.add_argument('--scaling_factors', type=float, nargs=2, default=[0.2, 0.5], help='Foreground scaling range')
    parser.add_argument('--avoid_collisions', action='store_true', help='Avoid overlapping objects')
    parser.add_argument('--parallelize', action='store_true', help='Enable parallel processing')

    args = parser.parse_args()

    generator = SyntheticImageGenerator(
        args.input_dir, args.output_dir, args.image_number, args.max_objects_per_image,
        args.image_width, args.image_height, args.augmentation_path,
        args.scale_foreground_by_background_size, args.scaling_factors,
        args.avoid_collisions, args.parallelize
    )
    generator.generate_images()
