
# Synthetic Image Generator
![Output Example](output_example.PNG)

This code allows you to create a synthetic data-set, for Instance Segmentation or Object Detection. The [app.py](app.py) script outputs data in [LabelMe format](https://roboflow.com/formats/labelme-json), which can also be converted to other formats like the [COCO JSON format](https://cocodataset.org/).
## Overview
The Synthetic Image Generator script creates synthetic datasets by overlaying foreground images on background images. 
It also generates JSON annotations for object detection tasks, making it suitable for training AI models.

## Features
- Overlay foreground objects onto background images.
- Scale and position foreground objects randomly.
- Support for Albumentations augmentations.
- Generate annotations in JSON format (e.g., bounding boxes).
- Parallelized image generation for faster performance.
- Customizable parameters via command-line arguments.

## Requirements
- **Python**: Version 3.8 or later.
- **Dependencies**: Install via `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

## Directory Structure

### Input Directory
The `input` directory must contain the following subdirectories:
```
input/
├── backgrounds/
│   ├── background1.jpg
│   ├── background2.png
│   ...
├── foregrounds/
    ├── category1/
    │   ├── object1.png
    │   ├── object2.png
    ├── category2/
        ├── object1.png
        ├── object2.png
```

### Output Directory
The `output` directory will be created automatically and will store:
- **Generated Images**: `.jpg` format.
- **Annotations**: `.json` files for each image.

## Usage

### Command-Line Arguments
```bash
python app.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --image_number N [OPTIONS]
```

### Required Arguments
| Argument       | Description                                                     |
|----------------|-----------------------------------------------------------------|
| `--input_dir`  | Path to the directory containing `backgrounds` and `foregrounds`.|
| `--output_dir` | Path to the directory where images and annotations will be saved.|
| `--image_number` | Number of synthetic images to generate.                      |

### Optional Arguments
| Argument                           | Default Value | Description                                          |
|------------------------------------|---------------|------------------------------------------------------|
| `--max_objects_per_image`          | 3             | Maximum number of foreground objects per image.      |
| `--image_width`                    | 640           | Width of the output images.                          |
| `--image_height`                   | 480           | Height of the output images.                         |
| `--augmentation_path`              | transform.yml | Path to an Albumentations YAML file for augmentations.|
| `--scaling_factors`                | [0.2, 0.5]    | Range for resizing foreground images (proportionally).|
| `--parallelize`                    | False         | Enable multi-core processing for faster generation.  |

## Examples

### Basic Usage
Generate 10 synthetic images with default parameters:
```bash
python app.py --input_dir ./input --output_dir ./output --image_number 10
```

### Add Scaling and Parallel Processing
Generate 20 images, scaling foregrounds to 30%–70% of their original size, and using parallel processing:
```bash
python app.py --input_dir ./input --output_dir ./output --image_number 20 --scaling_factors 0.3 0.7 --parallelize
```

### Limit to One Foreground Object per Background
Generate 5 images with only one foreground object per image:
```bash
python app.py --input_dir ./input --output_dir ./output --image_number 5 --max_objects_per_image 1
```

## Annotation Format
The JSON annotations for each image include:
```json
{
  "imagePath": "00000001.jpg",
  "imageWidth": 640,
  "imageHeight": 480,
  "shapes": [
    {
      "label": "category1",
      "points": [[x1, y1], [x2, y2]],
      "shape_type": "rectangle"
    }
  ]
}
```

- `shapes`: List of objects in the image.
- `label`: Category of the foreground object.
- `points`: Bounding box coordinates for the object.
- `shape_type`: Fixed as "rectangle".

## Customizing Foreground Sizes
Modify `--scaling_factors` to control foreground size:

### Smaller Foregrounds
```bash
--scaling_factors 0.1 0.3
```

### Larger Foregrounds
```bash
--scaling_factors 0.7 1.0
```

## Troubleshooting

### Common Errors
1. **No valid foreground images found**:
   - Ensure the `foregrounds/` folder has subdirectories containing `.png` images.
2. **Cannot write mode RGBA as JPEG**:
   - Ensure the script converts RGBA to RGB before saving (already fixed in this version).

### Logs
The script logs its progress and errors. Check the console for detailed messages.

## License
Feel free to use and modify this script for your projects. Attribution is appreciated but not required.

