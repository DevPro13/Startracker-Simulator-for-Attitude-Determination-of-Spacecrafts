from PIL import Image

def scale_image(image_path, max_width=400, max_height=300):
    """
    Scales an image proportionally to fit within the specified maximum dimensions while preserving aspect ratio.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the scaled image.
        max_width (int, optional): Maximum allowed width for the image (in pixels). Defaults to None (no limit).
        max_height (int, optional): Maximum allowed height for the image (in pixels). Defaults to None (no limit).
    """

    try:
        with Image.open(image_path) as image:
            width, height = image.size

            # Determine scaling factor based on provided dimensions
            if max_width and max_height:
                scale_factor = min(max_width / width, max_height / height)
            elif max_width:
                scale_factor = max_width / width
            elif max_height:
                scale_factor = max_height / height
            else:
                # No scaling required (original size)
                scale_factor = 1.0

            # Calculate new dimensions while maintaining aspect ratio
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Scale and save the image
            scaled_image = image.resize((new_width, new_height), Image.ANTIALIAS)
            #scaled_image.save(output_path)
            #print(f"Image scaled and saved to: {output_path}")
            return scaled_image
    
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
    except Exception as e:
        print(f"Error scaling image: {e}")