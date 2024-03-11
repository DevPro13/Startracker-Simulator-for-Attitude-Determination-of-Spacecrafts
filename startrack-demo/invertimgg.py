from PIL import Image, ImageOps

def invert_image(image_path, output_path):
  """
  Inverts the colors of an image and saves it.

  Args:
      image_path: Path to the input image file.
      output_path: Path to save the inverted image.
  """
  try:
    image = Image.open(image_path)
    inverted_image = image.convert('RGB').convert('L').convert('RGB')  # Ensure RGB mode
    inverted_image = ImageOps.invert(inverted_image)
    inverted_image.save(output_path)
    print(f"Image inverted and saved to: {output_path}")
  except FileNotFoundError:
    print(f"Error: File not found at {image_path}")
  except Exception as e:
    print(f"Error inverting image: {e}")

# Example usage
image_path = "test.png"
output_path = "inverted_test.png"
invert_image(image_path, output_path)