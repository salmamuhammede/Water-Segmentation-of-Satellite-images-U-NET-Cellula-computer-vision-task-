from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from io import BytesIO
import base64
import tifffile
import segmentation_models_pytorch as smp
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS to communicate with React

# Define the model architecture
class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.pre_conv = nn.Conv2d(12, 3, kernel_size=1)
        self.unet = smp.Unet(
            encoder_name='vgg16',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        x = self.pre_conv(x)
        return self.unet(x)

# Instantiate and load the model
model = UNetModel()
model.load_state_dict(torch.load('unet_model_vgg16.pth'))
model.eval()

def normalize_image(image):
    norm_img = np.zeros_like(image, dtype=np.float32)
    
    for b in range(image.shape[2]):
        band = image[:, :, b]
        min_val = np.min(band)
        max_val = np.max(band)
        
        if max_val > min_val:
            norm_img[:, :, b] = (band - min_val) / (max_val - min_val)  # Normalize to 0-1
        else:
            norm_img[:, :, b] = 0  # If max_val is min_val, leave band unchanged
    return norm_img
import cv2
import numpy as np

def display_image(img_array, title='Image'):
    img = np.array(img_array)
    if img.ndim == 2:  # Grayscale image
        cv2.imshow(title, img)
    else:  # Color image
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Convert image to base64
def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def convert_to_image(array, mode='RGB'):
    # Check if the array is already of the correct shape for the mode
    if mode == 'RGB':
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("For mode 'RGB', input array must have 3 dimensions with the third dimension size of 3.")
    elif mode == 'L':
        if array.ndim == 3 and array.shape[2] == 1:
            # Remove the singleton dimension if present
            array = array.squeeze(-1)
        elif array.ndim != 2:
            raise ValueError("For mode 'L', input array must have 2 dimensions.")
    else:
        raise ValueError("Unsupported mode. Use 'RGB' or 'L'.")
    
    # Ensure the array is of type uint8
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    
    try:
        img = Image.fromarray(array, mode=mode)
    except Exception as e:
        print(f"Error converting array to image: {e}")
        raise
    
    return img


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        print("No image file found in request.")
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    
    if image_file:
        print('Processing image...')
        try:
            # Use tifffile to read the multi-channel TIFF image
            image = tifffile.imread(image_file)
            
            if image.ndim != 3 or image.shape[2] != 12:
                return jsonify({"error": "Image must have 12 channels."}), 400

        except Exception as e:
            print(f"Error opening image: {e}")
            return jsonify({"error": "Error processing image"}), 500

        # Normalize the image
        normalized_image = normalize_image(image)
          
        # Create RGB Composite (assuming Blue = Band 2, Green = Band 3, Red = Band 4)
        blue_band = normalized_image[:, :, 1]  # Band 2 - Blue
        green_band = normalized_image[:, :, 2]  # Band 3 - Green
        red_band = normalized_image[:, :, 3]  # Band 4 - Red

        # Stack the bands into an RGB composite
        rgb_composite = np.dstack((red_band, green_band, blue_band))
        # Create NDWI (Normalized Difference Water Index)
        nir_band = normalized_image[:, :, 4]  # Band 5 - NIR
        ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-8)

        # Create SWIR composite
        swir1_band = normalized_image[:, :, 5]  # SWIR 1
        swir2_band = normalized_image[:, :, 6]  # SWIR 2
        swir_composite = np.dstack((swir1_band, swir2_band, green_band))

        # Predict using the model
        input_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = torch.sigmoid(output).squeeze(0).permute(1, 2, 0).numpy()
            binary_predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
        
        # Convert all images to base64 for sending back to React
        try:
            print(rgb_composite.shape)
            print(ndwi.shape)
            print(swir_composite.shape)
            print(predicted_mask.shape)
            print(binary_predicted_mask.shape)
            rgb_img = convert_to_image((rgb_composite * 255).astype(np.uint8), mode='RGB')
            ndwi_img = convert_to_image((ndwi * 255).astype(np.uint8), mode='L')
            swir_img = convert_to_image((swir_composite * 255).astype(np.uint8),  mode='RGB')
            predicted_img = convert_to_image((predicted_mask * 255).astype(np.uint8), mode='L')
            binary_predicted_img = convert_to_image((binary_predicted_mask * 255).astype(np.uint8), mode='L')
        except ValueError as e:
            print(f"ValueError: {e}")
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            print(f"Unexpected error: {e}")
            return jsonify({"error": "Unexpected error occurred"}), 500

        rgb_img_base64 = image_to_base64(rgb_img)
        ndwi_img_base64 = image_to_base64(ndwi_img)
        swir_img_base64 = image_to_base64(swir_img)
        predicted_img_base64 = image_to_base64(predicted_img)
        binary_predicted_img_base64 = image_to_base64(binary_predicted_img)

        return jsonify({
            'rgb_composite': rgb_img_base64,
            'ndwi': ndwi_img_base64,
            'swir_composite': swir_img_base64,
            'predicted_mask': predicted_img_base64,
            'binary_predicted_mask': binary_predicted_img_base64
        })
    else:
        print("Image file is empty.")
        return jsonify({"error": "Image file is empty"}), 400


if __name__ == '__main__':
    app.run(debug=True)