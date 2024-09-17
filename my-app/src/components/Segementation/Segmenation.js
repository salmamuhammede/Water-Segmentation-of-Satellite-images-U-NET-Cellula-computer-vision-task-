import React, { useState } from "react";
import axios from "axios";
import "./Segmentation.css";
// Ensure this path is correct
import RGB from "../../assets/rgb.PNG";
import NIR from "../../assets/nir.PNG";
import Soil from "../../assets/soil.PNG";
import mask1 from "../../assets/mask1.PNG";
import mask2 from "../../assets/mask2.PNG";
import SparkleButton from "../button/Button"; // Ensure this path is correct

function Segmentation() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [segmentedImages, setSegmentedImages] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false); // Add loading state

  // Handle image change (upload)
  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
  };

  // Handle form submission to send image to backend
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) {
      alert("Please upload an image");
      return;
    }

    setLoading(true); // Start loading

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/process_image", // Adjust if needed
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      // Check if response.data contains the images
      if (response.data && response.data.rgb_composite) {
        setSegmentedImages(response.data);
        setError(null); // Clear any previous errors
      } else {
        setError("No images returned from the server.");
      }
    } catch (error) {
      setError("Error uploading image. Please try again.");
      console.error("Error uploading image:", error);
    } finally {
      setLoading(false); // End loading
    }
  };

  return (
    <div className="Segmentation">
      <div className="container1">
        <div className="box">
          {error && <div className="error-message">{error}</div>}{" "}
          {/* Display error message */}
          <div className="Left">
            {/* Display images if they exist */}
            <div className="imgdef">
              <img src={RGB} width="200px" height="20px" />
              {segmentedImages.rgb_composite && (
                <img
                  src={`data:image/png;base64,${segmentedImages.rgb_composite}`}
                  alt="RGB Composite"
                  style={{ width: "200px", height: "200px" }} // Fixed height/width issue
                  className="image-display"
                />
              )}
            </div>
            <div className="imgdef">
              <img src={NIR} width="200px" height="5px" />
              {segmentedImages.ndwi && (
                <img
                  src={`data:image/png;base64,${segmentedImages.ndwi}`}
                  alt="NDWI"
                  style={{ width: "200px", height: "200px" }} // Fixed height/width issue
                  className="image-display"
                />
              )}
            </div>
            <div className="imgdef">
            <img src={Soil} width="200px" height="20px" />
            {segmentedImages.swir_composite && (
              <img
                src={`data:image/png;base64,${segmentedImages.swir_composite}`}
                alt="SWIR Composite"
                style={{ width: "200px", height: "200px" }} // Fixed height/width issue
                className="image-display"
              />
            )}
            </div>
          </div>
          <div className="Middle">
            {loading && <div className="loading-message">Uploading...</div>}{" "}
            {/* Loading feedback */}
            <form onSubmit={handleSubmit} className="odoo">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="upload"
              />
              <SparkleButton text="Upload Image" /> {/* Removed onClick */}
            </form>
          </div>
          <div className="Right">
            {/* Display other images if they exist */}
            <div className="imgdef">
            <img src={mask1} width="200px" height="10px" />
            {segmentedImages.predicted_mask && (
              <img
                src={`data:image/png;base64,${segmentedImages.predicted_mask}`}
                alt="Predicted Mask"
                style={{ width: "200px", height: "200px" }} // Fixed height/width issue
                className="image-display"
              />
            )}
            </div>
            <div className="imgdef">
            <img src={mask2} width="200px" height="10px" />
            {segmentedImages.binary_predicted_mask && (
              <img
                src={`data:image/png;base64,${segmentedImages.binary_predicted_mask}`}
                alt="Binary Predicted Mask"
                style={{ width: "200px", height: "200px" }} // Fixed height/width issue
                className="image-display"
              />
            )}
              </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Segmentation;
