import "./Home.css";
import { Link } from "react-router-dom"; // Import Link for navigation
import logo from "../../assets/aqua-.png";
import SparkleButton from "../button/Button";

function Home() {
  return (
    <div className="Home">
      <div className="main" align="center">
        <div className="container">
          <div className="box">
            <img src={logo} alt="AquaVision Logo" />
            <div>
              <p>Welcome to AquaVision, your cutting-edge tool for precise water segmentation from satellite images.</p>
            </div>
          </div>
        </div>
        {/* Wrap SparkleButton in Link to navigate to the segmentation page */}
        <Link to="/Seg">
          <SparkleButton text="Segmentify" />
        </Link>
      </div>
    </div>
  );
}

export default Home;
