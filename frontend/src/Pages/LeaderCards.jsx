import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { connect, useSelector } from "react-redux";
import "../css/Mainpage.css"; // Ensure this path is correct

const Home = ({ user }) => {
  // Get token and auth status using useSelector
  const { access, isAuthenticated } = useSelector((state) => state.AuthReducer);
  const navigate = useNavigate();
  const [leaders, setLeaders] = useState([]);

  // Fetch leader data using axios (with token authorization)
  useEffect(() => {
    if (!isAuthenticated || !access) {
      navigate("/login");
      return;
    }

    const fetchLeaders = async () => {
      try {
        const config = {
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${access}`,
          },
        };

        const response = await axios.get(
          `${process.env.REACT_APP_API_URL}/api/leaders/`,
          config
        );
        setLeaders(response.data);
      } catch (error) {
        console.error("Error fetching leaders:", error);
        // Optionally redirect to login if unauthorized
        if (error.response && error.response.status === 401) {
          navigate("/login");
        }
      }
    };

    fetchLeaders();
  }, [access, isAuthenticated, navigate]);

  // Handle click on the user logo in the navbar
  const handleUserLogoClick = () => {
    navigate("/user");
  };

  return (
    <div className="page-container">
      {/* 🔹 Navbar */}
      <div className="navbar">
        <h2 className="navbar-title">SilAI</h2>
        <img
          src={user?.profilePicture || "./user.svg"}
          alt="User Logo"
          className="user-logo"
          onClick={handleUserLogoClick}
        />
      </div>

      {/* 🔹 Greeting */}
      <h1 className="greet">Hello, {user?.first_name || "User"}</h1>

      {/* 🔹 People Section (showing leader cards) */}
      <div className="list">
        <h4 className="l-header">People</h4>
        <div className="scroll-container">
          {leaders.length > 0 ? (
            leaders.map((leader) => (
              <div key={leader.id} className="person-box">
                <img
                  src={leader.image || "./default-leader.svg"}
                  alt={leader.name}
                  className="person-image"
                />
                <p className="person-name">{leader.name}</p>
                <button className="chat-button">Chat</button>
              </div>
            ))
          ) : (
            <p className="loading-text">Loading...</p>
          )}
        </div>
      </div>

      {/* 🔹 Symbolism Section (placeholder or add your data)
      <div className="list">
        <h4 className="l-header">Symbolism</h4>
        <div className="scroll-container">
          Replace below with your symbolism data mapping if available
          <p className="loading-text">No symbolism data available</p>
        </div>
      </div> */}
    </div>
  );
};

const mapStateToProps = (state) => ({
  user: state.AuthReducer.user, // User info from your Redux state
});

export default connect(mapStateToProps)(Home);
