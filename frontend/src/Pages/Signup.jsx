// import React from "react";
// import { Link } from "react-router-dom";
// import { Navigate } from "react-router-dom";
// import { useState } from "react";
// import { connect } from "react-redux";
// import { signup } from "../reducer/Actions";

// const Signup = ({ signup }) => {
//     const [ status, setStatus ] = useState (false);
//     const [ formData, setFormData ] = useState ({
//         email: "",
//         first_name: "",
//         last_name: "",
//         password1: "",
//         password2: ""
//     });
//     const [ error, setError ] = useState(null);

//     const { email, first_name, last_name, password1, password2 } = formData;
//     const handlingInput = (e) => setFormData({...formData, [e.target.name]: e.target.value});
//     const handlingSubmit = (e) => {
//         e.preventDefault();
//         signup( email, first_name, last_name, password1, password2 ).catch(err => {
//             if (err.response && err.response.data && err.response.data.detail) {
//                 setError(err.response.data.detail);
//             } else {
//                 setError("An error occurred. Please try again.");
//             }
//         });
//         setStatus(true)
//     }
//     if (status) {
//         return <Navigate to={"../"}></Navigate>
//     }
//     return (
//         <div className="page-container">
//         <h1 className="main-title">SilAI</h1>

//         <div className="login-container">
//             <div className="login-card">
//             <h2 className="text-center">Signup</h2>

//             <form className="login-form" onSubmit={handlingSubmit}>
//                 <div className="form-group">
//                 <label>First Name</label>
//                 <input
//                     type="text"
//                     name="first_name"
//                     value={first_name}
//                     onChange={handlingInput}
//                     placeholder="First name ..."
//                     className="form-control"
//                 />
//                 </div>

//                 <div className="form-group">
//                 <label>Last Name</label>
//                 <input
//                     type="text"
//                     name="last_name"
//                     value={last_name}
//                     onChange={handlingInput}
//                     placeholder="Last name ..."
//                     className="form-control"
//                 />
//                 </div>

//                 <div className="form-group">
//                 <label>Email address</label>
//                 <input
//                     type="email"
//                     name="email"
//                     value={email}
//                     onChange={handlingInput}
//                     placeholder="name@example.com"
//                     className="form-control"
//                 />
//                 </div>

//                 <div className="form-group">
//                 <label>Password</label>
//                 <input
//                     type="password"
//                     name="password1"
//                     value={password1}
//                     onChange={handlingInput}
//                     placeholder="Password ..."
//                     className="form-control"
//                 />
//                 </div>

//                 <div className="form-group">
//                 <label>Re-enter Password</label>
//                 <input
//                     type="password"
//                     name="password2"
//                     value={password2}
//                     onChange={handlingInput}
//                     placeholder="Password ..."
//                     className="form-control"
//                 />
//                 </div>

//                 <button className="login-button" type="submit">Signup</button>
//             </form>

//             {error && <div className="error-message">{error}</div>}
//             <p className="signup-link">
//                 Already have an account? <Link to="../login/">Login</Link>
//             </p>
//             </div>
//         </div>
//         </div>
//     )
// }

// export default connect(null, { signup })(Signup)

import React, { useState, useEffect } from "react";
import { Link, Navigate } from "react-router-dom";
import { connect } from "react-redux";
import { signup } from "../reducer/Actions";

// Utility function to check email from backend
const checkEmailExists = async (email) => {
  try {
    const response = await fetch(
      `${process.env.REACT_APP_API_URL}/accounts/check-email/?email=${encodeURIComponent(email)}`
    );
    console.log(response);
    const data = await response.json();
    return data.exists;
  } catch (error) {
    console.error("Error checking email:", error);
    return false;
  }
};

// Helper function to validate email format
const validateEmail = (email) => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
};

const Signup = ({ signup }) => {
  const [status, setStatus] = useState(false);
  const [emailExists, setEmailExists] = useState(false);
  const [emailCheckError, setEmailCheckError] = useState(null);
  const [formData, setFormData] = useState({
    email: "",
    first_name: "",
    last_name: "",
    password1: "",
    password2: ""
  });
  const [error, setError] = useState(null);

  const { email, first_name, last_name, password1, password2 } = formData;

  // When email changes, check its availability only if the email is valid
  useEffect(() => {
    if (email.trim() !== "" && validateEmail(email)) {
      checkEmailExists(email)
        .then((exists) => {
          setEmailExists(exists);
          setEmailCheckError(null);
        })
        .catch((error) => {
          setEmailCheckError("Error checking email");
          console.error(error);
        });
    } else {
      // Reset emailExists if the email field is empty or invalid
      setEmailExists(false);
    }
  }, [email]);

  const handlingInput = (e) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  const handlingSubmit = (e) => {
    e.preventDefault();
    if (!validateEmail(email)) {
      alert("Please enter a valid email address.");
      return;
    }
    if (emailExists) {
      alert(
        "This email is already registered. Please use a different email or login."
      );
      return;
    }
    if (password1 !== password2) {
      alert("Passwords do not match.");
      return;
    }
    // Dispatch the signup action
    signup(email, first_name, last_name, password1, password2).catch((err) => {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError("An error occurred. Please try again.");
      }
    });
    setStatus(true);
  };

  if (status) {
    return <Navigate to={"../"} />;
  }

  return (
    <div className="page-container">
      <h1 className="main-title">SilAI</h1>

      <div className="login-container">
        <div className="login-card">
          <h2 className="text-center">Signup</h2>

          <form className="login-form" onSubmit={handlingSubmit}>
            <div className="form-group">
              <label>First Name</label>
              <input
                type="text"
                name="first_name"
                value={first_name}
                onChange={handlingInput}
                placeholder="First name ..."
                className="form-control"
              />
            </div>

            <div className="form-group">
              <label>Last Name</label>
              <input
                type="text"
                name="last_name"
                value={last_name}
                onChange={handlingInput}
                placeholder="Last name ..."
                className="form-control"
              />
            </div>

            <div className="form-group">
              <label>Email address</label>
              <input
                type="email"
                name="email"
                value={email}
                onChange={handlingInput}
                placeholder="name@example.com"
                className="form-control"
              />
              {/* Show an error if email format is invalid */}
              {email && !validateEmail(email) && (
                <small style={{ color: "red" }}>
                  Please enter a valid email address.
                </small>
              )}
              {email && validateEmail(email) && (
                <small style={{ color: emailExists ? "red" : "green" }}>
                  {emailExists
                    ? "This email is already registered."
                    : "Email available."}
                </small>
              )}
              {emailCheckError && (
                <small style={{ color: "red" }}>{emailCheckError}</small>
              )}
            </div>

            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                name="password1"
                value={password1}
                onChange={handlingInput}
                placeholder="Password ..."
                className="form-control"
              />
            </div>

            <div className="form-group">
              <label>Re-enter Password</label>
              <input
                type="password"
                name="password2"
                value={password2}
                onChange={handlingInput}
                placeholder="Password ..."
                className="form-control"
              />
            </div>

            {/* Disable the button if the email is already registered or invalid */}
            <button
              className="login-button"
              type="submit"
              disabled={emailExists || !validateEmail(email)}
            >
              Signup
            </button>
          </form>

          {error && <div className="error-message">{error}</div>}
          <p className="signup-link">
            Already have an account? <Link to="../login/">Login</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default connect(null, { signup })(Signup);
