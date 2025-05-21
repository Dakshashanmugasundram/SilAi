import React from "react";
import { Link,Navigate } from "react-router-dom";
import { useState } from "react";
import { connect } from "react-redux";
import { resetPassword } from "../reducer/Actions";


const ResetPassword = ({ resetPassword }) => {
    const [ status, setStatus ] = useState (false);
    const [ formData, setFormData ] = useState ({
        email: ""
    });
    const { email } = formData;
    const handlingInput = (e) => setFormData({...formData, [e.target.name]: e.target.value});
    const handlingSubmit = (e) => {
        e.preventDefault();
        resetPassword( email );
        setStatus(true)
    }
    if (status) {
        return <Navigate to={"../"}></Navigate>
    }
    return (
        <div className="page-container">
        <h1 className="main-title">SilAI</h1>
  
        <div className="login-container">
          <div className="login-card">
            <h2 className="text-center">Reset Password</h2>
            <h5 className="text-center">
              Please input your registered email. The link to set your new password will be sent to your email.
            </h5>
  
            <form className="login-form" onSubmit={handlingSubmit}>
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
              </div>
              <button className="login-button" type="submit">Send Link</button>
            </form>
            <p className="signup-link">
                Already have an account? <Link to="../login/">Login</Link>
            </p>
          </div>
        </div>
      </div>
    )
}

export default connect(null, { resetPassword })(ResetPassword);