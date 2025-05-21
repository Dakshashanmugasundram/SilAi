import React from "react";
import { Navigate } from "react-router-dom";
import { connect } from "react-redux";
import { useState } from "react";
import { changePassword } from "../reducer/Actions";

const ChangePassword = ({ isAuthenticated, changePassword }) => {
    const [ formData, setFormData ] = useState ({
        new_password1: "",
        new_password2: "",
        old_password: ""
    });
    const { new_password1, new_password2, old_password } = formData;
    const handlingInput = (e) => setFormData({...formData, [e.target.name]: e.target.value});
    const handlingSubmit = (e) => {
        e.preventDefault();
        changePassword( new_password1, new_password2, old_password );
    }
    if ( !isAuthenticated && !localStorage.getItem('access') ) {
        return <Navigate to={"../login"}></Navigate>
    }
    return (
        <div className="page-container">
        <h1 className="main-title">SilAI</h1>

        <div className="login-container">
            <div className="login-card">
            <h2 className="text-center">Change Password</h2>

            <form className="login-form" onSubmit={handlingSubmit}>
                <div className="form-group">
                <label>New Password</label>
                <input
                    type="password"
                    name="new_password1"
                    value={new_password1}
                    onChange={handlingInput}
                    placeholder="New password ..."
                    className="form-control"
                />
                </div>

                <div className="form-group">
                <label>Re-enter Password</label>
                <input
                    type="password"
                    name="new_password2"
                    value={new_password2}
                    onChange={handlingInput}
                    placeholder="Re-enter new password ..."
                    className="form-control"
                />
                </div>

                <div className="form-group">
                <label>Old Password</label>
                <input
                    type="password"
                    name="old_password"
                    value={old_password}
                    onChange={handlingInput}
                    placeholder="Old password ..."
                    className="form-control"
                />
                </div>

                <button className="login-button" type="submit">Change Password</button>
            </form>
            </div>
        </div>
        </div>

    )
}

const mapStateToProps = ( state ) => {
    return {
        isAuthenticated: state.AuthReducer.isAuthenticated
    }
}

export default connect(mapStateToProps, { changePassword })(ChangePassword);