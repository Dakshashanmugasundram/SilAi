import React from "react";
import { Navigate, useParams } from "react-router-dom";
import { useState } from "react";
import { connect } from "react-redux";
import { resetPasswordConfirm } from "../reducer/Actions";

const ResetPasswordConfirm = ({ resetPasswordConfirm }) => {
    const [ status, setStatus ] = useState (false);
    const { uid, token } = useParams();
    const [ formData, setFormData ] = useState ({
        new_password1: "",
        new_password2: ""
    });
    const { new_password1, new_password2 } = formData;
    const handlingInput = (e) => setFormData({...formData, [e.target.name]: e.target.value});
    const handlingSubmit = (e) => {
        e.preventDefault();
        resetPasswordConfirm( uid, token, new_password1, new_password2);
        setStatus(true);
    }
    if (status) {
        return <Navigate to={"../login/"}></Navigate>
    }
    return (
    <div className="page-container">
    <h1 className="main-title">SilAI</h1>

    <div className="login-container">
        <div className="login-card">
        <h2 className="text-center">Set Password</h2>

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

            <button className="login-button" type="submit">Set Password</button>
        </form>
        </div>
    </div>
    </div>
    )
}

export default connect(null, { resetPasswordConfirm })(ResetPasswordConfirm);