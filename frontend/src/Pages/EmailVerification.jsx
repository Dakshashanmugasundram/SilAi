import React from "react";
import { Navigate, useParams } from "react-router-dom";
import { useState } from "react";
import { connect } from "react-redux";
import { emailVerification } from "../reducer/Actions";

const EmailVerification = ({ emailVerification }) => {
    const [ status, setStatus ] = useState (false);
    const { key } = useParams();
    const handlingSubmit = (e) => {
        e.preventDefault();
        emailVerification( key );
        setStatus(true)
    }
    if (status) {
        return <Navigate to={"../login/"}></Navigate>
    }
    return (
        <div className="page-container">
        <h1 className="main-title">SilAI</h1>

        <div className="login-container">
            <div className="login-card">
            <h2 className="text-center">Activate Account</h2>
            <h5 className="text-center">
                Click the below link to activate your account
            </h5>

            <form className="login-form" onSubmit={handlingSubmit}>
                <div className="d-grid gap-2">
                <button className="login-button" type="submit">Activate Account</button>
                </div>
            </form>
            </div>
        </div>
        </div>

    )
}

export default connect(null, { emailVerification })(EmailVerification);