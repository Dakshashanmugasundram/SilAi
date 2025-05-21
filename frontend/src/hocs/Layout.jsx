import React, { useEffect } from "react";
import { connect } from "react-redux";
import { useLocation } from "react-router-dom";
import queryString from "query-string";
import Navbar from "../Component/navbar";
import Alert from "../Component/alert";
import { verify, getUser, googleLogin } from "../reducer/Actions";

const Layout = (props) => {
    let location = useLocation();

    useEffect(() => {
        const values = queryString.parse(location.search);
        const code = values.code;
        if (code) {
            props.googleLogin(code);
        } else {
            props.verify();
            props.getUser();
        }
    }, [location]);

    return (
        <div>
            {props.isAuthenticated && <Navbar />}
            {props.message ? <Alert message={props.message} /> : null}
            {props.children}
        </div>
    );
};

const mapStateToProps = (state) => ({
    message: state.AuthReducer.message,
    access: state.AuthReducer.access,
    refresh: state.AuthReducer.refresh,
    isAuthenticated: state.AuthReducer.isAuthenticated,
    user: state.AuthReducer.user
});

export default connect(mapStateToProps, { verify, getUser, googleLogin })(Layout);
