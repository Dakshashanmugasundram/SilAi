import { Navigate } from "react-router-dom";
import { connect } from "react-redux";

const PublicRoute = ({ children, isAuthenticated }) => {
    if (isAuthenticated) {
        return <Navigate to="/home" replace />;
    }
    return children;
};

const mapStateToProps = state => ({
    isAuthenticated: state.AuthReducer.isAuthenticated
});

export default connect(mapStateToProps)(PublicRoute);