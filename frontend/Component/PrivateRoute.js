import { Navigate } from "react-router-dom";
import { connect } from "react-redux";

const PrivateRoute = ({ children, isAuthenticated }) => {
    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }
    return children;
};

const mapStateToProps = state => ({
    isAuthenticated: state.AuthReducer.isAuthenticated
});

export default connect(mapStateToProps)(PrivateRoute);