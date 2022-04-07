# regression model: Baseline(linear regression) or Random Forest Regressor
def regression_model(X, y, test_size=0.2, model_type="baseline"):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    error = 0
    mdl = None
    
    if model_type == "baseline":
        # baseline model
        mdl = LinearRegression().fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
    elif model_type == "random_forest":
        # random forest model
        mdl = RandomForestRegressor(max_depth=6, random_state=0)
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
    
    return mdl, error