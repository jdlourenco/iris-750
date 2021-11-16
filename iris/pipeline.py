from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class IrisPackage():
    def __init__(self):
        pass
    
    def get_pipeline(self):
        pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        return pipeline