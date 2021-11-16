from iris.data import get_data, holdout
from iris.pipeline import IrisPackage
import joblib

class Trainer():
    def __init__(self):
        pass
    
    def fit(self):
        self.pipeline.fit(self.X_train, self.y_train)
    
    def save_model(self):
        joblib.dump(self.pipeline, 'pipeline.joblib')
    
    def train(self):
        print("load data")
        df = get_data()
        
        print("holdout")
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        
        print("create pipeline")
        self.pipeline = IrisPackage().get_pipeline()

        print("train model")
        self.fit()
        print(self.pipeline)
        
        print("savel model")
        self.save_model()
        
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()