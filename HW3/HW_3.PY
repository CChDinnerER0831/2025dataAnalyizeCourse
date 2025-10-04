import numpy as np 
import statsmodels.api as sm 

class Hw3:
    def __init__(self):
    
        pass

    def generate_data(seeds = 42):

        np.random.seed(seeds)
        datas = dict()
        datas['x_1']  = np.random.normal(0, 0.4, 100)
        datas['x_2']  = np.random.normal(0, 0.6, 100)
        datas['x_3']  = np.random.normal(0, 0.2, 100)
        datas['eps']  = np.random.normal(0, 0.05, 100)
        datas['y'] = 0.1 * datas['x_1'] + datas['x_2'] + 0.7 * datas['x_3'] + datas['eps']

        return datas
    
    def linear_regression(datas):

        datas = datas
        X = sm.add_constant(np.column_stack((datas['x_1'], datas['x_2'], datas['x_3'])))
        y = datas['y']
        model = sm.OLS(y, X)
        results = model.fit()
        return results.summary()


if __name__ == "__main__":
    data = Hw3.generate_data(42)
    print(Hw3.linear_regression(data))