''' 9/26 hw 2
download the Advertising.csv dataset from the course website. https://www.statlearning.com/s/Advertising.csv
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm  

Ad = pd.read_csv('HW2/Advertising.csv')

def solution(y_value,x_value):

    X = pd.DataFrame({'intercept':np.ones(Ad.shape[0]),x_value:Ad[x_value]})
    y = Ad[y_value]
    model = sm.OLS(y,X)       
    results = model.fit()
    CI = results.conf_int(alpha=0.05)  
    CI.columns = ['Lower 95% CI', 'Upper 95% CI']
    p_value = results.pvalues[x_value]

    if (p_value < 0.05):  
      print('H_0: Beta = 0 reject')
    else:
      print('H_0: Beta = 0 accept')

    print(pd.DataFrame({'Coefficient':results.params,'p-value':results.pvalues}),'\n',CI)

    return 0

def main(): 

    solution('sales', 'radio')# (y_value >> TV,radio,newspaper,sales , x_value >> TV,radio,newspaper,sales )
    solution('sales', 'newspaper')# (y_value >> TV,radio,newspaper,sales , x_value >> TV,radio,newspaper,sales )

    return 0

if __name__ == "__main__":
    main()