import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from matplotlib import rcParams
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('D:/app/PyCharm Community Edition 2024.2.4/data/Bs1.xlsx')

# 特征选择和预处理
x = df.drop(columns=['Bs'])
y = df['Bs']



# 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

# 定义模型
model = XGBRegressor()
model.fit(x_train, y_train)
model.predict(x_test)
# 定义参数网格
param_grid = {
    "max_depth":np.arange(3,8,1),
    "n_estimators":np.arange(300,1001,100),
    "learning_rate":np.arange(0.1,1,0.1),

    }

# 网格搜索
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数重新训练模型
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

# 评估模型性能
y_pred = best_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("均方根误差 (RMSE):", rmse)
print("决定系数 (R2):", r2)


# 查看特征变量的特征重要性
features = x.columns
importances = best_model.feature_importances_
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
# 打印特征重要性
print(importances_df)

# 可视化特征重要性
rcParams['font.family'] = 'SimHei'
importances_df = importances_df.sort_values(by='特征重要性', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='特征重要性', y='特征名称', data=importances_df,color='skyblue')
plt.title('特征重要性分析', fontsize=18)
plt.xlabel('特征重要性', fontsize=14)
plt.ylabel('特征名称', fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()