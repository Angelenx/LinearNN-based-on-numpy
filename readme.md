使用numpy实现的简单神经网络（仅包含线性层以及激活层），并实现批量梯度下降算法。
并将其应用于Concrete数据集测试。
根据结果，该模型对于预测混凝土抗压强度的准确率随着训练的进行不断波动上升，且变化幅度逐渐减小，最终趋于稳定，趋近值大致为0.82，而RMSE则在波动(波动幅度较小)下降，且变化幅度逐渐减小，最终趋于0.058。
根据实验对数据进行的预处理过程，得到最终单位为MPa的RMSE误差值5.8，效果不错。