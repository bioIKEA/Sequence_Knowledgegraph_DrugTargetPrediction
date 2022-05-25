* 版本依赖
  * python 3.7.3
  * sklearn 0.0
  * numpy 1.21.0
  * pytorch 1.10.0
* 使用说明
  * 先解压data文件夹下的数据
  * 运行aff版本代码
    * 输出 模型权重：model_aff_round_ncv_m.pkl（n代表轮次，m代表交叉验证的折数）
    * 输出 auc指标：test_auc_aff_n （n代表交叉验证的折数）
  * 运行没有aff版本代码
    * 如果-t参数为unique
      * 输出 模型权重：model_round_unique_n.pkl（n代表交叉验证的折数）
      * 输出 auc指标：test_auc_unique_n （n代表交叉验证的折数）
    * -t不是unique
      * 输出 模型权重：model_round_ncv_m.pkl（n代表轮次，m代表交叉验证的折数）
      * 输出 auc指标：test_auc_n （n代表交叉验证的折数）