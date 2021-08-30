# 功能说明

本脚本会使用分词、textCNN方法自动进行二分类/多分类的文本分类。

运行脚本：
   sh textCNN/run_textCNN_train.sh    
   // 配置脚本参数，在 run_textCNN_train.sh 中修改

# 注意事项
1.样本格式为：label, content，并且首行为这两个字段    
2. stop words配置目录：data/stopwords.txt    
3. 脚本支持二分类及多分类，多分类情况下注意只能有其中一个类为正样本，不支持一个样本同时属于多个类别。    
4. 模型引用地址：https://github.com/brightmart/text_classification/blob/master/a02_TextCNN/p7_TextCNN_model.py    
   如需使用 FastText、textRNN 、textRCNN、transformer等，可复用代码中的样本生成（提供了h5py、pickle保存），将代码引入调用即可。


# 依赖的环境
pip install --upgrade pip    
pip install -i https://mirrors.aliyun.com/pypi/simple numpy    
pip install -i https://mirrors.aliyun.com/pypi/simple sklearn    
pip install -i https://mirrors.aliyun.com/pypi/simple pandas    
pip install -i https://mirrors.aliyun.com/pypi/simple word2vec    
pip install -i https://mirrors.aliyun.com/pypi/simple numba
pip install -i https://mirrors.aliyun.com/pypi/simple jieba
