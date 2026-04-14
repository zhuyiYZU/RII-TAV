import hanlp


pipeline = hanlp.load('UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6')

# 测试输入文本
text = ['我爱自然语言处理']

# 输出分词和词性标注结果
result = pipeline(text)
print(result)
print(hanlp.pretrained.ALL)
