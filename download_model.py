from modelscope.hub.snapshot_download import snapshot_download
import os
os.environ['MODELSCOPE_CACHE'] ='/home/zy-4090-1/hqq/recommendation_intent_recognition/model'
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

snapshot_download(model_name)
