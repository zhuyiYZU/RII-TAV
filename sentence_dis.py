import csv
import os
import time
from collections import defaultdict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TemporalAnchorManager:
    def __init__(self, anchor_groups, model_path, csv_path='keyword_history.csv', half_life=100):
        self.model = SentenceTransformer(model_path)
        self.anchor_groups = anchor_groups
        self.csv_path = csv_path


        self.anchor_embeddings = []
        for group in anchor_groups:
            group_embeddings = [self.model.encode(word, convert_to_tensor=True) for word in group]
            self.anchor_embeddings.append(group_embeddings)

        self.keyword_history = {
            'chat': defaultdict(dict),
            'rec': defaultdict(dict)
        }
        self.current_batch = 0
        self.half_life = half_life
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'batch', 'group', 'word',
                    'distance', 'weight', 'last_seen',
                    'current_score', 'delta_score'
                ])

    def calculate_distance(self, word):
        word_embedding = self.model.encode(word, convert_to_tensor=True)
        min_distances = []
        for group_embeddings in self.anchor_embeddings:
            distances = [1 - torch.nn.functional.cosine_similarity(word_embedding, anchor_emb, dim=0).item()
                         for anchor_emb in group_embeddings]
            min_distances.append(min(distances))
        closest_idx = np.argmin(min_distances)
        return {
            'word': word,
            'group': 'chat' if closest_idx == 0 else 'rec',
            'distance': min_distances[closest_idx]
        }

    def update_keywords(self, new_words,num):
        self.current_batch += 1
        print(f"Current batch: {self.current_batch}, Half-life: {self.half_life}")
        decay_factor = 0.5 ** (1 / self.half_life)
        print(f"Decay factor: {decay_factor:.6f}")                         #半衰
        # decay_factor = max(0, 1 - self.current_batch / self.half_life)
        # 保存上一轮得分
        previous_scores = {}
        for group in ['chat', 'rec']:
            previous_scores[group] = {
                word: (1 - record['distance']) * record['weight']
                for word, record in self.keyword_history[group].items()
            }

        seen_words = set(w['word'] for w in new_words)

        # 所有词统一衰减
        for group in ['chat', 'rec']:
            for word in self.keyword_history[group]:
                self.keyword_history[group][word]['weight'] *= decay_factor

        # 新词/出现词处理
        for word_info in new_words:
            word = word_info['word']
            group = word_info['group']
            distance = word_info['distance']

            if word in self.keyword_history[group]:
                self.keyword_history[group][word].update({
                    'last_seen': self.current_batch,
                    'distance': min(self.keyword_history[group][word]['distance'], distance),
                    'weight': max(self.keyword_history[group][word]['weight'], 1.0)
                })
            else:
                self.keyword_history[group][word] = {
                    'last_seen': self.current_batch,
                    'distance': distance,
                    'weight': 1.0
                }

        # 计算当前得分并保留每组 top_k
        top_words = {}
        for group in ['chat', 'rec']:
            scored = []
            for word, record in self.keyword_history[group].items():
                score = (1 - record['distance']) * record['weight']
                scored.append((word, score))
            # 排序并截断
            top_scored = sorted(scored, key=lambda x: -x[1])[:num]
            top_words[group] = top_scored

            # 只保留 top_k 的词
            keep_set = set(word for word, _ in top_scored)
            self.keyword_history[group] = {
                word: self.keyword_history[group][word]
                for word in keep_set
            }

        # 保存到 CSV（仅 top15）
        self._save_top_k_to_csv(top_words, previous_scores)

        return [w[0] for w in top_words['chat']], [w[0] for w in top_words['rec']]

    def _save_top_k_to_csv(self, top_words, previous_scores):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        records = []
        for group in ['chat', 'rec']:
            for word, current_score in top_words[group]:
                record = self.keyword_history[group][word]
                prev_score = previous_scores[group].get(word, 0.0)
                delta_score = current_score - prev_score
                records.append({
                    'timestamp': timestamp,
                    'batch': self.current_batch,
                    'group': group,
                    'word': word,
                    'distance': record['distance'],
                    'weight': record['weight'],
                    'last_seen': record['last_seen'],
                    'current_score': current_score,
                    'delta_score': delta_score
                })

        # 重写 CSV
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'batch', 'group', 'word',
                'distance', 'weight', 'last_seen',
                'current_score', 'delta_score'
            ])
            writer.writeheader()
            for row in records:
                writer.writerow(row)
