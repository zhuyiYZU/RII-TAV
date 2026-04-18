RII-TAV: Cross-domain Recommendation Intent Identification in Live-streaming Sales
This is the official implementation of the paper "Cross-domain Recommendation Intent Identification in Live-streaming Sales via Bullet Chat Streams". The RII-TAV framework addresses the challenges of rapid semantic drift and cross-domain distribution shifts in live-streaming bullet chats through target-aligned source reconstruction, automated agentic prompt induction, and dynamic verbalizer adaptation.

Environment Setup
1.Install OpenPrompt: The framework is strictly built on OpenPrompt v0.1.1.
2.pip install torch==2.2.2 pandas scikit-learn tqdm hanlp requests openpyxl
3.Model Preparation: Download your pre-trained backbone (e.g., chinese-roberta-wwm-ext) and place it in the models/ directory.
4.API Configuration: Ensure your XIAOHU_API_KEY is configured within the scripts for the automated prompt evolution process.

Usage Steps
1. 1.Offline Preparation: TASR & APIRIn this stage, the system aligns cross-domain knowledge and induces optimal prompts without accessing actual target comments.Target-Aligned Source Reconstruction (TASR): Uses an LLM to synthesize target-style pseudo-labeled data based on source labeling rules and target platform/product metadata, ensuring zero data leakage.Auto-Prompting Iterative Refinement (APIR):Run fewshot.py with the --auto_prompt flag to start the Generator-Reflector-Curator loop:Generator: Predicts intent with reasoning.Reflector: Critiques the prediction against pseudo-labels.Curator: Updates the "Learning Playbook" and distills it into a compact, deployment-efficient hard template.
2.
3. 2. Dataset PreparationOrganize your dataset according to the following structure (using the rec-dy subset as an example):Training Set: datasets/TextClassification/rec-dy/train.csv.Test Stream: datasets/veb/rec-related/all_test.csv.
4.
5. 3. Run Streaming Pipeline
