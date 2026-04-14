"""
Auto Prompt Generator based on training data and business rules
Automatically generates dataset-specific prompts from training examples
"""
import json
import pandas as pd
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import os


class PromptGenerator:
    """Generate dataset-specific prompts from training data and business rules"""
    
    def __init__(self, playbook_path: str = "result/ace_playbook.json"):
        """
        Initialize prompt generator with business rules
        
        Args:
            playbook_path: Path to ace_playbook.json containing business rules
        """
        self.playbook_path = playbook_path
        self.playbook = self._load_playbook()
        self.class_labels = ["闲聊", "购买"]  # Default labels
        
    def _load_playbook(self) -> Dict:
        """Load business rules from ace_playbook.json"""
        try:
            with open(self.playbook_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.playbook_path} not found, using default rules")
            return {
                "version": "v1.0",
                "notes": [],
                "rules_top4": [],
                "rules_extra": []
            }
    
    def extract_examples_from_data(self, train_data: List, num_examples_per_class: int = 5) -> Dict[str, List[str]]:
        """
        Extract representative examples from training data
        
        Args:
            train_data: List of InputExample objects or tuples (text, label)
            num_examples_per_class: Number of examples to extract per class
            
        Returns:
            Dictionary mapping class labels to example texts
        """
        examples_by_class = defaultdict(list)
        
        # Handle different input formats
        for item in train_data:
            if hasattr(item, 'text_a') and hasattr(item, 'label'):
                text = item.text_a
                label = item.label
            elif isinstance(item, tuple) and len(item) >= 2:
                text = item[0]
                label = item[1]
            else:
                continue
                
            # Map label index to class name
            if isinstance(label, int):
                class_name = self.class_labels[label] if label < len(self.class_labels) else str(label)
            else:
                class_name = str(label)
            
            examples_by_class[class_name].append(text)
        
        # Sample examples for each class
        sampled_examples = {}
        for class_name, texts in examples_by_class.items():
            # Remove duplicates and sample
            unique_texts = list(set(texts))
            sampled = random.sample(unique_texts, min(num_examples_per_class, len(unique_texts)))
            sampled_examples[class_name] = sampled
        
        return sampled_examples
    
    def extract_keywords_from_data(self, train_data: List, top_k: int = 10) -> Dict[str, List[str]]:
        """
        Extract distinguishing keywords/phrases from training data
        
        Args:
            train_data: List of training examples
            top_k: Number of keywords to extract per class
            
        Returns:
            Dictionary mapping class labels to keyword lists
        """
        from collections import Counter
        import re
        
        keywords_by_class = defaultdict(list)
        
        for item in train_data:
            if hasattr(item, 'text_a') and hasattr(item, 'label'):
                text = item.text_a
                label = item.label
            elif isinstance(item, tuple) and len(item) >= 2:
                text = item[0]
                label = item[1]
            else:
                continue
            
            # Map label to class name
            if isinstance(label, int):
                class_name = self.class_labels[label] if label < len(self.class_labels) else str(label)
            else:
                class_name = str(label)
            
            # Extract Chinese words/phrases (2-6 characters)
            # Handle Chinese text without spaces
            text = str(text).strip()
            
            # Extract 2-6 character phrases from Chinese text
            # Use sliding window to extract phrases
            for i in range(len(text)):
                for length in [2, 3, 4, 5, 6]:
                    if i + length <= len(text):
                        phrase = text[i:i+length]
                        # Only keep Chinese characters
                        if re.match(r'^[\u4e00-\u9fa5]+$', phrase):
                            keywords_by_class[class_name].append(phrase)
            
            # Also extract individual words if text has spaces
            if ' ' in text or '\t' in text:
                words = re.split(r'[\s\t]+', text)
                for word in words:
                    word = word.strip()
                    if 2 <= len(word) <= 6 and re.match(r'^[\u4e00-\u9fa5]+$', word):
                        keywords_by_class[class_name].append(word)
        
        # Get top keywords for each class
        top_keywords = {}
        for class_name, keywords in keywords_by_class.items():
            counter = Counter(keywords)
            # Get top keywords, but filter out very common ones
            most_common = counter.most_common(top_k * 2)  # Get more to filter
            # Filter: keep keywords that appear at least 2 times or are in top_k
            filtered = [word for word, count in most_common[:top_k] if count >= 2] or \
                      [word for word, _ in most_common[:top_k]]
            top_keywords[class_name] = filtered[:top_k]
        
        return top_keywords
    
    def build_business_rules_text(self) -> str:
        """Build business rules text from playbook"""
        rules_text = ""
        
        if "rules_top4" in self.playbook and self.playbook["rules_top4"]:
            rules_text += "判别规则要点：\n"
            for i, rule in enumerate(self.playbook["rules_top4"], 1):
                rules_text += f"{i}. {rule}\n"
        
        if "rules_extra" in self.playbook and self.playbook["rules_extra"]:
            rules_text += "\n补充规则：\n"
            for rule in self.playbook["rules_extra"]:
                rules_text += f"- {rule}\n"
        
        return rules_text.strip()
    
    def generate_prompt_template(self, 
                                 train_data: List,
                                 include_examples: bool = True,
                                 include_keywords: bool = True,
                                 num_examples: int = 3,
                                 num_keywords: int = 8) -> str:
        """
        Generate a complete prompt template based on training data
        
        Args:
            train_data: List of training examples
            include_examples: Whether to include example texts
            include_keywords: Whether to include distinguishing keywords
            num_examples: Number of examples per class to include
            num_keywords: Number of keywords per class to include
            
        Returns:
            Generated prompt template string in ManualTemplate format
        """
        # Extract examples and keywords from training data
        examples = self.extract_examples_from_data(train_data, num_examples_per_class=num_examples)
        keywords = self.extract_keywords_from_data(train_data, top_k=num_keywords)
        
        # Build prompt components
        prompt_parts = []
        
        # Base instruction
        base_instruction = "判断弹幕类别，可选：闲聊 / 购买。"
        prompt_parts.append(base_instruction)
        
        # Business rules from playbook
        rules_text = self.build_business_rules_text()
        if rules_text:
            # Escape quotes in rules text for JSON format
            rules_text_escaped = rules_text.replace('"', '\\"').replace('\n', ' ')
            prompt_parts.append(f'{{"text":"{rules_text_escaped}","shortenable":"True"}}')
        
        # Distinguishing keywords
        if include_keywords and keywords:
            keyword_text = ""
            for class_name in ["闲聊", "购买"]:
                if class_name in keywords and keywords[class_name]:
                    class_keywords = keywords[class_name][:num_keywords]
                    # Limit each keyword display length
                    display_keywords = [kw[:10] for kw in class_keywords if kw]
                    if display_keywords:
                        keyword_text += f"区分词提示：更偏「{class_name}」的常见表达有：{', '.join(display_keywords)}。"
            
            if keyword_text:
                # Escape quotes
                keyword_text_escaped = keyword_text.replace('"', '\\"')
                prompt_parts.append(f'{{"text":"{keyword_text_escaped}","shortenable":"True"}}')
        
        # Example texts
        if include_examples and examples:
            example_text = ""
            for class_name in ["闲聊", "购买"]:
                if class_name in examples and examples[class_name]:
                    # Take first example for this class
                    example = str(examples[class_name][0])
                    # Truncate if too long
                    if len(example) > 30:
                        example = example[:27] + "..."
                    # Escape quotes and brackets
                    example = example.replace('"', '\\"').replace('[', '\\[').replace(']', '\\]')
                    example_text += f"示例（{class_name}）：弹幕「{example}」。"
            
            if example_text:
                # Escape quotes
                example_text_escaped = example_text.replace('"', '\\"')
                prompt_parts.append(f'{{"text":"{example_text_escaped}","shortenable":"True"}}')
        
        # Final instruction with placeholder
        prompt_parts.append('目标弹幕是："{"placeholder":"text_a","shortenable":"True"}"。请在{"mask"}位置输出最合适的类别名称。')
        
        # Combine all parts
        prompt_template = "".join(prompt_parts)
        
        return prompt_template
    
    def generate_hard_template(self, 
                               train_data: List = None,
                               template_style: str = "default") -> str:
        """
        Generate a hard template (简洁硬模板) similar to the provided format
        
        Args:
            train_data: Optional training data to customize template
            template_style: Style of template ("default", "customer_service", "analyst", "classifier")
            
        Returns:
            Hard template string in ManualTemplate format
        """
        # Template variations based on style
        # Note: Using raw strings to preserve exact format for ManualTemplate
        templates = {
            "default": '你是一名智能客服助手，负责辅助判断电商直播弹幕是否表达了用户购买兴趣。可选类别：闲聊 / 购买。目标弹幕是："{"placeholder":"text_a","shortenable":"True"}"。用户表达的含义具有{"mask"}倾向。请只输出一个类别名，不要输出多余文字。',
            
            "customer_service": '你是一名智能客服助手，负责辅助判断电商直播弹幕是否表达了用户购买兴趣。可选类别：闲聊 / 购买。目标弹幕是："{"placeholder":"text_a","shortenable":"True"}"。用户表达的含义具有{"mask"}倾向。请只输出一个类别名，不要输出多余文字。',
            
            "analyst": '你是一位直播电商的智能语义分析师，专注于识别用户弹幕中是否包含购买意图。现在有一条弹幕："{"placeholder":"text_a","shortenable":"True"}"。请判断该用户言论是否体现出{"mask"}动机。',
            
            "classifier": '判断弹幕类别，可选：闲聊 / 购买。目标弹幕是："{"placeholder":"text_a","shortenable":"True"}"。请在{"mask"}位置输出最合适的类别名称。',
            
            "intent": '你的工作是为每条电商直播弹幕打上是否"推荐相关"的标签。给定弹幕："{"placeholder":"text_a","shortenable":"True"}"。请判断这是否属于{"mask"}表达。'
        }
        
        # Select template based on style
        if template_style in templates:
            base_template = templates[template_style]
        else:
            base_template = templates["default"]
        
        # If training data is provided, we can customize the template slightly
        # For now, return the base template (can be extended later)
        return base_template
    
    def generate_and_save_hard_template(self,
                                        output_path: str,
                                        train_data: List = None,
                                        template_style: str = "default") -> str:
        """
        Generate and save hard template
        
        Args:
            output_path: Path to save the generated hard template
            train_data: Optional training data
            template_style: Style of template
            
        Returns:
            Generated hard template string
        """
        hard_template = self.generate_hard_template(train_data, template_style)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(hard_template + '\n')
        
        print(f"Generated hard template saved to: {output_path}")
        print(f"Hard template content:\n{hard_template}")
        
        return hard_template
    
    def generate_and_save_prompt(self, 
                                train_data: List,
                                output_path: str,
                                include_examples: bool = True,
                                include_keywords: bool = True,
                                num_examples: int = 3,
                                num_keywords: int = 8) -> str:
        """
        Generate prompt and save to file
        
        Args:
            train_data: List of training examples
            output_path: Path to save the generated prompt template
            include_examples: Whether to include example texts
            include_keywords: Whether to include distinguishing keywords
            num_examples: Number of examples per class
            num_keywords: Number of keywords per class
            
        Returns:
            Generated prompt template string
        """
        prompt_template = self.generate_prompt_template(
            train_data=train_data,
            include_examples=include_examples,
            include_keywords=include_keywords,
            num_examples=num_examples,
            num_keywords=num_keywords
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file (append if file exists, or create new)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt_template + '\n')
        
        print(f"Generated prompt saved to: {output_path}")
        print(f"Prompt preview (first 200 chars): {prompt_template[:200]}...")
        
        return prompt_template
    
    def generate_from_csv(self, 
                         csv_path: str,
                         output_path: str,
                         text_col: int = 2,
                         label_col: int = 3,
                         num_samples: int = None,
                         use_hard_template: bool = False,
                         template_style: str = "default") -> str:
        """
        Generate prompt from CSV file
        
        Args:
            csv_path: Path to training CSV file
            output_path: Path to save generated prompt
            text_col: Column index for text (default: 2)
            label_col: Column index for label (default: 3)
            num_samples: Number of samples to use (None = all)
            use_hard_template: If True, generate hard template instead of detailed prompt
            template_style: Style for hard template ("default", "customer_service", "analyst", etc.)
            
        Returns:
            Generated prompt template string
        """
        # Read CSV
        df = pd.read_csv(csv_path, header=None)
        
        # Sample if needed
        if num_samples and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42)
        
        # Convert to training data format
        train_data = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            label = int(row[label_col])
            train_data.append((text, label))
        
        # Generate prompt or hard template
        if use_hard_template:
            return self.generate_and_save_hard_template(
                output_path=output_path,
                train_data=train_data,
                template_style=template_style
            )
        else:
            return self.generate_and_save_prompt(
                train_data=train_data,
                output_path=output_path
            )


if __name__ == "__main__":
    # Example usage
    generator = PromptGenerator()
    
    # Generate from CSV
    csv_path = "datasets/TextClassification/rec-dy/train.csv"
    output_path = "scripts/TextClassification/rec-dy/auto_generated_template.txt"
    
    if os.path.exists(csv_path):
        # Generate hard template (简洁硬模板)
        generator.generate_from_csv(
            csv_path=csv_path,
            output_path=output_path,
            num_samples=100,  # Use first 100 samples
            use_hard_template=True,  # Use hard template
            template_style="default"  # or "customer_service", "analyst", "classifier", "intent"
        )
    else:
        print(f"CSV file not found: {csv_path}")

