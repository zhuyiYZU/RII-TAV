"""
Lightweight ACE Framework Implementation
A modular system for Generate, Reflect, and Curate without large language models
"""
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re


class KnowledgeBase:
    """
    Manual/KnowledgeBase: Manages structured domain knowledge (ace_playbook.json)
    Stores strategies, rules, and tools in structured format
    Similar to the example: knowledge_base = {"task": {"description": "...", "steps": [...]}}
    """
    
    def __init__(self, playbook_path: str = "result/ace_playbook.json"):
        """
        Initialize knowledge base with playbook
        
        Args:
            playbook_path: Path to ace_playbook.json
        """
        self.playbook_path = playbook_path
        self.playbook = self._load_playbook()
        self.history = []  # Track changes to playbook
        self._initialize_strategies()
        
    def _load_playbook(self) -> Dict:
        """Load playbook from JSON file"""
        try:
            with open(self.playbook_path, 'r', encoding='utf-8') as f:
                playbook = json.load(f)
                # Ensure structure compatibility
                if "generation_strategies" not in playbook:
                    playbook["generation_strategies"] = {}
                if "error_patterns" not in playbook:
                    playbook["error_patterns"] = []
                return playbook
        except FileNotFoundError:
            print(f"Warning: {self.playbook_path} not found, creating default playbook")
            return {
                "version": "v1.0",
                "notes": [],
                "rules_top4": [],
                "rules_extra": [],
                "generation_strategies": {},
                "error_patterns": []
            }
    
    def _initialize_strategies(self):
        """Initialize generation strategies in structured format"""
        if "generation_strategies" not in self.playbook:
            self.playbook["generation_strategies"] = {}
        
        # Default strategies for prompt generation
        default_strategies = {
            "prompt_generation_small_dataset": {
                "description": "小数据集（<200条）的prompt生成策略",
                "steps": [
                    "使用硬模板格式",
                    "包含业务规则",
                    "包含3-5个示例",
                    "包含8-10个关键词",
                    "使用简洁模板风格"
                ],
                "conditions": {"num_examples": "<200"}
            },
            "prompt_generation_large_dataset": {
                "description": "大数据集（>=200条）的prompt生成策略",
                "steps": [
                    "使用硬模板格式",
                    "包含业务规则",
                    "包含2-3个示例",
                    "包含5-8个关键词",
                    "使用默认模板风格"
                ],
                "conditions": {"num_examples": ">=200"}
            },
            "prompt_generation_low_performance": {
                "description": "低性能时的prompt生成策略",
                "steps": [
                    "使用硬模板格式",
                    "包含业务规则",
                    "增加示例数量（5个）",
                    "增加关键词数量（10个）",
                    "包含更多区分词提示"
                ],
                "conditions": {"accuracy": "<0.7"}
            }
        }
        
        # Merge with existing strategies
        for key, value in default_strategies.items():
            if key not in self.playbook["generation_strategies"]:
                self.playbook["generation_strategies"][key] = value
    
    def save_playbook(self):
        """Save playbook to file"""
        with open(self.playbook_path, 'w', encoding='utf-8') as f:
            json.dump(self.playbook, f, ensure_ascii=False, indent=2)
        print(f"Playbook saved to {self.playbook_path}")
    
    def add_rule(self, rule: str, category: str = "rules_extra", priority: int = None):
        """
        Add a new rule to the playbook
        
        Args:
            rule: Rule text
            category: Category to add rule to ("rules_top4", "rules_extra", etc.)
            priority: Priority level (for top4 rules)
        """
        if category not in self.playbook:
            self.playbook[category] = []
        
        if category == "rules_top4" and priority is not None:
            self.playbook[category].insert(priority, rule)
        else:
            self.playbook[category].append(rule)
        
        self.history.append({
            "action": "add_rule",
            "rule": rule,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
    
    def update_rule(self, old_rule: str, new_rule: str, category: str = "rules_extra"):
        """Update an existing rule"""
        if category in self.playbook:
            for i, rule in enumerate(self.playbook[category]):
                if old_rule in rule:
                    self.playbook[category][i] = new_rule
                    self.history.append({
                        "action": "update_rule",
                        "old_rule": old_rule,
                        "new_rule": new_rule,
                        "category": category,
                        "timestamp": datetime.now().isoformat()
                    })
                    return True
        return False
    
    def get_rules(self, category: str = None) -> List[str]:
        """Get rules from playbook"""
        if category:
            return self.playbook.get(category, [])
        else:
            all_rules = []
            for key in ["rules_top4", "rules_extra"]:
                all_rules.extend(self.playbook.get(key, []))
            return all_rules
    
    def add_error_pattern(self, pattern: str, description: str, solution: str):
        """Add an error pattern for reflection"""
        if "error_patterns" not in self.playbook:
            self.playbook["error_patterns"] = []
        
        self.playbook["error_patterns"].append({
            "pattern": pattern,
            "description": description,
            "solution": solution,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_error_patterns(self) -> List[Dict]:
        """Get error patterns for reflection"""
        return self.playbook.get("error_patterns", [])


class Generator:
    """
    Generator: Selects strategies/templates from knowledge base based on task/input
    Similar to the example: generate_strategy(task) -> returns strategy steps
    Uses keyword matching and simple heuristics to select relevant strategies
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize generator with knowledge base
        
        Args:
            knowledge_base: KnowledgeBase instance
        """
        self.kb = knowledge_base
        self.class_labels = ["闲聊", "购买"]
    
    def generate_strategy(self, task: str, train_data: List = None, context: Dict = None) -> Dict:
        """
        Generate strategy for a given task (similar to example)
        
        Args:
            task: Task name (e.g., "prompt_generation")
            train_data: Training examples (optional)
            context: Additional context (e.g., performance metrics)
            
        Returns:
            Strategy dictionary with steps and parameters
        """
        # Get strategy from knowledge base
        strategies = self.kb.playbook.get("generation_strategies", {})
        
        # Select appropriate strategy based on task and conditions
        if task == "prompt_generation":
            return self._select_prompt_strategy(train_data, context, strategies)
        else:
            # Default strategy
            return {
                "description": f"默认策略 for {task}",
                "steps": ["使用默认配置"],
                "use_hard_template": True,
                "template_style": "default"
            }
    
    def _select_prompt_strategy(self, train_data: List, context: Dict, strategies: Dict) -> Dict:
        """Select prompt generation strategy based on conditions"""
        # Analyze training data
        data_stats = self._analyze_data(train_data) if train_data else {}
        num_examples = data_stats.get("num_examples", 0)
        
        # Select strategy based on conditions
        selected_strategy = None
        
        # Check low performance condition first
        if context and context.get("accuracy", 1.0) < 0.7:
            strategy_key = "prompt_generation_low_performance"
            if strategy_key in strategies:
                selected_strategy = strategies[strategy_key]
        
        # Check dataset size conditions
        if not selected_strategy:
            if num_examples < 200:
                strategy_key = "prompt_generation_small_dataset"
            else:
                strategy_key = "prompt_generation_large_dataset"
            
            if strategy_key in strategies:
                selected_strategy = strategies[strategy_key]
        
        # Convert strategy steps to execution parameters
        if selected_strategy:
            return self._strategy_to_params(selected_strategy, data_stats, context)
        else:
            # Default strategy
            return {
                "use_hard_template": True,
                "template_style": "default",
                "include_examples": True,
                "include_keywords": True,
                "num_examples": 3,
                "num_keywords": 8,
                "use_business_rules": True
            }
    
    def _strategy_to_params(self, strategy: Dict, data_stats: Dict, context: Dict) -> Dict:
        """Convert strategy steps to execution parameters"""
        steps = strategy.get("steps", [])
        params = {
            "use_hard_template": True,
            "template_style": "default",
            "include_examples": "示例" in "".join(steps),
            "include_keywords": "关键词" in "".join(steps),
            "num_examples": 3,
            "num_keywords": 8,
            "use_business_rules": "业务规则" in "".join(steps)
        }
        
        # Extract numbers from steps
        for step in steps:
            if "示例" in step:
                import re
                nums = re.findall(r'\d+', step)
                if nums:
                    params["num_examples"] = int(nums[0])
            if "关键词" in step:
                import re
                nums = re.findall(r'\d+', step)
                if nums:
                    params["num_keywords"] = int(nums[0])
        
        return params
    
    def select_strategy(self, train_data: List, context: Dict = None) -> Dict:
        """
        Select generation strategy (backward compatibility)
        
        Args:
            train_data: Training examples
            context: Additional context (e.g., performance metrics)
            
        Returns:
            Strategy dictionary with selected approach
        """
        return self.generate_strategy("prompt_generation", train_data, context)
    
    def _analyze_data(self, train_data: List) -> Dict:
        """Analyze training data characteristics"""
        examples_by_class = defaultdict(list)
        
        for item in train_data:
            if isinstance(item, tuple) and len(item) >= 2:
                text, label = item[0], item[1]
            elif hasattr(item, 'text_a') and hasattr(item, 'label'):
                text, label = item.text_a, item.label
            else:
                continue
            
            if isinstance(label, int):
                class_name = self.class_labels[label] if label < len(self.class_labels) else str(label)
            else:
                class_name = str(label)
            
            examples_by_class[class_name].append(text)
        
        return {
            "num_examples": len(train_data),
            "num_examples_per_class": {k: len(v) for k, v in examples_by_class.items()},
            "class_distribution": {k: len(v) / len(train_data) for k, v in examples_by_class.items()}
        }
    
    def generate_prompt(self, train_data: List, strategy: Dict = None) -> str:
        """
        Generate prompt based on selected strategy
        
        Args:
            train_data: Training examples
            strategy: Generation strategy (if None, will select automatically)
            
        Returns:
            Generated prompt template string
        """
        if strategy is None:
            strategy = self.select_strategy(train_data)
        
        # Import prompt generator
        from prompt_generator import PromptGenerator
        prompt_gen = PromptGenerator(self.kb.playbook_path)
        
        if strategy.get("use_hard_template", True):
            return prompt_gen.generate_hard_template(
                train_data=train_data,
                template_style=strategy.get("template_style", "default")
            )
        else:
            return prompt_gen.generate_prompt_template(
                train_data=train_data,
                include_examples=strategy.get("include_examples", True),
                include_keywords=strategy.get("include_keywords", True),
                num_examples=strategy.get("num_examples", 3),
                num_keywords=strategy.get("num_keywords", 8)
            )


class Reflector:
    """
    Reflector: Compares generated output with expected results
    Similar to the example: reflect_on_output(task, output, expected) -> feedback
    Uses simple rules to detect inconsistencies or errors
    Analyzes errors by pattern (e.g., calculation errors, understanding errors)
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize reflector with knowledge base
        
        Args:
            knowledge_base: KnowledgeBase instance
        """
        self.kb = knowledge_base
    
    def reflect_on_output(self, task: str, output: any, expected: any) -> str:
        """
        Reflect on a single output (similar to example)
        
        Args:
            task: Task name
            output: Generated output
            expected: Expected output
            
        Returns:
            Reflection feedback string
        """
        if output == expected:
            return "输出正确。"
        else:
            return f"输出不匹配：期望 {expected}，但得到 {output}。"
    
    def reflect(self, 
                predictions: List[int],
                true_labels: List[int],
                texts: List[str] = None,
                metrics: Dict = None) -> Dict:
        """
        Reflect on prediction results and identify issues
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            texts: Input texts (optional, for error analysis)
            metrics: Performance metrics (optional)
            
        Returns:
            Reflection dictionary with identified issues and suggestions
        """
        reflection = {
            "accuracy": accuracy_score(true_labels, predictions),
            "precision": precision_score(true_labels, predictions, average='macro'),
            "recall": recall_score(true_labels, predictions, average='macro'),
            "f1_score": f1_score(true_labels, predictions, average='macro'),
            "errors": [],
            "error_patterns": [],
            "suggestions": []
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        reflection["confusion_matrix"] = cm.tolist()
        
        # Identify errors
        errors = self._identify_errors(predictions, true_labels, texts)
        reflection["errors"] = errors
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(errors, texts)
        reflection["error_patterns"] = error_patterns
        
        # Generate suggestions
        suggestions = self._generate_suggestions(reflection, error_patterns)
        reflection["suggestions"] = suggestions
        
        return reflection
    
    def _identify_errors(self, predictions: List[int], true_labels: List[int], texts: List[str] = None) -> List[Dict]:
        """Identify individual errors"""
        errors = []
        for i, (pred, true_label) in enumerate(zip(predictions, true_labels)):
            if pred != true_label:
                error = {
                    "index": i,
                    "predicted": pred,
                    "true": true_label,
                    "text": texts[i] if texts else None
                }
                errors.append(error)
        return errors
    
    def _analyze_error_patterns(self, errors: List[Dict], texts: List[str] = None) -> List[Dict]:
        """Analyze patterns in errors"""
        if not errors or not texts:
            return []
        
        error_patterns = []
        
        # Group errors by class
        errors_by_class = defaultdict(list)
        for error in errors:
            errors_by_class[(error["predicted"], error["true"])].append(error)
        
        # Analyze common patterns
        for (pred_class, true_class), error_group in errors_by_class.items():
            if len(error_group) > 2:  # Only analyze patterns with multiple errors
                # Extract common keywords from error texts
                error_texts = [e["text"] for e in error_group if e["text"]]
                common_keywords = self._extract_common_keywords(error_texts)
                
                pattern = {
                    "type": f"误分类: {pred_class} -> {true_class}",
                    "count": len(error_group),
                    "common_keywords": common_keywords[:5],
                    "examples": error_texts[:3]
                }
                error_patterns.append(pattern)
        
        return error_patterns
    
    def _extract_common_keywords(self, texts: List[str], top_k: int = 5) -> List[str]:
        """Extract common keywords from error texts"""
        all_words = []
        for text in texts:
            # Extract Chinese words (2-6 characters)
            words = re.findall(r'[\u4e00-\u9fa5]{2,6}', text)
            all_words.extend(words)
        
        counter = Counter(all_words)
        return [word for word, _ in counter.most_common(top_k)]
    
    def _generate_suggestions(self, reflection: Dict, error_patterns: List[Dict]) -> List[str]:
        """Generate suggestions based on reflection"""
        suggestions = []
        
        # Low accuracy suggestions
        if reflection["accuracy"] < 0.7:
            suggestions.append("准确率较低，建议增加训练数据或调整prompt模板")
        
        # Low recall suggestions
        if reflection["recall"] < 0.6:
            suggestions.append("召回率较低，建议增加示例或关键词")
        
        # Low precision suggestions
        if reflection["precision"] < 0.6:
            suggestions.append("精确率较低，建议优化业务规则或增加区分度")
        
        # Error pattern suggestions
        for pattern in error_patterns:
            if pattern["count"] > 5:
                suggestions.append(
                    f"发现常见错误模式: {pattern['type']} ({pattern['count']}次), "
                    f"建议在业务规则中添加针对关键词: {', '.join(pattern['common_keywords'])}"
                )
        
        return suggestions


class Curator:
    """
    Curator: Updates playbook based on reflection results
    Similar to the example: update_knowledge_base(task, new_strategy)
    Can be manual or automated by flagging insights that may improve task-solving strategies
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize curator with knowledge base
        
        Args:
            knowledge_base: KnowledgeBase instance
        """
        self.kb = knowledge_base
    
    def update_knowledge_base(self, task: str, new_strategy: Dict, auto_save: bool = True):
        """
        Update knowledge base with new strategy (similar to example)
        
        Args:
            task: Task name
            new_strategy: New strategy dictionary with description and steps
            auto_save: Whether to automatically save playbook
        """
        if "generation_strategies" not in self.kb.playbook:
            self.kb.playbook["generation_strategies"] = {}
        
        self.kb.playbook["generation_strategies"][task] = new_strategy
        
        self.kb.history.append({
            "action": "update_strategy",
            "task": task,
            "strategy": new_strategy,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"已更新任务 '{task}' 的知识库。")
        
        if auto_save:
            self.kb.save_playbook()
    
    def curate(self, reflection: Dict, auto_update: bool = False) -> Dict:
        """
        Update playbook based on reflection results
        
        Args:
            reflection: Reflection results from Reflector
            auto_update: If True, automatically update playbook; if False, return suggestions
            
        Returns:
            Dictionary with update actions taken or suggested
        """
        updates = {
            "suggested_rules": [],
            "suggested_updates": [],
            "suggested_strategies": {},
            "applied_updates": []
        }
        
        # Analyze error patterns and suggest rules
        for pattern in reflection.get("error_patterns", []):
            if pattern["count"] > 3:
                # Suggest new rule based on error pattern
                suggested_rule = self._suggest_rule_from_pattern(pattern)
                updates["suggested_rules"].append(suggested_rule)
                
                if auto_update:
                    self.kb.add_rule(suggested_rule["rule"], category="rules_extra")
                    updates["applied_updates"].append(f"Added rule: {suggested_rule['rule'][:50]}...")
        
        # Update based on suggestions
        for suggestion in reflection.get("suggestions", []):
            if "关键词" in suggestion and auto_update:
                # Extract keywords and create rule
                keywords_match = re.search(r'关键词: ([^,]+(?:, [^,]+)*)', suggestion)
                if keywords_match:
                    keywords = keywords_match.group(1).split(", ")
                    rule = f"针对关键词 {', '.join(keywords)} 的误分类，需要特别关注"
                    self.kb.add_rule(rule, category="rules_extra")
                    updates["applied_updates"].append(f"Added rule based on suggestion: {rule[:50]}...")
        
        # Suggest strategy updates based on performance
        if reflection.get("accuracy", 1.0) < 0.7:
            new_strategy = {
                "description": "改进后的prompt生成策略（基于低准确率）",
                "steps": [
                    "使用硬模板格式",
                    "包含业务规则",
                    "增加示例数量（5个）",
                    "增加关键词数量（10个）",
                    "包含更多区分词提示"
                ],
                "conditions": {"accuracy": "<0.7"}
            }
            updates["suggested_strategies"]["prompt_generation_low_performance"] = new_strategy
        
        # Save playbook if updates were applied
        if auto_update and updates["applied_updates"]:
            self.kb.save_playbook()
        
        return updates
    
    def _suggest_rule_from_pattern(self, pattern: Dict) -> Dict:
        """Suggest a rule based on error pattern"""
        pattern_type = pattern.get("type", "")
        keywords = pattern.get("common_keywords", [])
        
        if "购买" in pattern_type and "闲聊" in pattern_type:
            # Misclassification between classes
            rule_text = f"注意区分：包含关键词 {', '.join(keywords[:3])} 的表达需要根据上下文判断，"
            rule_text += "明确购买意向的归为购买，否则归为闲聊"
        else:
            rule_text = f"针对关键词 {', '.join(keywords[:3])} 的分类规则需要进一步明确"
        
        return {
            "rule": rule_text,
            "category": "rules_extra",
            "priority": "low",
            "source": "error_pattern_analysis"
        }


class ACEFramework:
    """
    Main ACE Framework: Orchestrates Generate, Reflect, and Curate
    """
    
    def __init__(self, playbook_path: str = "result/ace_playbook.json"):
        """
        Initialize ACE Framework
        
        Args:
            playbook_path: Path to playbook JSON file
        """
        self.kb = KnowledgeBase(playbook_path)
        self.generator = Generator(self.kb)
        self.reflector = Reflector(self.kb)
        self.curator = Curator(self.kb)
    
    def run_cycle(self, 
                  train_data: List,
                  test_data: List = None,
                  predictions: List[int] = None,
                  true_labels: List[int] = None,
                  auto_curate: bool = False) -> Dict:
        """
        Run one ACE cycle: Generate -> Reflect -> Curate
        
        Args:
            train_data: Training examples
            test_data: Test examples (optional)
            predictions: Model predictions (optional, if None will need to run model)
            true_labels: True labels for test data
            auto_curate: Whether to automatically update playbook
            
        Returns:
            Dictionary with results from all stages
        """
        results = {
            "generation": {},
            "reflection": {},
            "curation": {}
        }
        
        # 1. Generate: Select strategy and generate prompt
        print("=" * 50)
        print("[ACE Framework] Stage 1: Generate")
        print("=" * 50)
        
        # Use generate_strategy method (similar to example)
        strategy = self.generator.generate_strategy("prompt_generation", train_data)
        prompt = self.generator.generate_prompt(train_data, strategy)
        results["generation"] = {
            "strategy": strategy,
            "strategy_steps": self.kb.playbook.get("generation_strategies", {}).get("prompt_generation", {}).get("steps", []),
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt
        }
        print(f"Selected strategy: {strategy.get('description', 'N/A')}")
        print(f"Strategy steps: {results['generation'].get('strategy_steps', [])}")
        print(f"Generated prompt (first 200 chars): {prompt[:200]}...")
        
        # 2. Reflect: Analyze results if predictions are available
        if predictions is not None and true_labels is not None:
            print("\n" + "=" * 50)
            print("[ACE Framework] Stage 2: Reflect")
            print("=" * 50)
            texts = [item[0] if isinstance(item, tuple) else getattr(item, 'text_a', '') 
                    for item in test_data] if test_data else None
            reflection = self.reflector.reflect(predictions, true_labels, texts)
            results["reflection"] = reflection
            
            # Use reflect_on_output for individual errors (similar to example)
            if reflection["errors"]:
                print("Sample error reflections:")
                for error in reflection["errors"][:3]:  # Show first 3 errors
                    feedback = self.reflector.reflect_on_output(
                        "classification",
                        error["predicted"],
                        error["true"]
                    )
                    print(f"  - {feedback}")
            
            print(f"Accuracy: {reflection['accuracy']:.4f}")
            print(f"F1 Score: {reflection['f1_score']:.4f}")
            print(f"Found {len(reflection['errors'])} errors")
            print(f"Identified {len(reflection['error_patterns'])} error patterns")
            print(f"Suggestions: {len(reflection['suggestions'])}")
        
            # 3. Curate: Update playbook based on reflection
            if reflection.get("errors"):
                print("\n" + "=" * 50)
                print("[ACE Framework] Stage 3: Curate")
                print("=" * 50)
                curation = self.curator.curate(reflection, auto_update=auto_curate)
                results["curation"] = curation
                print(f"Suggested {len(curation['suggested_rules'])} new rules")
                
                # Use update_knowledge_base for strategy updates (similar to example)
                if auto_curate and curation.get("suggested_strategies"):
                    for task, strategy in curation["suggested_strategies"].items():
                        self.curator.update_knowledge_base(task, strategy, auto_save=False)
                    self.kb.save_playbook()
                    print(f"Applied {len(curation.get('suggested_strategies', {}))} strategy updates to playbook")
                elif auto_curate:
                    print(f"Applied {len(curation['applied_updates'])} updates to playbook")
                else:
                    print("Updates suggested but not applied (set auto_curate=True to apply)")
        
        return results


if __name__ == "__main__":
    # Example usage (similar to the provided example)
    print("ACE Framework Example Usage")
    print("=" * 50)
    
    # Initialize framework
    ace = ACEFramework()
    
    # Example 1: Generate strategy (similar to generate_strategy in example)
    print("\n[Example 1] Generate Strategy")
    print("-" * 50)
    task = "prompt_generation"
    train_data = [
        ("有独立小包装的巧克力吗", 1),
        ("普通话比我还标准", 0),
        ("牛肉干有吗辣不辣", 1),
        ("欢迎江总出场", 0)
    ]
    strategy = ace.generator.generate_strategy(task, train_data)
    print(f"任务 '{task}' 的策略：")
    print(f"  描述: {strategy.get('description', 'N/A')}")
    print(f"  参数: {strategy}")
    
    # Example 2: Reflect on output (similar to reflect_on_output in example)
    print("\n[Example 2] Reflect on Output")
    print("-" * 50)
    output = 1  # Predicted label
    expected = 0  # True label
    reflection = ace.reflector.reflect_on_output("classification", output, expected)
    print(f"反思结果: {reflection}")
    
    # Example 3: Update knowledge base (similar to update_knowledge_base in example)
    print("\n[Example 3] Update Knowledge Base")
    print("-" * 50)
    new_strategy = {
        "description": "改进后的prompt生成策略",
        "steps": [
            "使用硬模板格式",
            "包含业务规则",
            "包含5个示例",
            "包含10个关键词",
            "使用默认模板风格"
        ],
        "conditions": {"num_examples": "<200", "accuracy": "<0.7"}
    }
    ace.curator.update_knowledge_base("prompt_generation_improved", new_strategy, auto_save=True)
    
    # Example 4: Full ACE cycle
    print("\n[Example 4] Full ACE Cycle")
    print("-" * 50)
    # Example test results
    predictions = [1, 0, 1, 1]  # Last one is wrong
    true_labels = [1, 0, 1, 0]
    test_data = train_data
    
    # Run ACE cycle
    results = ace.run_cycle(
        train_data=train_data,
        test_data=test_data,
        predictions=predictions,
        true_labels=true_labels,
        auto_curate=False  # Set to True to auto-update playbook
    )
    
    print("\n" + "=" * 50)
    print("ACE Cycle Complete")
    print("=" * 50)
    print("\n总结:")
    print(f"- 生成策略: {len(results.get('generation', {}).get('strategy_steps', []))} 个步骤")
    print(f"- 反思结果: 准确率 {results.get('reflection', {}).get('accuracy', 0):.4f}")
    print(f"- 整合建议: {len(results.get('curation', {}).get('suggested_rules', []))} 条新规则")

