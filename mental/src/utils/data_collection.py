#!/usr/bin/env python3
"""
Psychology abstract collection utilities.
Collect psychology-related paper abstracts from multiple sources.
"""

import requests
import feedparser
import json
import time
import random
from typing import List, Dict, Optional
from pathlib import Path
import csv

class PsychologyAbstractCollector:
    """Psychology abstract collector."""

    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # create subdirectories
        (self.output_dir / "abstracts").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

    def collect_from_arxiv(self) -> List[Dict]:
        """Collect computational psychology papers from ArXiv."""
        print("Collecting computational psychology papers from ArXiv...")

        base_url = "http://export.arxiv.org/api/query"

        # psychology-related search queries
        psychology_queries = [
            '"cognitive psychology"',
            '"computational psychology"',
            '"cognitive modeling"',
            '"psychological modeling"',
            '"cognitive science"',
            '"human cognition" modeling',
            '"behavioral modeling"',
            '"emotion recognition"',
            '"mental health" AI',
            '"psychological experiment" computational'
        ]

        all_papers = []

        for query in psychology_queries:
            print(f"Searching: {query}")

            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': 20,  # limit to 20 results per query
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            try:
                response = requests.get(base_url, params=params)
                feed = feedparser.parse(response.content)

                for entry in feed.entries:
                    paper = {
                        'source': 'arxiv',
                        'id': entry.id.split('/')[-1],
                        'title': entry.title.strip(),
                        'abstract': entry.summary.strip(),
                        'authors': [author.name for author in entry.authors],
                        'categories': [tag.term for tag in entry.tags if hasattr(tag, 'term')],
                        'published': entry.published,
                        'url': entry.link,
                        'search_query': query
                    }
                    all_papers.append(paper)

                # polite delay between requests
                time.sleep(1)

            except Exception as e:
                print(f"Error searching '{query}': {e}")
                continue

        # deduplicate
        unique_papers = []
        seen_ids = set()

        for paper in all_papers:
            if paper['id'] not in seen_ids:
                seen_ids.add(paper['id'])
                unique_papers.append(paper)

        print(f"Collected {len(unique_papers)} unique papers from ArXiv")
        return unique_papers

    def collect_sample_classic_papers(self) -> List[Dict]:
        """Return a small sample set of classic psychology paper abstracts."""

        classic_papers = [
            {
                'source': 'sample',
                'id': 'bandura_1977',
                'title': 'Self-efficacy: Toward a unifying theory of behavioral change',
                'abstract': 'This article presents a cognitive theory of motivation that integrates concepts from social learning and cognitive approaches. The theory proposes that self-efficacy expectations—beliefs about one\'s capabilities to organize and execute the courses of action required to manage prospective situations—serve as a major determinant of people\'s choice of activities, effort expenditure, persistence, and thought patterns. Self-efficacy expectations are distinguished from related constructs such as outcome expectations, self-concept, and perceived control. Empirical support for the theory is reviewed, and implications for behavioral change are discussed.',
                'authors': ['Albert Bandura'],
                'subfield': 'cognitive_psychology',
                'published': '1977',
                'url': '',
                'key_concepts': ['self-efficacy', 'cognitive theory', 'motivation', 'behavioral change']
            },
            {
                'source': 'sample',
                'id': 'festinger_1957',
                'title': 'A Theory of Cognitive Dissonance',
                'abstract': 'Cognitive dissonance theory proposes that individuals experience psychological discomfort when they hold two or more contradictory beliefs, ideas, or values, or when their beliefs are inconsistent with their actions. This dissonance creates a motivational drive to reduce the inconsistency by changing beliefs, attitudes, or behaviors. The theory explains various phenomena including attitude change, decision justification, and effort justification. Cognitive dissonance can be reduced through changing cognitions, adding new cognitions, or decreasing the importance of conflicting cognitions.',
                'authors': ['Leon Festinger'],
                'subfield': 'social_psychology',
                'published': '1957',
                'url': '',
                'key_concepts': ['cognitive dissonance', 'attitude change', 'psychological discomfort', 'motivation']
            },
            {
                'source': 'sample',
                'id': 'piaget_1952',
                'title': 'The Origins of Intelligence in Children',
                'abstract': 'This work presents a developmental theory of cognitive development in children, proposing that intelligence emerges from the interaction between biological maturation and environmental experience. The theory describes four major stages of cognitive development: sensorimotor, preoperational, concrete operational, and formal operational. Each stage is characterized by distinct ways of thinking and problem-solving abilities. Cognitive development occurs through processes of assimilation (incorporating new experiences into existing schemas) and accommodation (modifying schemas to fit new experiences). Children construct knowledge through active exploration and problem-solving.',
                'authors': ['Jean Piaget'],
                'subfield': 'developmental_psychology',
                'published': '1952',
                'url': '',
                'key_concepts': ['cognitive development', 'schemas', 'assimilation', 'accommodation', 'stages']
            },
            {
                'source': 'sample',
                'id': 'beck_1967',
                'title': 'Depression: Clinical, Experimental, and Theoretical Aspects',
                'abstract': 'This comprehensive work presents a cognitive theory of depression, proposing that depressive disorders result from systematic negative biases in information processing. The theory identifies three main cognitive patterns in depression: negative views of oneself, the world, and the future (the cognitive triad). Depression is maintained by automatic negative thoughts, cognitive distortions, and dysfunctional underlying beliefs. The cognitive model provides the foundation for cognitive therapy, which helps patients identify and challenge negative thought patterns and develop more balanced thinking styles.',
                'authors': ['Aaron T. Beck'],
                'subfield': 'clinical_psychology',
                'published': '1967',
                'url': '',
                'key_concepts': ['cognitive therapy', 'cognitive triad', 'negative thinking', 'information processing']
            },
            {
                'source': 'sample',
                'id': 'mischel_1968',
                'title': 'Personality and Assessment',
                'abstract': 'This work challenges traditional trait theories of personality, arguing that behavior is more strongly influenced by situational factors than by stable personality traits. The author proposes a cognitive-social learning approach that emphasizes the interaction between person and situation. The book introduces the concept of cognitive-affective personality systems, suggesting that behavior results from the interaction of cognitive units (encoding, expectations, affects) with situational variables. This perspective led to the development of modern personality theories that integrate traits and situational influences.',
                'authors': ['Walter Mischel'],
                'subfield': 'personality_psychology',
                'published': '1968',
                'url': '',
                'key_concepts': ['person-situation debate', 'cognitive-affective systems', 'situational factors', 'social learning']
            }
        ]

        print(f"Created {len(classic_papers)} classic paper samples")
        return classic_papers

    def collect_psychology_textbook_excerpts(self) -> List[Dict]:
        """Collect textbook excerpt samples for psychology topics."""

        textbook_excerpts = [
            {
                'source': 'textbook',
                'id': 'intro_psych_ch1',
                'title': 'Introduction to Psychology: Science of Mind and Behavior',
                'content': 'Psychology is the scientific study of mind and behavior. It encompasses the biological influences, social pressures, and environmental factors that affect how people think, act, and feel. Gaining a richer understanding of psychology can help people achieve insights into their own actions as well as a better understanding of other people. Modern psychology is characterized by its scientific approach to understanding human behavior and mental processes. Psychologists use empirical methods to investigate questions about human nature, including experiments, correlational studies, case studies, and naturalistic observations.',
                'authors': ['Multiple Authors'],
                'subfield': 'introduction',
                'published': '2020',
                'textbook': 'Introduction to Psychology',
                'chapter': 'Chapter 1',
                'key_concepts': ['psychology', 'scientific method', 'mind', 'behavior', 'empirical methods']
            },
            {
                'source': 'textbook',
                'id': 'cognitive_psych_attention',
                'title': 'Attention and Information Processing',
                'content': 'Attention refers to the limited capacity to process information from multiple sources simultaneously. Selective attention allows us to focus on relevant stimuli while ignoring irrelevant distractions. The cocktail party phenomenon demonstrates our ability to attend to one conversation while ignoring others. Divided attention tasks show that performance decreases when trying to attend to multiple stimuli. Attention can be automatic (involuntary) or controlled (voluntary). Modern theories of attention include bottleneck models, capacity models, and feature integration theory. Working memory plays a crucial role in attention processes by temporarily storing and manipulating information.',
                'authors': ['Multiple Authors'],
                'subfield': 'cognitive_psychology',
                'published': '2019',
                'textbook': 'Cognitive Psychology: Mind and Brain',
                'chapter': 'Chapter 4',
                'key_concepts': ['attention', 'selective attention', 'working memory', 'cocktail party phenomenon', 'information processing']
            },
            {
                'source': 'textbook',
                'id': 'social_psych_attitudes',
                'title': 'Attitudes and Attitude Change',
                'content': 'Attitudes are evaluations of people, objects, or ideas. They have three components: affective (emotional), behavioral (action tendencies), and cognitive (beliefs). Attitudes can be explicit (conscious) or implicit (unconscious). Strong attitudes are more accessible from memory and more resistant to change. Attitude change can occur through persuasion, cognitive dissonance, or elaboration likelihood. Central route processing involves careful consideration of message arguments, while peripheral route processing relies on superficial cues. Social influence, cognitive consistency, and motivated reasoning all play roles in attitude formation and change.',
                'authors': ['Multiple Authors'],
                'subfield': 'social_psychology',
                'published': '2021',
                'textbook': 'Social Psychology',
                'chapter': 'Chapter 7',
                'key_concepts': ['attitudes', 'persuasion', 'cognitive dissonance', 'elaboration likelihood', 'social influence']
            }
        ]

        print(f"Created {len(textbook_excerpts)} textbook excerpt samples")
        return textbook_excerpts

    def save_data(self, data: List[Dict], filename: str):
        """Save collected data to JSON and CSV files."""
        filepath = self.output_dir / "abstracts" / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Data saved to: {filepath}")

        # also save a CSV version for easier inspection
        csv_filepath = self.output_dir / "metadata" / filename.replace('.json', '.csv')

        if data:
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['source', 'title', 'authors', 'subfield', 'published']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for item in data:
                    row = {field: item.get(field, '') for field in fieldnames}
                    if 'authors' in item and isinstance(item['authors'], list):
                        row['authors'] = ', '.join(item['authors'])
                    writer.writerow(row)

            print(f"Metadata saved to: {csv_filepath}")

    def create_annotation_guidelines(self):
        """Create annotation guidelines and save to JSON."""
        guidelines = {
            "purpose": "心理学术语提取标注指南",
            "term_definition": "心理学术语是指具有特定心理学含义的专业词汇或短语",
            "inclusion_criteria": [
                "心理学理论概念 (如：cognitive dissonance, operant conditioning)",
                "心理学构念 (如：self-efficacy, attachment style)",
                "实验方法和工具 (如：reaction time, Stroop test)",
                "心理现象和过程 (如：memory consolidation, attention)",
                "心理障碍和症状 (如：depression, anxiety disorder)",
                "治疗技术和干预 (如：cognitive behavioral therapy, exposure therapy)"
            ],
            "exclusion_criteria": [
                "通用学术词汇 (如：study, research, analysis)",
                "常规英语词汇 (如：people, think, feel)",
                "统计术语 (如：significant, correlation) - 除非是心理学特定的统计方法",
                "生物医学术语 (如：neuron, brain) - 除非直接涉及心理学功能"
            ],
            "multiword_terms": "多词术语应作为整体标注，不应分割",
            "boundary_guidelines": "术语边界应基于语言学和心理学专业标准",
            "examples": [
                {
                    "text": "The study examined cognitive dissonance reduction strategies.",
                    "terms": ["cognitive dissonance"],
                    "explanation": "cognitive dissonance是一个心理学术语，reduction strategies不是"
                },
                {
                    "text": "Participants completed the Beck Depression Inventory.",
                    "terms": ["Beck Depression Inventory"],
                    "explanation": "这是一个心理评估工具的完整名称"
                }
            ]
        }

        guidelines_path = self.output_dir / "annotation_guidelines.json"
        with open(guidelines_path, 'w', encoding='utf-8') as f:
            json.dump(guidelines, f, indent=2, ensure_ascii=False)

        print(f"Annotation guidelines saved to: {guidelines_path}")

    def collect_all_data(self):
        """Collect all data types (arXiv, classic samples, textbook excerpts)."""
        print("Starting collection of psychology data...")

        all_data = []

        # 1. Collect from ArXiv
        arxiv_papers = self.collect_from_arxiv()
        all_data.extend(arxiv_papers)

        # 2. Add classic paper samples
        classic_papers = self.collect_sample_classic_papers()
        all_data.extend(classic_papers)

        # 3. Add textbook excerpt samples
        textbook_excerpts = self.collect_psychology_textbook_excerpts()
        all_data.extend(textbook_excerpts)

        # save combined data
        self.save_data(all_data, "psychology_data_collection.json")

        # save separate files by source
        if arxiv_papers:
            self.save_data(arxiv_papers, "arxiv_papers.json")
        if classic_papers:
            self.save_data(classic_papers, "classic_papers.json")
        if textbook_excerpts:
            self.save_data(textbook_excerpts, "textbook_excerpts.json")

        # create annotation guidelines
        self.create_annotation_guidelines()

        print(f"\nData collection completed.")
        print(f"Total documents: {len(all_data)}")
        print(f"- ArXiv papers: {len(arxiv_papers)}")
        print(f"- Classic papers: {len(classic_papers)}")
        print(f"- Textbook excerpts: {len(textbook_excerpts)}")

        return all_data

if __name__ == "__main__":
    # Install dependencies
    # pip install requests feedparser

    collector = PsychologyAbstractCollector()
    data = collector.collect_all_data()