# Stress Therapy Bot

This project implements a stress therapy bot using three different approaches, each designed to provide supportive conversations and guidance for individuals dealing with stress and mental health concerns.

## Table of Contents

1. [Version 1: Self_Help_Bot_v1.py](#version-1-self_help_bot_v1py)
2. [Version 2: Self_Help_Bot_v2.py](#version-2-self_help_bot_v2py)
3. [Version 3: Self_Help_Bot_v3.py](#version-3-self_help_bot_v3py)
4. [Key Features](#key-features)
5. [Installation and Dependencies](#installation-and-dependencies)
6. [Usage](#usage)
7. [Ethical Considerations](#ethical-considerations)
8. [Future Improvements](#future-improvements)

## Version 1: Self_Help_Bot_v1.py

This version uses traditional NLP techniques and machine learning models. The bot can be fine tuned on any help self book that is provided to it.

### Technical Details:
- PyTorch for deep learning
- PEFT (Parameter-Efficient Fine-Tuning)
- Sentence Transformers for text embeddings
- LangGraph for conversation flow
- Hugging Face Transformers for language models

### Key Components:
- PDF text extraction
- QA pair generation
- Fine-tuning of Meta-Llama-3.1-8B
- Multiple "agent" models for different therapy aspects
- Cosine similarity for information retrieval

## Version 2: Self_Help_Bot_v2.py

This version uses the DSPy framework for a more dynamic approach.

### Technical Details:
- DSPy framework
- Role-playing with multiple therapist characters
- GPT-4 for response generation and evaluation

### Key Components:
- TherapistTrait and CharacterProfile classes
- EnhancedRolePlayingBot for conversation handling
- PatientSimulator for realistic responses
- TherapistEvaluator for performance assessment

## Version 3: Self_Help_Bot_v3.py

This version incorporates fine-tuning of a custom language model.

### Technical Details:
- Hugging Face Transformers for model fine-tuning
- DSPy framework integration
- Pre-trained GPT-2 model as base

### Key Components:
- DatasetLoader for managing training data
- TherapistModelFineTuner for custom model fine-tuning
- FineTunedTherapistBot using the custom model

## Key Features

- Multiple therapist personas with distinct traits
- Adaptive conversation flow
- Integration of pre-trained and fine-tuned models
- Performance evaluation and trait improvement
- Realistic patient simulation

## Installation and Dependencies

Install the required libraries:
```!pip install dspy torch transformers datasets langgraph peft sentence-transformers spacy nltk PyPDF2 sklearn```

Set up API keys for OpenAI and Hugging Face in your environment variables.

## Usage

To use a specific version:

1. Ensure all dependencies are installed
2. Set up the required API keys
3. Run the desired Python file, e.g., `python3 Self_Help_Bot_v3.py`

## Ethical Considerations

- Not a substitute for professional mental health care
- Ensure user privacy and data protection
- Be aware of potential biases in models and data
- Include clear disclaimers about bot capabilities and limitations

## Future Improvements

- Implement advanced natural language understanding
- Incorporate multi-modal inputs (voice, images)
- Develop sophisticated performance metrics
- Enhance crisis situation handling
- Implement user feedback mechanisms
