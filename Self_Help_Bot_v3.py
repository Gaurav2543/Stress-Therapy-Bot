# !pip install dspy

import os
import re
import dspy
import json
import random
import numpy as np
from google.colab import userdata
from datasets import load_dataset
from typing import List, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

os.environ["OPENAI_API_KEY"] = userdata.get('OpenAIKey')
os.environ["HF_TOKEN"] = userdata.get('HuggingFaceToken')

class TherapistTrait:
    def __init__(self, name: str, definition: str, contexts: List[str], examples: List[str]):
        self.name = name
        self.definition = definition
        self.contexts = contexts
        self.examples = examples

class CharacterProfile:
    def __init__(self, name: str, background: str, personality_traits: List[str], communication_style: str, specialization: List[str]):
        self.name = name
        self.background = background
        self.personality_traits = personality_traits
        self.communication_style = communication_style
        self.specialization = specialization

class TherapistDictionary:
    def __init__(self):
        self.traits: Dict[str, TherapistTrait] = {}

    def add_trait(self, trait: TherapistTrait):
        self.traits[trait.name] = trait

    def get_trait(self, name: str) -> TherapistTrait:
        return self.traits.get(name)

    def update_trait(self, name: str, new_definition: str = None, new_contexts: List[str] = None, new_examples: List[str] = None):
        trait = self.traits.get(name)
        if trait:
            if new_definition:
                trait.definition = new_definition
            if new_contexts:
                trait.contexts.extend(new_contexts)
            if new_examples:
                trait.examples.extend(new_examples)


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)
        self.train_data = self.dataset['train']
        self.formatted_examples = self._format_all_examples()

    def _format_all_examples(self) -> str:
        formatted = "Here are examples of mental health counseling conversations:\n\n"
        for example in self.train_data:
            formatted += f"Patient: {example['Context']}\nTherapist: {example['Response']}\n\n"
        return formatted

class TherapistModelFineTuner:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_dataset(self, dataset):
        def tokenize_function(examples):
            texts = [f"Patient: {context}\nTherapist: {response}" for context, response in zip(examples["Context"], examples["Response"])]
            return self.tokenizer(texts, truncation=True, padding="max_length", max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        return tokenized_datasets

    def fine_tune(self, dataset, output_dir: str = "./fine_tuned_therapist_model", num_train_epochs: int = 3):
        tokenized_datasets = self.prepare_dataset(dataset)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"] if "test" in tokenized_datasets else None,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )

        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Fine-tuning complete. Model saved to {output_dir}")

class FineTunedTherapistBot(dspy.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.generate_response = self._generate_response
        self.characters: List[CharacterProfile] = self._initialize_characters()
        self.current_character: CharacterProfile = None
        self.conversation_history: List[str] = []
        self.exchange_counter: int = 0
        self.threshold: int = random.randint(3, 5)
        self.therapist_dictionary: TherapistDictionary = self._initialize_therapist_dictionary()
        self.progress_scores: List[float] = []
        self.dataset_loader: DatasetLoader = DatasetLoader("Amod/mental_health_counseling_conversations")

    def _initialize_characters(self) -> List[CharacterProfile]:
        return [
            CharacterProfile(
                name="Coach Mike Johnson",
                background="Former athlete turned life coach, specializes in motivation and goal-setting",
                personality_traits=["energetic", "direct", "optimistic"],
                communication_style="Uses sports analogies, asks challenging questions, but not intrusive or hurtful",
                specialization = ["Motivation", "Goal-setting"]
            ),
            CharacterProfile(
                name="Dr. Emily Chen",
                background="Experienced therapist with a focus on work-related and financial stress",
                personality_traits=["empathetic", "practical", "insightful"],
                communication_style="Warm and encouraging, uses real-world examples to illustrate coping strategies",
                specialization=["Work-related Stressors", "Financial Stressors"]
            ),
            CharacterProfile(
                name="Dr. Michael Rodriguez",
                background="Clinical psychologist specializing in emotional and psychological stress",
                personality_traits=["patient", "analytical", "supportive"],
                communication_style="Calm and methodical, often uses cognitive-behavioral techniques in explanations",
                specialization=["Emotional Stressors", "Psychological Stressors"]
            ),
            CharacterProfile(
                name="Dr. Sarah Johnson",
                background="Trauma-informed therapist with expertise in PTSD and acute stress disorders",
                personality_traits=["compassionate", "gentle", "reassuring"],
                communication_style="Uses a lot of validation and normalization, emphasizes safety and trust",
                specialization=["Traumatic Stressors", "Social Stressors"]
            ),
            CharacterProfile(
                name="Dr. David Lee",
                background="Holistic health practitioner focusing on physical and lifestyle-related stress",
                personality_traits=["energetic", "optimistic", "motivational"],
                communication_style="Enthusiastic about mind-body connections, often suggests practical lifestyle changes",
                specialization=["Physical Stressors", "Lifestyle Stressors"]
            ),
            CharacterProfile(
                name="Dr. Lisa Patel",
                background="Educational psychologist specializing in academic and technology-related stress",
                personality_traits=["understanding", "tech-savvy", "solution-oriented"],
                communication_style="Relates well to students and professionals, offers concrete strategies for managing digital overwhelm",
                specialization=["Academic Stressors", "Technology-related Stressors"]
            )
        ]

    def _initialize_therapist_dictionary(self) -> TherapistDictionary:
        dictionary = TherapistDictionary()
        dictionary.add_trait(TherapistTrait(
            name="Empathy",
            definition="The ability to understand and share the feelings of another",
            contexts=["Emotional distress", "Physical pain", "Life challenges"],
            examples=[
                "I can understand why you'd feel that way. It sounds like a really challenging situation.",
                "That must be incredibly difficult to deal with. I'm here to listen and support you."
            ]
        ))
        dictionary.add_trait(TherapistTrait(
            name="Non-judgmental",
            definition="Avoiding making judgments about a person's thoughts, feelings, or behaviors",
            contexts=["Confessions", "Mistakes", "Life choices"],
            examples=[
                "Thank you for sharing that with me. I'm here to understand and support you, not to judge.",
                "Everyone faces challenges in life. Let's focus on understanding your experiences and finding a way forward."
            ]
        ))
        return dictionary

    def _generate_response(self, context: str, character_profile: str, conversation_history: str, user_input: str, dataset_examples: str, therapist_dictionary: str) -> str:
        prompt = f"{context}\n\nCharacter Profile:\n{character_profile}\n\nConversation History:\n{conversation_history}\n\nPatient: {user_input}\nTherapist:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=input_ids.shape[1] + 100, num_return_sequences=1, no_repeat_ngram_size=2)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the therapist's response
        therapist_response = response.split("Therapist:")[-1].strip()

        # For simplicity, we're returning placeholder values for internal_state, next_action, and trait_evaluations
        return therapist_response, "Internal State", "Next Action", "{}"

    def _parse_trait_evaluations(self, trait_evaluations_str: str) -> Dict[str, float]:
        try:
            if isinstance(trait_evaluations_str, list):
                trait_evaluations_str = " ".join(trait_evaluations_str)
            return json.loads(trait_evaluations_str)
        except json.JSONDecodeError:
            trait_dict = {}
            pattern = r'(\w+):\s*([\d.]+)'
            matches = re.findall(pattern, trait_evaluations_str)
            for trait, score in matches:
                try:
                    trait_dict[trait] = float(score)
                except ValueError:
                    trait_dict[trait] = 0.0
            return trait_dict

    def forward(self, context: str, user_input: str) -> tuple:
        self.exchange_counter += 1
        if self.exchange_counter >= self.threshold and self.current_character is None:
            self.choose_character_based_on_input(user_input)

        if self.current_character is None:
            self.current_character = random.choice(self.characters)

        character_info = self._format_character_info(self.current_character)
        history = "\n".join(self.conversation_history[-5:])

        # Convert therapist_dictionary to a string representation
        therapist_dict_str = json.dumps({name: trait.__dict__ for name, trait in self.therapist_dictionary.traits.items()})

        bot_response, internal_state, next_action, trait_evaluations = self.generate_response(
            context=context,
            character_profile=character_info,
            conversation_history=history,
            user_input=user_input,
            dataset_examples=self.dataset_loader.formatted_examples,
            therapist_dictionary=therapist_dict_str
        )

        self._update_conversation_history(user_input, bot_response)
        self._update_therapist_dictionary(trait_evaluations)

        return bot_response, internal_state, next_action

    def _update_therapist_dictionary(self, trait_evaluations: str):
        # Parse the trait_evaluations string into a dictionary
        try:
            trait_evaluations_dict = json.loads(trait_evaluations)
        except json.JSONDecodeError:
            print("Error parsing trait evaluations. Skipping dictionary update.")
            return

        for trait, score in trait_evaluations_dict.items():
            if score < 0.7:
                trait_obj = self.therapist_dictionary.get_trait(trait)
                if trait_obj:
                    new_definition = f"Improved {trait_obj.definition}. Focus on increasing score above 0.7."
                    trait_obj.definition = new_definition
                    new_example = f"Example for improving {trait}: [Insert specific example based on recent conversation]"
                    trait_obj.examples.append(new_example)
                    new_context = f"Situations where {trait} score is below 0.7"
                    trait_obj.contexts.append(new_context)

        avg_score = sum(trait_evaluations_dict.values()) / len(trait_evaluations_dict)
        self.progress_scores.append(avg_score)

    def get_progress_report(self) -> str:
        if not self.progress_scores:
            return "No progress scores available."
        initial_score = self.progress_scores[0]
        current_score = self.progress_scores[-1]
        overall_change = current_score - initial_score
        report = f"Initial average score: {initial_score:.2f}\n"
        report += f"Current average score: {current_score:.2f}\n"
        report += f"Overall change: {overall_change:.2f}\n"
        if overall_change > 0:
            report += "The therapist is showing improvement."
        elif overall_change < 0:
            report += "The therapist's performance has declined."
        else:
            report += "The therapist's performance has remained stable."
        return report

    def _format_character_info(self, character: CharacterProfile) -> str:
        return (
            f"Name: {character.name}\n"
            f"Background: {character.background}\n"
            f"Personality: {', '.join(character.personality_traits)}\n"
            f"Communication Style: {character.communication_style}\n"
            f"Specialization: {', '.join(character.specialization)}"
        )

    def _update_conversation_history(self, user_input: str, bot_response: str):
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"{self.current_character.name}: {bot_response}")

    def choose_character_based_on_input(self, user_input: str):
        # Implementation similar to your original code, but more sophisticated
        keywords = {
            "work": ["Dr. Emily Chen"],
            "emotional": ["Dr. Michael Rodriguez", "Dr. Sarah Johnson"],
            "physical": ["Dr. David Lee"],
            "academic": ["Dr. Lisa Patel"],
            "financial": ["Dr. Emily Chen"],
            "exercise": ["Dr. David Lee"],
            "technology": ["Dr. Lisa Patel"],
            "student": ["Dr. Lisa Patel"],
            "trauma": ["Dr. Sarah Johnson"],
            "PTSD": ["Dr. Sarah Johnson"],
            "lifestyle": ["Dr. David Lee"],
            "stress": ["Dr. Michael Rodriguez", "Dr. Sarah Johnson", "Dr. David Lee", "Dr. Lisa Patel"],
            "job": ["Dr. Emily Chen"]
        }

        matched_characters = set()
        for keyword, characters in keywords.items():
            if keyword.lower() in user_input.lower():
                matched_characters.update(characters)

        if matched_characters:
            self.current_character = next((c for c in self.characters if c.name in matched_characters), None)
        else:
            self.current_character = random.choice(self.characters)

def fine_tune_model(dataset_name: str, output_dir: str):
    dataset = load_dataset(dataset_name)
    fine_tuner = TherapistModelFineTuner()
    fine_tuner.fine_tune(dataset, output_dir)

def run_conversation_with_fine_tuned_model(model_path: str, num_exchanges: int = 5, save_json: bool = False, json_filename: str = "conversation.json"):
    bot = FineTunedTherapistBot(model_path)
    patient = PatientSimulator()
    evaluator = TherapistEvaluator()
    context = """You are an AI role-playing as a supportive therapist specializing in stress management. You have been trained on a comprehensive dataset of mental health counseling conversations, which are provided to you. Use these examples to inform your responses, adapting the style and content to the current conversation.
    Engage in a natural, human-like conversation based on your character's profile and the provided examples. Show genuine interest in the user's feelings and experiences.
    Ask questions to understand the user's problems or concerns if they are unclear. Use your character's unique communication
    style and background to inform your responses. Offer support and guidance when appropriate, but avoid giving direct advice
    unless asked. Your goal is to help the user feel heard, understood, and supported while maintaining the authenticity of your
    character. Always try to solve the problem or concerns of the patient on your own before suggesting third party sources.
    Do not be intrusive or harmful in any way. Please ensure that all responses, including trait evaluations, are provided in valid JSON format."""

    conversation = Conversation()
    conversation_history = []
    print("Therapist: Hello! How are you feeling today?")

    for turn in range(num_exchanges):
        if turn == 0:
            patient_response, mood, challenge_level = patient.forward("", "")
        else:
            patient_response, mood, challenge_level = patient.forward("\n".join(conversation_history), bot_response)

        print(f"\033[1mPatient: {patient_response}\033[0m")
        print(f"[Mood: {mood}, Challenge Level: {challenge_level}]")

        bot_response, internal_state, next_action = bot.forward(context, patient_response)
        print(f"\033[1m{bot.current_character.name}: {bot_response}\033[0m")
        print(f"[Internal State: {internal_state}]")
        print(f"[Next Action: {next_action}]")

        # Convert therapist_dictionary to a string representation for the evaluator
        therapist_dict_str = json.dumps({name: trait.__dict__ for name, trait in bot.therapist_dictionary.traits.items()})
        trait_evaluations = evaluator.forward(bot_response, patient_response, therapist_dict_str)
        print("Trait Evaluations:")
        for trait, score in trait_evaluations.items():
            print(f"  {trait}: {score:.2f}")
        print()

        bot._update_therapist_dictionary(json.dumps(trait_evaluations))

        conversation.add_exchange(ConversationExchange(
            patient_response=patient_response,
            therapist_response=bot_response,
            mood=mood,
            challenge_level=challenge_level,
            trait_evaluations=trait_evaluations
        ))

        conversation_history.extend([f"Patient: {patient_response}", f"{bot.current_character.name}: {bot_response}"])

        print(f"Progress Report after exchange {turn + 1}:")
        print(bot.get_progress_report())
        print()

    if save_json:
        save_conversation_to_json(conversation, json_filename)
        print(f"Conversation saved to {json_filename}")

# Fine-tune the model (run this once)
fine_tune_model("Amod/mental_health_counseling_conversations", "./fine_tuned_therapist_model")

# Run conversation with the fine-tuned model
run_conversation_with_fine_tuned_model("./fine_tuned_therapist_model", num_exchanges=6, save_json=False, json_filename="therapy_session.json")

