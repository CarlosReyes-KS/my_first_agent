# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import requests # For interacting with Ollama API
from datetime import datetime


data = pd.read_csv('creditcard.csv')
df = data.copy()
print("Dataset loaded successfully.")

df.drop_duplicates(inplace=True)

# Scale 'Amount' feature as done in previous examples
amount = df['Amount'].values.reshape(-1, 1)
scaler_amount = StandardScaler()
df['Amount'] = scaler_amount.fit_transform(amount)

# Define X and y for the agent to work with
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and testing sets (even for simple rule, useful for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# --- Agent Implementation ---

class FraudDetectionAgent:
    def __init__(self, ollama_model_name="mistral:latest", ollama_api_base_url="http://localhost:11434/api/generate"):
        """
        Initializes the FraudDetectionAgent to interact with a local Ollama LLM.
        Args:
            ollama_model_name (str): The name of the model downloaded in Ollama (e.g., "mistral").
            ollama_api_base_url (str): The endpoint for Ollama's generate API.
        """
        self.ollama_model_name = ollama_model_name
        self.ollama_api_base_url = ollama_api_base_url
        print(f"FraudDetectionAgent initialized. Interacting with local Ollama model: {ollama_model_name}")

    def _get_llm_suggestion(self):
        """
        Interacts with the local Ollama LLM to get a simple fraud detection rule.
        """
        """
        You are a basic fraud detection agent. Your goal is to identify fraudulent transactions in a highly imbalanced dataset where fraud is very rare (0.17% of transactions).
        You have access to the following features: {', '.join(features)}.
        Suggest a single, simple rule to detect fraud based on one of these features and a numerical threshold.
        The rule should be in the format: "If [feature_name] is [comparison_operator] [threshold_value], then it's fraud."
        For example: "If Amount is > 100.0, then it's fraud."
        Or: "If V10 is < -5.0, then it's fraud."

        Think step-by-step about which feature might be most indicative of fraud and a reasonable threshold.
        Consider that 'V' features are often PCA-transformed features, and some (e.g., V17, V14, V12) are known to be highly correlated with fraud (often having very negative values for fraudulent transactions). 'Amount' is also a direct financial feature.

        Provide your suggestion as a JSON object with 'feature_name' (string) and 'threshold_value' (float), and 'comparison_operator' (string, either '>' or '<').
        Example output: {{"feature_name": "V14", "threshold_value": -5.0, "comparison_operator": "<"}}
        """
        prompt = f"""
        You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) for the user to execute.
1. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When you provide the code the code should be separeted by statements contained in a list with the necessary line breaks so that a human can read it.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Return your answer as JSON
Additional requirements:
1. Your answer needs to be formatted in the following way steps list including each step and a separate statements list where every new line of code is an item. Example: steps: ["Import", "Prepare data"], statements: ["import pandas as pd"].
Your task is to train a AutoML model using Pycaret's binary classification strategy that can detect fraudulent transactions when provided a dataset in CSV format already accesible the system that's already PCA-transformed with the name 'creditcard.csv'.
        """
        
        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "format": "json", # Request JSON output from Ollama
            "stream": False, # We want a single, complete response
            "options": {
                "temperature": 0.0 # Make the LLM's response deterministic
            }
        }

        try:
            response = requests.post(self.ollama_api_base_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result.get("response"):
                # Ollama's /api/generate puts the model's output in the 'response' field
                json_str = result["response"]
                #print(f"JSON from Ollama {json_str}")
                return json.loads(json_str)
            else:
                print("Ollama response structure unexpected:", result)
                return None
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to Ollama server at {self.ollama_api_base_url}.")
            print("Please ensure Ollama is installed and the '{self.ollama_model_name}' model is running.")
            print("You might need to run 'ollama run {self.ollama_model_name}' in your terminal.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None
        except json.JSONDecodeError:
            print("Ollama response was not valid JSON. Model might not have followed instructions.")
            print(f"Raw Ollama response: {result.get('response', 'N/A')}")
            return None

    def detect_fraud(self):
        """
        Agent's core function to generate code to solve a task using Python.
        """
        print("\nAgent is thinking and generating a solution using the local LLM...")
        
        # Get a rule from the local Ollama LLM
        rule_suggestion = self._get_llm_suggestion()
        if not rule_suggestion:
            print("Agent failed to get a valid solution from LLM. Using a fallback rule.")
            # Fallback rule if LLM fails or returns invalid JSON
            steps = []
            statements= []
            now = datetime.now()
        else:
            steps = rule_suggestion.get('steps')
            statements = rule_suggestion.get('statements')
            now = datetime.now()
            # Basic validation of the LLM's suggestion
            """@TODO: Missing proper validation for the code generation..."""
        print(f"Agent's suggested solution contains: '{steps} steps' with this amount of LoC {len(statements)}")
        return steps, statements, now

# --- Main Execution ---

# Initialize the agent to use the Mistral model with Ollama
agent = FraudDetectionAgent(ollama_model_name="mistral")

# Have the agent detect fraud on the test set
steps, statements, now = agent.detect_fraud()

# --- Evaluate Agent's Performance ---
#print("\n--- Agent's Rule Performance ---")
#accuracy = accuracy_score(y_test, y_pred_agent)
#precision = precision_score(y_test, y_pred_agent, zero_division=0)
#recall = recall_score(y_test, y_pred_agent, zero_division=0)
#f1 = f1_score(y_test, y_pred_agent, zero_division=0)

print(f"Solution: {steps}, LoC: {len(statements)} - Generated: {now} ")

with open(f"/home/kappa/python/my_first_agent/results/{now}.py", "w") as f:
    for s in statements:
        f.write(s + "\n")
#print(f"Accuracy: {accuracy:.4f}")
#print(f"Precision: {precision:.4f}")
#print(f"Recall: {recall:.4f}")
#print(f"F1 Score: {f1:.4f}")

# Display Confusion Matrix
#cm = confusion_matrix(y_test, y_pred_agent)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
#fig, ax = plt.subplots(figsize=(6, 6))
#disp.plot(cmap='Blues', ax=ax, values_format='d')
#ax.set_title(f"Agent Rule Confusion Matrix\nAccuracy: {accuracy:.4f}")
#plt.show()

print("\n--- Comparison Notes ---")
print("This agent-based approach uses a simple, LLM-suggested rule from a local Ollama instance.")
print("It is designed as a 'toy example' to illustrate the concept of an agent making decisions with a local LLM.")
print("Its performance is expected to be significantly lower than complex ML models (like LGBM, ANN, CNN, RNN, LSTM) or AutoML solutions,")
print("which learn complex patterns from data rather than relying on a single, human-interpretable rule.")
print("The value here is in the interpretability and the agent's 'reasoning' process, not raw performance.")


# %%
