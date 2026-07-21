import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from pathlib import Path

# --- 1. Define the Model Architecture ---
# This class must match the architecture used during training.
# You can copy this class from the original training script.
class SDGClassifier(nn.Module):
    def __init__(self, model_path, pooler_dropout, class_number):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
        # Checkpoint stores custom pooler AS bert.pooler (overwrites LUKE's built-in)
        # so assign it directly onto self.bert.pooler, not as self.pooler
        self.bert.pooler = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        )
        
        self.dropout = nn.Dropout(pooler_dropout)
        self.tanh = nn.Tanh()
        
        # cls.0.* → Sequential
        self.cls = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, class_number)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, position, labels=None):
        out = self.bert(
            input_ids,
            attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
        hidden = out.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).float()
        avg    = (hidden * mask).sum(1) / mask.sum(1)

        pooled = self.tanh(self.bert.pooler(self.dropout(avg)))
        logits = self.cls(pooled)
        return logits, avg, out.attentions

# --- 2. Setup and Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
BASE_MODEL = 'studio-ousia/luke-large-lite'
NUM_CLASSES = 17
DROPOUT_RATE = 0.26 # This is the optimized dropout rate from the paper's training

# Instantiate the model
model = SDGClassifier(model_path=BASE_MODEL, pooler_dropout=DROPOUT_RATE, class_number=NUM_CLASSES).to(device)
model.eval() # Set to evaluation mode

# Download the fine-tuned weights from this Hub
model_weights_path = hf_hub_download(
    repo_id="GE-Lab/SDGs-classifier",
    filename="best_model.pt"
)

# # Load the weights into the model
# model.load_state_dict(torch.load(model_weights_path, map_location=device))
state_dict   = torch.load(model_weights_path, map_location=device)
model.load_state_dict(state_dict, strict=True)

print("Model loaded successfully!")

# --- 3. Prepare Input ---
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# text = "Our research focuses on renewable energy solutions to combat climate change and ensure a sustainable future for all."

# inputs = tokenizer(
#     text,
#     None,
#     add_special_tokens=True,
#     max_length=512,
#     padding='max_length',
#     return_token_type_ids=True,
#     truncation=True,
#     return_tensors='pt'
# ).to(device)

# # The model's forward pass requires these additional dummy inputs
# inputs['position'] = torch.arange(0, inputs['input_ids'].shape[1]).unsqueeze(0).to(device)
# inputs['labels'] = torch.zeros(1, NUM_CLASSES).to(device) # Dummy labels for inference

# # --- 4. Get Predictions ---
# with torch.no_grad():
#     logits, _, _ = model(**inputs)
#     probabilities = torch.sigmoid(logits).cpu().numpy()[0]
#     predictions = (probabilities > 0.5).astype(int)

# # --- 5. Interpret the Results ---
# goal_contents = ['Goal 1: No Poverty','Goal 2: Zero Hunger','Goal 3: Good Health and Well-being','Goal 4: Quality Education','Goal 5: Gender Equality','Goal 6: Clean Water and Sanitation','Goal 7: Affordable and Clean Energy','Goal 8: Decent Work and Economic Growth','Goal 9: Industry, Innovation and Infrastructure','Goal 10: Reduced Inequalities','Goal 11: Sustainable Cities and Communities','Goal 12: Responsible Consumption and Production','Goal 13: Climate Action','Goal 14: Life Below Water','Goal 15: Life on Land','Goal 16: Peace, Justice and Strong Institutions','Goal 17: Partnerships for the Goals']

# print(f"\nText: '{text}'")
# print("\n--- Predicted SDGs (Threshold > 0.5) ---")
# predicted_goals = [goal_contents[i] for i, pred in enumerate(predictions) if pred == 1]
# if predicted_goals:
#     for goal in predicted_goals:
#         print(goal)
# else:
#     print("No SDGs detected with a probability > 0.5")

# print("\n--- All SDG Probabilities ---")
# for i, prob in enumerate(probabilities):
#     print(f"{goal_contents[i]:<55}: {prob:.2%}")
