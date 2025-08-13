from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset, ClassLabel
import transformers
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("Transformers imported from:", transformers.__file__)
print("Transformers version:", transformers.__version__)

# Load dataset
dataset = load_dataset("csv", data_files="startup_pitches.csv")
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

# Check columns for debug
print("Columns:", dataset['train'].column_names)
print("Sample data:", dataset['train'][0])

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Map string labels to integers - Fixed the label mapping to match our dataset
label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

def encode_labels(example):
    # Handle the exact case from our dataset (Positive, Negative, Neutral)
    example['label'] = label_mapping[example['label'].strip()]
    return example

dataset = dataset.map(encode_labels)

# Tokenization function
def tokenize(batch):
    return tokenizer(batch['pitch_text'], padding=True, truncation=True, max_length=512)

# Apply tokenization
dataset = dataset.map(tokenize, batched=True)

# Set format for PyTorch
dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])

# Load model with correct number of labels and mappings
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=3,
    id2label={0: 'Negative', 1: 'Neutral', 2: 'Positive'},
    label2id={'Negative': 0, 'Neutral': 1, 'Positive': 2}
)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./startup_sentiment_bert",
    eval_strategy="epoch",  # Fixed parameter name
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,  # Reduced from 20 to 3 for faster training
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_strategy="epoch",
    report_to=None,  # Disable wandb if not needed
    seed=42
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,  # Added metrics computation
)

# Start training
print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save final model and tokenizer
print("Saving model and tokenizer...")
model.save_pretrained("./startup_sentiment_bert")
tokenizer.save_pretrained("./startup_sentiment_bert")

print("Training completed successfully!")

# Test the model with a sample prediction
def test_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
    label_names = ['Negative', 'Neutral', 'Positive']
    return label_names[predicted_class], confidence