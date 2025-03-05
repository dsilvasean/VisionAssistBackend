import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = "/home/sean/repos/project_repos/VisionAssistBackend"

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class IntentTrainer:
    def __init__(self, project_root, model_name="bert-base-uncased"):
        self.project_root = project_root
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.training_args = None

    def load_and_prepare_data(self, file_path):
        df = pd.read_csv(file_path)
        texts = df['input'].tolist()
        labels = self.label_encoder.fit_transform(df['intent'].tolist())
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        self.train_dataset = IntentDataset(X_train, y_train, self.tokenizer)
        self.test_dataset = IntentDataset(X_test, y_test, self.tokenizer)
    
    def setup_model(self):
        num_labels = len(self.label_encoder.classes_)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        self.model.to('cuda')
    
    def setup_training_args(self, output_dir, epochs=50, batch_size=16, warmup_steps=500, weight_decay=0.01):
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10
        )
    
    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset
        )
        trainer.train()
    
    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir, safe_serialization=True)
    
    def run(self):
        data_path = os.path.join(self.project_root, 'data/dataset1.csv')
        results_dir = os.path.join(self.project_root, 'data/results')
        model_dir = os.path.join(self.project_root, 'data/fine_tuned_model')

        # Load and prepare data
        self.load_and_prepare_data(data_path)

        # Setup model and training arguments
        self.setup_model()
        self.setup_training_args(output_dir=results_dir)

        # Train the model
        self.train()

        # Save the trained model
        self.save_model(model_dir)


class IntentClassifier:
    def __init__(self, model_dir, label_encoder_path):
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Load label encoder
        self.label_encoder = pd.read_pickle(label_encoder_path)
        # self.label_encoder = LabelEncoder()
        # self.label_encoder.classes_ = pd.read_pickle(label_encoder_path)

    def predict(self, text):
        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors='pt'
        )
        
        # Move input to the device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
        # Decode the class ID back to the original label
        predicted_label = self.label_encoder.inverse_transform([predicted_class_id])[0]
        return predicted_label


        
if __name__ == "__main__":
    # model_dir = os.path.join(PROJECT_ROOT, 'data/fine_tuned_model')
    # label_encoder_path = os.path.join(PROJECT_ROOT, 'data/label_encoder.pkl')
    # classifier = IntentClassifier(model_dir, label_encoder_path)
    # test_input = "what is the time right now"
    # predicted_intent = classifier.predict(test_input)
    # print(predicted_intent)

    trainer = IntentTrainer(PROJECT_ROOT)
    trainer.run()