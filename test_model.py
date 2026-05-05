from transformers import AutoTokenizer, AutoModel

model_name = "law-ai/InLegalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Model loaded successfully!")
