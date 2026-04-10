import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_NAME = "PieroAguinaga/tinyllama-sql-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_NAME)
base  = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_NAME)
model.eval()
model.generation_config.max_length = None

TEMPLATE = """\
Given the following SQL schema and a question, write the correct SQL query.

### Schema:
{context}

### Question:
{question}

### SQL Query:
"""

MAX_SEQ_LENGTH   = 512 
def generate_sql(schema: str, question: str) -> str:
    prompt = TEMPLATE.format(context=schema.strip(), question=question.strip())
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        model.generation_config.max_length = None
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).split('\n')[0].strip()

demo = gr.Interface(
    fn=generate_sql,
    inputs=[
        gr.Textbox(label="Schema (CREATE TABLE statements)", lines=5,
                   placeholder="CREATE TABLE singer (Singer_ID int, Name text, Age int)"),
        gr.Textbox(label="Question", lines=2,
                   placeholder="How many singers do we have?"),
    ],
    outputs=gr.Textbox(label="Generated SQL"),
    title="TinyLlama Text-to-SQL — QLoRA Fine-Tuned",
    description="Fine-tuned on 1,000 samples from `b-mc2/sql-create-context` using QLoRA on a free Colab T4.",
    examples=[
        [
            "CREATE TABLE singer (Singer_ID int, Name text, Country text, Age int)",
            "How many singers do we have?"
        ],
        [
            "CREATE TABLE table_name_64 (date VARCHAR, home_team VARCHAR, away_team VARCHAR)",
            "What day did Collingwood play as the home team?"
        ],
    ],
)

demo.launch()