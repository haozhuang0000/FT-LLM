# FT-LLM

#### Description:

Finetune llama3:8b in order to extract financial statistics (Table & Paragraph) from 10K report with Unsloth framework

- `conda create -n ENV_NAME python=3.11`
- `pip install -r requirements.txt`

#### Please check Unsloth github for up-to-date version: https://github.com/unslothai/unsloth

- `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
- `pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes`
- `pip install torch`

#### Set up .env file:

- `OPEN_AI_KEY=sk-xxxxxxx`
- `HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxx`
- `HF_DATA_PATH=HUGGINGFACE/PATH`
- `HF_MODEL_PATH=HUGGINGFACE/PATH`

#### Code running instruction:

- `GPT4_GenerateTemplate.py`
- `ftllama_extract_htmltable.py`
