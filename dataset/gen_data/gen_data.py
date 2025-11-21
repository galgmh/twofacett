import os
import json
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

from config import CONFIG
# ====== API Setup ======
client = OpenAI(
    api_key=CONFIG.OPENAI_API_KEY,
    base_url=CONFIG.OPENAI_BASE_URL
)

# ====== Load WritingPrompts dataset ======
def load_writingprompts(n=1000, split="train"):
    """ 从 Hugging Face 加载 WritingPrompts 数据集 每条数据包含: prompt, story """
    ds = load_dataset(
        "euclaise/writingprompts",
        split=split,
        token=CONFIG.HUGGINGFACE_API_KEY
    )
    if n and n < len(ds):
        ds = ds.select(range(n))
    return ds

# ====== Generate AI-modified human text ======
def gen_ai_human_modified(text, model="gpt-3.5-turbo"):
    prompt = (
        "Please polish the following text while keeping the main meaning unchanged. "
        "Make it more fluent, clear, and logically consistent. "
        "You may slightly expand it, but do not completely rewrite it.\n\n"
        f"Original text:\n{text}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful text polishing assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        content = resp.choices[0].message.content
        return content.strip()

    except Exception as e:
        print("⚠️ Editing error:", e)
        return None


# ====== Generate AI-written story ======
def gen_ai_story(prompt_text, model="gpt-3.5-turbo"):
    prompt = (
        "You are a creative writer. Please write a short story based on the following writing prompt. "
        "Make sure the story is coherent and creative.\n\n"
        f"Prompt:\n{prompt_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and creative story-writing assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
        content = resp.choices[0].message.content
        return content.strip()

    except Exception as e:
        print("⚠️ AI story generation error:", e)
        return None


# ====== Build Dataset ======
def build_dataset_by_WP(output_path="dataset_wp.jsonl", n=1000, split="train"):
    """ 构建数据集:
    prompt: 提示词 
    text: 文本
    label: AI改写等级(human, ai_m, ai_c)
    model: 生成用的大模型"""
    wp = load_writingprompts(n=n, split=split)
    model = "gpt-4"
    dataset = []

    for sample in tqdm(wp, desc="Processing WritingPrompts"):
        prompt = sample.get("prompt", "")
        story = sample.get("story", "")

        if story:
            dataset.append({"prompt": prompt, "text": story, "label": "human", "model": "WP"})
            
            modified = gen_ai_human_modified(story, model=model)
            dataset.append({"prompt": prompt, "text": modified, "label": "ai_m", "model": model})

            ai_story = gen_ai_story(prompt)
            dataset.append({"prompt": prompt, "text": ai_story, "label": "ai_c", "model": model})

    # ===== Save JSONL =====
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved dataset to {output_path}, total {len(dataset)} samples")


# ====== Run ======
build_dataset_by_WP(output_path="data/dataset_wp.jsonl", n=1000, split="train")
