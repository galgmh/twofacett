import json
import os
from openai import OpenAI

from config import CONFIG

client = OpenAI(
    api_key=CONFIG.OPENAI_API_KEY,
    base_url=CONFIG.OPENAI_BASE_URL
)

INPUT_DIR = "dataset/raw_data"
FILE_NAMES = ["arxiv.json", "writing_prompt.json", "xsum.json", "yelp_review.json"]
HUMAN_TEXT = ["abstract", "story", "document", "content"]
OUTPUT_PATH = "dataset/benchmark_data/iterate.json"

# Number of rewriting iterations
NUM_ITER = 16

# OpenAI model to use
MODEL = "gpt-4.1-mini"

# system prompt for iterative rewriting
SYSTEM_PROMPT = [
    """
    You rewrite academic abstracts. Preserve all scientific facts. Improve clarity, precision, coherence, and academic tone. Do not add or remove information or mention rewriting.
    """,
    """
    You rewrite fiction. Keep plot, characters, events, and setting exactly the same. You may alter style, tone, pacing, and sentence structure. Do not add or remove story content or mention rewriting.
    """,
    """
    You rewrite news text. Preserve all facts, names, dates, numbers, and events exactly. Adjust structure or wording, but keep a neutral journalistic tone. Do not add commentary or meta text.
    """,
    """
    You rewrite user reviews. Keep all opinions, emotions, and experiences unchanged. You may modify tone, structure, pacing, and wording. Do not add details or summarize or mention rewriting.
    """,
    
    ]

# rewriting prompt
REWRITE_PROMPT = [
    [
    # Structural Reorganization
    """
    Rewrite by reorganizing sentence structure for clearer logical flow. Keep all scientific content.

    Text:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite with more precise academic vocabulary and smoother transitions. Preserve technical meaning.

    Text:
    {}
    """,
    # Conceptual Clarification
    """
    Rewrite to clarify conceptual relationships and improve explanation quality. Keep facts unchanged.

    Text:
    {}
    """,
    # Formal Academic Enhancement 
    """
    Rewrite to be more formal, concise, and academically polished. Do not add information.

    Text:
    {}
    """
    ],
    [
    # Structural Reorganization
    """
    Rewrite by restructuring narrative flow and adjusting transitions without changing plot or events.

    Text:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite using varied vocabulary and enhanced descriptive language. Keep all events and actions.

    Text:
    {}
    """,
    # Tone Variation
    """
    Rewrite with a different tone or narrative voice, without altering story events.

    Text:
    {}
    """,
    # Pacing Adjustment
    """
    Rewrite focusing on pacing and clarity; adjust rhythm and detail density while keeping events intact.

    Text:
    {}
    """
    ],
    [
    # Structural Reorganization
    """
    Rewrite by reorganizing sentence structure while preserving all facts, names, and dates.

    Text:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite using alternate journalistic phrasing while keeping all factual details identical.

    Text:
    {}
    """,
    # Concision Improvement
    """
    Rewrite to be clearer and more concise, maintaining neutral news style and all factual elements.

    Text:
    {}
    """,
    # Tone Variation
    """
    Rewrite with a slightly different journalistic voice while preserving all facts.

    Text:
    {}
    """
    ],
    [
    # Structural Reorganization 
    """
    Rewrite by restructuring sentences and reorganizing flow, preserving all opinions and experiences.

    Review:
    {}
    """,
    # Tone Variation
    """
    Rewrite using a different tone or voice while keeping meaning and sentiment unchanged.

    Review:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite using varied vocabulary while maintaining the same opinions and emotional content.

    Review:
    {}
    """,
    # Pacing Adjustment
    """
    Rewrite to improve pacing and clarity without altering core complaints or experiences.

    Review:
    {}
    """,
    ]
]

def rewrite_iterative(text, system_prompt, user_prompts, rounds=16, model="gpt-4o-mini"):
    """Perform N rounds of recursive rewriting."""
    
    current_text = text
    results = []
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    for r in range(1, rounds + 1):
        
        user_prompt = user_prompts[(r - 1) % len(user_prompts)].format(current_text)
        messages.append({"role": "user", "content": user_prompt})

        if (r < rounds // 2):
            temperature = 0.5   # low temperature for early rounds
        else:
            temperature = 0.7   # higher temperature for later rounds
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        current_text = response.choices[0].message.content.strip()

        if(r % 4 == 0):
            results.append(current_text)

    return results

def main():

    id = 1
    output = []
    for i in range(len(FILE_NAMES)):
        # Load original dataset
        input_path = os.path.join(INPUT_DIR, FILE_NAMES[i])
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            print(f"Processing ID: {id}")
            original = item[HUMAN_TEXT[i]]
            rewritten = rewrite_iterative(original, SYSTEM_PROMPT[i], REWRITE_PROMPT[i], NUM_ITER, MODEL)

            output.append({
                "id": id,
                "domain": item["domain"],
                "human_text": original,
                "llm_text_1": rewritten[0],
                "llm_text_2": rewritten[1],
                "llm_text_3": rewritten[2],
                "llm_text_4": rewritten[3],
                "llm_type": MODEL,
            })
            id += 1

    # Save processed results
    os.makedirs("dataset/benchmark_data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"data save to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
