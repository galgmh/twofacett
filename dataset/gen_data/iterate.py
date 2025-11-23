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
MODEL = "gpt-4o-mini"

# system prompt for iterative rewriting
SYSTEM_PROMPT = [
    """
    You are an expert academic writing assistant specializing in rewriting and refining research abstracts. 
    Your role is to produce text that is formal, coherent, precise, and stylistically consistent with academic writing standards. 
    You must preserve all scientific information and factual content while improving clarity and readability. 
    Do not add new claims, remove technical details, or mention that you are rewriting the text.
    """,
    """
    You are an advanced narrative rewriting assistant.  
    Your task is to rewrite fiction stories while preserving the original plot, characters, setting, and narrative events.  
    You may reorganize prose, vary sentence structure, alter pacing, and refine descriptions.  
    You must not add new story events, new characters, new lore, or new worldbuilding details.  
    You must not remove essential plot elements.  
    Maintain coherence, but feel free to shift tone, style, rhythm, or diction.  
    Each rewrite should feel like the story has been regenerated through a different AI model or writing tool.
    Avoid meta commentary, disclaimers, or talking about the rewriting process.
    """,
    """
    You are a professional news-language rewriting assistant.
    Your task is to rewrite news articles and summaries while preserving all factual information, dates, events, names, numbers, and entities exactly as stated.
    You may restructure sentences, change wording, and adjust phrasing, but you must not alter facts, introduce new information, or remove essential details.
    Maintain a neutral journalistic tone, similar to BBC/Reuters style.
    Do not provide commentary, opinions, or meta statements about the rewriting task.
    Your rewrites should sound like news articles written by different editors.
    """,
    """
    You are a rewriting assistant specializing in user-generated reviews.  
    Your goal is to rewrite the text while preserving all core opinions, complaints, experiences, and sentiments expressed by the reviewer.  
    You may change tone, sentence structure, pacing, and word choice, but you must not alter the meaning of the reviewer’s experience or add new details that were not present.  
    You must not remove any essential criticism or praise.  
    Avoid summarizing; produce a full rewritten version in natural language.  
    Do not reference the rewriting task or provide meta-commentary.
    """,
    
    ]

# rewriting prompt
REWRITE_PROMPT = [
    [
    # Structural Reorganization
    """
    Rewrite the text by reorganizing the sentence structure to improve logical flow. 
    Enhance clarity and precision while keeping all original scientific content intact. 
    Academic tone should remain formal and consistent.
 
    Text:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite the text by refining the vocabulary, using more precise academic wording, 
    and smoothing transitions between ideas. 
    Preserve all technical meaning and domain-specific terminology.

    Text:
    {}
    """,
    # Conceptual Clarification
    """
    Rewrite the text with a focus on clarifying the conceptual relationships and 
    improving the explanatory quality of the writing. 
    Keep all scientific facts unchanged and maintain an academic register.

    Text:
    {}
    """,
    # Formal Academic Enhancement 
    """
    Rewrite the following text to make it more formal, polished, and academical. 
    Ensure concise expression, improved sentence structure, and consistent scholarly tone. 
    Do not add new information.

    Text:
    {}
    """
    ],
    [
    # Structural Reorganization
    """
    Rewrite the following story by restructuring the narrative flow.  
    Modify sentence boundaries, reorder descriptive elements, and smooth or alter transitions between scenes.  
    Do not change the plot or introduce new events.

    Text:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite the story by varying the vocabulary and enhancing descriptive language.  
    Use different word choices, metaphors, or phrasing, while preserving all events and character actions.  
    Do not add new events or lore.

    Text:
    {}
    """,
    # Tone Variation
    """
    Rewrite the story with a shift in tone and narrative voice.  
    You may make the style more dramatic, more minimalistic, more lyrical, or more direct,  
    as long as the plot, events, and character actions remain unchanged.

    Text:
    {}
    """,
    # Pacing Adjustment
    """
    Rewrite the story with an emphasis on clarity and pacing.  
    Adjust rhythm, redistribute detail density, shorten or lengthen sentences,  
    and smooth narrative beats without altering the core events.

    Text:
    {}
    """
    ],
    [
    # Structural Reorganization
    """
    Rewrite the following news text by reorganizing sentence structure and altering the order of information to improve clarity or flow.  
    Keep all facts, dates, names, and events unchanged.

    Text:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite the news article using different wording and phrasing typical of professional journalism.  
    Maintain all factual details exactly as stated, without adding or removing information.

    Text:
    {}
    """,
    # Concision Improvement
    """
    Rewrite the news text to be clearer and more concise while maintaining a neutral reporting style.  
    Ensure that all factual elements remain unchanged.

    Text:
    {}
    """,
    # Tone Variation
    """
    Rewrite the news article with a slightly different journalistic voice—  
    for example, more formal, more direct, or more descriptive—while keeping the factual content strictly identical.

    Text:
    {}
    """
    ],
    [
    # Structural Reorganization 
    """
    Rewrite the following review by restructuring the sentences and reorganizing the flow,  
    while preserving all opinions, emotions, and experiences expressed by the reviewer.

    Review:
    {}
    """,
    # Tone Variation
    """
    Rewrite the following review using a different tone or writing voice  
    (slightly more formal, more conversational, or more direct),  
    while keeping the reviewer’s sentiment and experiences unchanged.

    Review:
    {}
    """,
    # Lexical Enhancement
    """
    Rewrite the following review by varying wording and expressions,  
    using different vocabulary while keeping the same meaning and emotional intensity.

    Review:
    {}
    """,
    # Pacing Adjustment
    """
    Rewrite the following review to improve clarity, pacing, and coherence.  
    You may rearrange sentences or rephrase them, but do not alter the core complaints or experiences.

    Review:
    {}
    """,
    ]
]

def rewrite_once(text, system_prompt, user_prompt, temperature, model="gpt-4o-mini"):
    """Call OpenAI to rewrite once"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "Target length: about 800 tokens."}
        ],
        temperature=temperature
    )
    print(f"Rewriting with temperature {temperature}, model {model}, system_prompt: {system_prompt[:50]}, user_prompt: {user_prompt[:50]}, text: {text[:50]}")
    return response.choices[0].message.content.strip()

def rewrite_iterative(text, system_prompt, user_prompts, rounds=16, model="gpt-4o-mini"):
    """Perform N rounds of recursive rewriting."""
    
    current_text = text
    results = []

    for r in range(1, rounds + 1):
        
        user_prompt = user_prompts[(r - 1) % len(user_prompts)].format(current_text)
        
        if (r < rounds // 2):
            temperature = 0.5   # low temperature for early rounds
        else:
            temperature = 0.7   # higher temperature for later rounds
        
        current_text = rewrite_once(current_text, system_prompt, user_prompt, temperature, model)

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

        for item in data[:1]:
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
