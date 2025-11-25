import os
import json
import asyncio
import aiofiles
from tqdm import tqdm
from openai import AsyncOpenAI
from config import CONFIG

client = AsyncOpenAI(
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

# concurrency settings
MAX_CONCURRENT = 30
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# auto save batch size
BATCH_SIZE = 200

# retry times
MAX_RETRIES = 3

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

async def safe_api_call(messages, model, temperature, max_retries=MAX_RETRIES):
    """
    Call OpenAI async API with light exponential backoff.
    Sleep times: 0, 0.3, 0.6, 1.2, ...
    """
    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=60
                )

            # validate response
            if (
                resp is None or
                not hasattr(resp, "choices") or
                len(resp.choices) == 0 or
                resp.choices[0] is None or
                getattr(resp.choices[0], "message", None) is None or
                getattr(resp.choices[0].message, "content", None) is None
            ):
                print(f"[safe_api_call] attempt={attempt}: empty/invalid response.")
            else:
                return resp.choices[0].message.content.strip()

        except Exception as e:
            print(f"[safe_api_call] attempt={attempt} exception: {e}")

        # ----------- short exponential backoff -----------
        if attempt < max_retries:
            sleep_time = 0.3 * (2 ** (attempt - 1))   # 0.3, 0.6, 1.2...
            await asyncio.sleep(sleep_time)

    print("[safe_api_call] all retries failed, returning empty string")
    return ""

async def rewrite_iterative(text, system_prompt, user_prompts, rounds=16, model="gpt-4o-mini"):
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
        
        new_text = await safe_api_call(messages, model=model, temperature=temperature)

        if not new_text:
            new_text = current_text

        current_text = new_text

        if(r % 4 == 0):
            results.append(current_text)

    return results

async def process_item(item_id, item, domain_idx):
    """Process a single dataset item asynchronously."""
    original = item[HUMAN_TEXT[domain_idx]]
    rewritten = await rewrite_iterative(original, SYSTEM_PROMPT[domain_idx], REWRITE_PROMPT[domain_idx], NUM_ITER, MODEL)

    return {
        "id": item_id,
        "domain": item["domain"],
        "human_text": original,
        "llm_text_1": rewritten[0],
        "llm_text_2": rewritten[1],
        "llm_text_3": rewritten[2],
        "llm_text_4": rewritten[3],
        "llm_type": MODEL,
    }


async def main():
    # load all items into a list of (item_id, item, domain_idx)
    tasks_meta = []
    item_id = 1
    for domain_idx, filename in enumerate(FILE_NAMES):
        path = os.path.join(INPUT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            tasks_meta.append((item_id, item, domain_idx))
            item_id += 1

    total = len(tasks_meta)
    print(f"Total items to process: {total}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # If there is existing partial output, load it and set start index
    existing_results = []
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as rf:
                existing_results = json.load(rf)
        except Exception:
            existing_results = []

    processed_ids = {r["id"] for r in existing_results} if existing_results else set()

    results_accum = existing_results[:]  # will accumulate results and be saved in batches

    # We'll process in batches to control memory and allow batch-saving
    pbar = tqdm(total=total, desc="Items", unit="item")
    # initialize progress bar with already processed
    pbar.update(len(processed_ids))

    idx = 0
    while idx < total:
        batch_meta = []
        # build a batch that excludes already processed ids
        while idx < total and len(batch_meta) < BATCH_SIZE:
            meta = tasks_meta[idx]
            if meta[0] not in processed_ids:
                batch_meta.append(meta)
            idx += 1

        if not batch_meta:
            continue

        # create tasks for this batch
        batch_tasks = [asyncio.create_task(process_item(item_id, item, domain_idx)) for (item_id, item, domain_idx) in batch_meta]

        # collect results as they complete and update progress bar
        for coro in asyncio.as_completed(batch_tasks):
            try:
                res = await coro
                results_accum.append(res)
                processed_ids.add(res["id"])
                pbar.update(1)
            except Exception as e:
                # Should not break the whole run; log and continue
                print(f"[run_all] task failed with exception: {e}")
                # mark as processed in terms of progress to avoid infinite loop
                # (optionally you could re-queue failed items)
                pbar.update(1)

        # batch save to disk (overwrite with accumulated results)
        try:
            async with aiofiles.open(OUTPUT_PATH, "w", encoding="utf-8") as wf:
                await wf.write(json.dumps(results_accum, indent=4, ensure_ascii=False))
            print(f"[run_all] Saved batch, total saved: {len(results_accum)}")
        except Exception as e:
            print(f"[run_all] Failed to save batch: {e}")

    pbar.close()
    print("All done.")


if __name__ == "__main__":
    asyncio.run(main())
    # sort by id
    try:
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except e:
        print(e)

    sorted_data = sorted(data, key=lambda x: x['id'])
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    print("Final sorting done.")