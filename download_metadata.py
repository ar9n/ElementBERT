import findpapers
import datetime
import json
import os
import time

#IEEE_API_TOKEN = os.environ.get("IEEE_API_TOKEN")
#if IEEE_API_TOKEN is None:
#    raise RuntimeError("IEEE_API_TOKEN environment variable is not set")

ELEMENT_LIST_FILE = "ansys_element_types.txt"
DATA_PATH = "../datasets/Elements"
QUERY_DELAY_SECONDS = 2

def load_element_list(path: str = ELEMENT_LIST_FILE) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

engine = findpapers.Engine(
    #ieee_api_key=IEEE_API_TOKEN
)

element_list = load_element_list(ELEMENT_LIST_FILE)

# Search for papers across all databases
for element in element_list:
    if os.path.exists(os.path.join(DATA_PATH, f'{element}.json')):
        print(f"Skipping {element} as it already exists.")
        continue
    print(f"Searching for papers related to: {element}")
    search_result = engine.search(
        query=f'[{element}]',
        databases=['openalex'],#['ieee', 'arxiv', 'semantic_scholar'],
        num_workers=4,
    )
    findpapers.save_to_json(search_result, os.path.join(DATA_PATH, f'{element}.json'))

    time.sleep(QUERY_DELAY_SECONDS)

