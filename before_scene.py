import os
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm
import networkx as nx
from pycocotools.coco import COCO
import nltk
import pickle

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import json
import numbers
import pickle

# Loading the COCO relationship data
with open("/home/staccato/Desktop/caption/scene_graph/output/all_relationships_2.pkl", "rb") as file:
    coco_relationships = pickle.load(file)

# Transforming Visual Genome relationships into the desired format
vg_transformed_relationships = []

def process_vg_chunk(chunk):
    try:
        data = json.loads(chunk)
        for entry in data:
            image_id = entry["image_id"]
            for relationship in entry["relationships"]:
                subject = relationship["subject"]["name"]
                predicate = relationship["predicate"]
                obj = relationship["object"]["name"]
                vg_transformed_relationships.append((image_id, (subject, predicate, obj)))
    except:
        pass  # In case of any invalid JSON chunk

# Reading the Visual Genome file in chunks and processing
chunk_size = 100000  # Size of each chunk in characters
with open("/home/staccato/Desktop/caption/scene_graph/vg/annotation/relationships.json", "r") as file:
    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        process_vg_chunk(chunk)

# Appending the transformed Visual Genome relationships to the COCO relationships
combined_relationships = coco_relationships + vg_transformed_relationships

def is_valid_entry(entry):
    # entry가 튜플이 아니라면 False 반환
    if not isinstance(entry, tuple):
        return False
    # 첫 번째 항목이 숫자면 False 반환 (이렇게 하면 숫자만 있는 항목들을 건너뛸 수 있습니다.)
    if isinstance(entry[0], numbers.Number):
        return False
    # 두 번째 항목이 튜플이어야 하며, 3개의 항목을 가져야 합니다.
    if not (isinstance(entry[1], tuple) and len(entry[1]) == 3):
        return False
    return True


print(f"Total number of entries in combined_relationships: {len(combined_relationships)}")
valid_entries = [rel for rel in combined_relationships if is_valid_entry(rel)]
print(f"Number of valid entries: {len(valid_entries)}")
print(combined_relationships[:10])

restructured_relationships = []
i = 0
while i < len(combined_relationships):
    if isinstance(combined_relationships[i], numbers.Number) and isinstance(combined_relationships[i+1], tuple):
        restructured_relationships.append((combined_relationships[i], combined_relationships[i+1]))
        i += 2
    else:
        i += 1

# restructured_relationships 리스트에 저장된 항목들을 파일에 저장
with open("/home/staccato/Desktop/caption/scene_graph/output/combined_relationships.txt", "w") as file:
    for rel in restructured_relationships:
        file.write(f"{rel[0]}, ({rel[1][0]}, {rel[1][1]}, {rel[1][2]})\n")


import networkx as nx
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import os

# -------------- Functions --------------

def create_scene_graph(relationships):
    graph = nx.DiGraph()
    for relationship in relationships:
        subject, predicate, obj = relationship
        graph.add_node(subject)
        graph.add_node(obj)
        graph.add_edge(subject, obj, predicate=predicate)
    return graph

def visualize_image_with_graph(image_path, graph, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    
    # Load and display image on the left
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img)
    axs[0].axis('off')  # Hide axes for image
    
    # Draw the graph on the right
    pos = nx.spring_layout(graph, k=0.5)  # Adjust k value for desired spacing
    nx.draw(graph, pos, with_labels=True, node_size=1500, node_shape='s', node_color="skyblue", 
            alpha=0.5, linewidths=40, ax=axs[1], arrowsize=20, width=2, edge_color="grey")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): graph[u][v]['predicate'] for u, v in graph.edges()}, 
                                 ax=axs[1], font_size=10)
    
    # Save the figure before showing
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    
    '''
    # Draw the graph on the right
    nx.draw(graph, pos, with_labels=True, node_size=5, node_shape='D', node_color="grey", 
            alpha=1.0, linewidths=10, ax=axs[1], arrowsize=10, width=2, edge_color="black")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): graph[u][v]['predicate'] for u, v in graph.edges()}, 
                                 ax=axs[1], font_size=12)
    
    plt.tight_layout()
    plt.show()
    '''

def get_image_path(image_id):
    image_id = str(image_id).zfill(12)
    
    paths = [
        f"/home/staccato/Desktop/caption/scene_graph/vg/VG_100K/{image_id}.jpg",
        f"/home/staccato/Desktop/caption/scene_graph/vg/VG_100K_2/{image_id}.jpg",
        f"/home/staccato/Desktop/caption/scene_graph/coco/Image/train2017/{image_id}.jpg",
        f"/home/staccato/Desktop/caption/scene_graph/coco/Image/test2017/{image_id}.jpg",
        f"/home/staccato/Desktop/caption/scene_graph/coco/Image/val2017/{image_id}.jpg"
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None

# -------------- Main Logic --------------

grouped_relationships = defaultdict(list)
for rel in restructured_relationships:
    image_id, relationship = rel
    grouped_relationships[image_id].append(relationship)

# For demonstration, visualize graph for a selected image_id
#selected_image_id = list(grouped_relationships.keys())[200]
#graph_for_selected_image = create_scene_graph(grouped_relationships[selected_image_id])

from tqdm import tqdm

# grouped_relationships의 모든 키(이미지 ID)에 대해 반복하면서 tqdm으로 진행 상태 표시
for image_id in tqdm(grouped_relationships.keys(), desc="Processing images"):
    
    graph = create_scene_graph(grouped_relationships[image_id])
    image_path = get_image_path(image_id)
    
    if image_path:
        output_path = f"/home/staccato/Desktop/caption/scene_graph/output/before_scene_graph/{image_id}_graph.png"
        visualize_image_with_graph(image_path, graph, output_path)
        print(f"Saved graph image for image_id: {image_id} at {output_path}")
        
    else:
        print(f"No image found for image_id: {image_id}")


'''
    # 콘솔에 노드와 에지 정보 출력
    print(f"\nImage ID: {image_id}")
    print("Nodes:")
    for node in graph.nodes():
        print(node)

    print("\nEdges:")
    for u, v, data in graph.edges(data=True):
        print(f"{u} --({data['predicate']})--> {v}")

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Relationships: {grouped_relationships[image_id]}")
'''

import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import random

# CUDA가 사용 가능한지 확인하고, 디바이스를 설정합니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델과 토크나이저를 로드합니다.
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-cased').to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)

def generate_caption_with_bert(graph):
    # Ensure model is in evaluation mode
    bert_model.eval()

    # Ensure the model is on the correct device
    bert_model.to('cuda')

    # Get unique edges by considering only the nodes, not the edge data
    edges = graph.edges(data=True)
    unique_edges = set((u, v) for u, v, d in edges)

    # Create sentence fragments and join them to form the complete sentence
    sentence_fragments = ["{} [MASK] {}".format(u, v) for u, v in unique_edges]
    sentence = ' '.join(sentence_fragments)

    # Tokenize the sentence and ensure the inputs are on the same device as the model
    inputs = bert_tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # Predict the [MASK] tokens using BERT
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits

    # Decode the predicted tokens
    mask_token_index = torch.where(inputs["input_ids"][0] == bert_tokenizer.mask_token_id)[0]
    predicted_tokens = [bert_tokenizer.decode([torch.argmax(logits[0, idx])]) for idx in mask_token_index]

    # Replace [MASK] with the predicted tokens
    for token in predicted_tokens:
        sentence = sentence.replace("[MASK]", token, 1)

    return sentence


def generate_caption_with_gpt2(graph):
    gpt2_model.eval()  # Set the model to evaluation mode
    gpt2_model.to('cuda')  # Move the model to the GPU

    # Get a list of unique edges by considering only the nodes, not the edge data
    edges = list(graph.edges())
    unique_edges = list(set(edges))  # Remove duplicate edges if any

    random.shuffle(unique_edges)
    selected_edges = unique_edges[:10]

    # Create sentence fragments for each unique edge
    sentence_fragments = [f"{u} {v}" for u, v in selected_edges]

    # If the list is too long, consider truncating or summarizing
    if len(sentence_fragments) > 10:  # Example threshold, adjust as needed
        sentence_fragments = sentence_fragments[:10]  # Keep only the first 10

    # Join the sentence fragments with a period and add a final period to indicate the end of the sentence
    sentence = '. '.join(sentence_fragments) + '.'

    # Tokenize the sentence and convert to tensor format, ensuring to move the tensor to the same device as the model
    inputs = gpt2_tokenizer.encode(sentence, return_tensors="pt")
    inputs = inputs.to('cuda')

    # Generate text using the GPT-2 model
    with torch.no_grad():
        outputs = gpt2_model.generate(
    inputs,
    max_length=30,
    num_return_sequences=1,
    num_beams=4,
    no_repeat_ngram_size=2,  # n-gram 반복을 방지
    temperature=0.9,  # 생성 다양성 조절
    pad_token_id=gpt2_tokenizer.eos_token_id
)

    # Decode the generated text to a string
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# 각 이미지를 처리하고 캡션을 생성하는 함수입니다.
def process_and_generate_captions(image_id, graph):
    # 캡션을 생성합니다.
    bert_caption = generate_caption_with_bert(graph)
    gpt2_caption = generate_caption_with_gpt2(graph)

    # 필요에 따라 캡션을 저장하거나 출력합니다.
    #print(f"이미지 ID: {image_id}")
    #print("BERT 캡션:", bert_caption)
    #print("GPT-2 캡션:", gpt2_caption)

     # 캡션을 텍스트 파일로 저장합니다.
    save_path = f"/home/staccato/Desktop/caption/scene_graph/output/before_caption/{image_id}_captions.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        file.write(f"BERT 캡션: {bert_caption}\n")
        file.write(f"GPT-2 캡션: {gpt2_caption}\n")
        
# grouped_relationships의 모든 키(이미지 ID)에 대해 반복하면서 tqdm으로 진행 상태 표시
for image_id in tqdm(grouped_relationships.keys(), desc="이미지 처리 중"):
    graph = create_scene_graph(grouped_relationships[image_id])
    process_and_generate_captions(image_id, graph)
