from collections import Counter
import json
import re
from collections import defaultdict


import json
import numbers
import pickle

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
for i in range(0, len(combined_relationships)-1, 2):  # Step by two to handle (number, tuple) pairs
    if isinstance(combined_relationships[i], int) and isinstance(combined_relationships[i+1], tuple):
        restructured_relationships.append((combined_relationships[i], combined_relationships[i+1]))
    else:
        # Log or handle the case where the expected pair structure is not met
        pass  # Placeholder for any logging or error handling you want to add

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

def visualize_image_with_graph(image_path, graph):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    
    # Load and display image on the left
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img)
    axs[0].axis('off')  # Hide axes for image
    
    # Adjust the layout
    pos = nx.spring_layout(graph, k=0.5)  # Adjust k value for desired spacing
    
    # Draw the graph on the right
    nx.draw(graph, pos, with_labels=True, node_size=5, node_shape='D', node_color="grey", 
            alpha=1.0, linewidths=10, ax=axs[1], arrowsize=10, width=2, edge_color="black")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): graph[u][v]['predicate'] for u, v in graph.edges()}, 
                                 ax=axs[1], font_size=12)
    
    plt.tight_layout()
    plt.show()
    
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
selected_image_id = list(grouped_relationships.keys())[200]
graph_for_selected_image = create_scene_graph(grouped_relationships[selected_image_id])

# 이미지 경로 얻기
image_path = get_image_path(selected_image_id)
if image_path:
    visualize_image_with_graph(image_path, graph_for_selected_image)
else:
    print(f"No image found for image_id: {selected_image_id}")

print("\nNodes:")
for node in graph_for_selected_image.nodes():
    print(node)

print("\nEdges:")
for u, v, data in graph_for_selected_image.edges(data=True):
    print(f"{u} --({data['predicate']})--> {v}")

print(f"Number of nodes for the selected image: {graph_for_selected_image.number_of_nodes()}")
print(f"Number of edges for the selected image: {graph_for_selected_image.number_of_edges()}")
print(f"Relationships for the selected image: {grouped_relationships[selected_image_id]}")



# Function to get most common word pairs from a list of words
def get_most_common_pairs(words, n=5):
    pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return Counter(pairs).most_common(n)

# Function to print and save the most common word pairs for each image
def save_most_common_pairs(most_common_pairs_per_image, file_path):
    with open(file_path, "w") as file:
        for image_id, common_pairs in most_common_pairs_per_image.items():
            line = f"Image ID: {image_id}\n"
            file.write(line)
            
            for pair, count in common_pairs:
                line = f"{pair}: {count}\n"
                file.write(line)
            
            file.write("\n")

# Your provided code follows here...


# Load the COCO captions data
with open("/home/staccato/Desktop/caption/scene_graph/coco/annotations/captions_train2017.json", "r") as file:
    coco_captions = json.load(file)

annotations = coco_captions['annotations']

# Preprocess the captions
def preprocess_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^a-z\s]', '', caption)  #소문자로 변경, 공백제거.
    return caption.split()

image_words_dict = defaultdict(list)
for annotation in annotations:
    image_id = annotation['image_id']
    caption_words = preprocess_caption(annotation['caption'])
    image_words_dict[image_id].extend(caption_words)


# Define a list of common non-nouns (verbs, adjectives, prepositions, conjunctions, etc.)
non_nouns = set([
    "a", "an", "the", "is", "are", "was", "were", "has", "have", "had", "been", "will", "would", 
    "can", "could", "and", "or", "but", "on", "with", "without", "of", "at", "from", "into", 
    "during", "including", "until", "against", "among", "throughout", "despite", "towards", 
    "upon", "concerning", "to", "in", "for", "about", "by", "according", "like", "through", 
    "over", "before", "between", "after", "since", "while", "during", "below", "above", "up", 
    "down", "infront", "behind", "inside", "outside", "onto", "off", "over", "under", "again", 
    "further", "then", "once", "that"
])

# Filter only the nouns from the words
filtered_image_words_dict = {}
for image_id, words in image_words_dict.items():
    filtered_image_words_dict[image_id] = [word for word in words if word not in non_nouns]

# Extracting most common word pairs for each image using only the nouns
most_common_pairs_per_image_nouns = {image_id: get_most_common_pairs(words) for image_id, words in filtered_image_words_dict.items()}

# Save and print the results
file_path_nouns = "/home/staccato/Desktop/caption/scene_graph/output/most_common_word_pairs_nouns.txt"
save_most_common_pairs(most_common_pairs_per_image_nouns, file_path_nouns)

def load_data_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    current_set = set()
    for line in lines:
        line = line.strip()
        if line.startswith("Image ID:"):
            if current_set:
                data.append(list(current_set))
                current_set = set()
        else:
            pair = tuple(line.split(":")[0][1:-1].split(", "))
            current_set.update(pair)
    
    if current_set:
        data.append(list(current_set))
    
    return data

# Load data from the file
file_path = "/home/staccato/Desktop/caption/scene_graph/output/most_common_word_pairs_nouns.txt"
transactions = load_data_from_file(file_path)
print(transactions[:5])
##=================================================================================================================##

from collections import defaultdict, Counter
import itertools
from tqdm import tqdm

# Step 1: Define helper functions

def calculate_support(itemset, transactions):
    """Calculate the support of an itemset given a list of transactions."""
    count = sum(1 for transaction in transactions if set(itemset).issubset(transaction))
    return count / len(transactions)

def generate_candidate_itemsets(level_k, level_k_1):
    """Generate candidate itemsets of length k using the frequent itemsets of length k-1."""
    candidate_itemsets = []
    for itemset1 in level_k_1:
        for itemset2 in level_k_1:
            if itemset1[:-1] == itemset2[:-1] and itemset1[-1] < itemset2[-1]:
                candidate_itemsets.append(itemset1 + (itemset2[-1],))
    return candidate_itemsets

# Step 2: Apriori Algorithm

def apriori(transactions, min_support, min_confidence):
    """Apriori algorithm to extract frequent itemsets and association rules."""
    # Calculate the support of individual items
    individual_supports = Counter(item for transaction in transactions for item in transaction)
    total_transactions = len(transactions)
    individual_supports = {item: support/total_transactions for item, support in individual_supports.items()}
    
    # Filter items that do not meet the minimum support
    frequent_itemsets = {1: {tuple([item]) for item, support in individual_supports.items() if support >= min_support}}
    
    k = 2
    while True:
        # Generate candidate itemsets of length k
        candidate_itemsets = generate_candidate_itemsets(k, frequent_itemsets[k-1])
        
        # Using tqdm to show progress
        candidate_supports = {itemset: calculate_support(itemset, transactions) for itemset in tqdm(candidate_itemsets, desc=f'Calculating supports for itemsets of length {k}')}
        
        # Filter itemsets that do not meet the minimum support
        frequent_itemsets[k] = {itemset for itemset, support in candidate_supports.items() if support >= min_support}
        
        if not frequent_itemsets[k]:
            break
        
        k += 1
    
    # Generate association rules
    association_rules = []
    for k, itemsets in frequent_itemsets.items():
        if k == 1:
            continue
        for itemset in itemsets:
            for i in range(1, k):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    if antecedent and consequent:
                        rule_support = calculate_support(itemset, transactions)
                        antecedent_support = calculate_support(antecedent, transactions)
                        confidence = rule_support / antecedent_support
                        if confidence >= min_confidence:
                            association_rules.append((antecedent, consequent, rule_support, confidence))
    
    return frequent_itemsets, association_rules

# Step 3: Use the apriori function on the transactions
frequent_itemsets, rules = apriori(transactions, min_support=0.01, min_confidence=0.1)
sorted_rules = sorted(rules, key=lambda x: x[3], reverse=True)

# Save the association rules to a file
output_path = "/home/staccato/Desktop/caption/scene_graph/output/association_rules.txt"
with open(output_path, "w") as file:
    file.write("Antecedent -> Consequent : Support, Confidence\n")
    for rule in sorted_rules:
        file.write(f"{rule[0]} -> {rule[1]} : {rule[2]:.4f}, {rule[3]:.4f}\n")
#=====================================================================================================##

from collections import defaultdict

# COCO 데이터셋과 Visual Genome 데이터셋을 합친다.
combined_relationships = coco_relationships + vg_transformed_relationships

# 이미지 아이디별 객체 목록을 저장하기 위한 딕셔너리 초기화
image_objects = defaultdict(set)

# restructured_relationships를 순회하며 이미지 아이디별로 객체를 추가
for image_id, relationship in restructured_relationships:
    subject, _, obj = relationship
    image_objects[image_id].add(subject)
    image_objects[image_id].add(obj)

# 이미지 아이디별로 객체 목록을 리스트로 변환
for image_id in image_objects:
    image_objects[image_id] = list(image_objects[image_id])


# 이미지 아이디별 연관 규칙을 저장할 딕셔너리 초기화
image_association_rules = {}

# 이미지 아이디별로 연관 규칙 계산
for image_id, objects in image_objects.items():
    # 이미지에 대한 객체 목록을 transactions로 사용
    transactions = [objects]
    _, rules = apriori(transactions, min_support=0.01, min_confidence=0.1)
    image_association_rules[image_id] = rules
    
    
    
# 이미지 아이디별 연관 규칙을 파일에 저장
output_path = "/home/staccato/Desktop/caption/scene_graph/output/image_association_rules_id.txt"
with open(output_path, "w") as file:
    for image_id, rules in image_association_rules.items():
        file.write(f"Image ID: {image_id}\n")
        file.write("Antecedent -> Consequent : Support, Confidence\n")
        for rule in rules:
            file.write(f"{rule[0]} -> {rule[1]} : {rule[2]:.4f}, {rule[3]:.4f}\n")
        file.write("-" * 50 + "\n")


with open("/home/staccato/Desktop/caption/scene_graph/output/combined_relationships.txt", "r") as file:
    lines = file.readlines()

image_relationships = defaultdict(list)

for line in lines:
    components = line.strip().split(", ")
    image_id = int(components[0])
    rel = (components[1].strip("()"), components[2], components[3].strip("()"))
    image_relationships[image_id].append(rel)

with open("/home/staccato/Desktop/caption/scene_graph/output/classes_output_v2.csv", "r") as file:
    header = file.readline().strip().split(",")
    print(header)

##===========================================================================================================##

import networkx as nx
from collections import defaultdict
from tqdm.auto import tqdm

# 1. 장면 그래프 생성
def create_scene_graph(relationships):
    graph = nx.DiGraph()
    for relationship in relationships:
        subject, predicate, obj = relationship
        graph.add_node(subject)
        graph.add_node(obj)
        graph.add_edge(subject, obj, predicate=predicate)
    return graph

# 2. 객체 탐지와 연관 규칙을 기반으로 노드와 엣지의 중요도 계산
def calculate_importance(graph, association_rules):
    importance = defaultdict(float)
    total_edges = graph.number_of_edges()  # 전체 엣지 수 계산
    pbar = tqdm(total=total_edges, desc="Calculating importance")  # 전체 진행률 바 초기화
    for u, v, data in graph.edges(data=True):
        for antecedent, consequent, _, confidence in association_rules:
            if u in antecedent and v in consequent:
                importance[(u, v)] += confidence
        pbar.update(1)  # 진행률 바 업데이트
    pbar.close()  # 진행률 바 닫기
    return importance

# 3. 중요도에 따른 파티션 생성
def partition_hypergraph(graph, importance, threshold=0.5):
    subgraphs = []
    visited = set()
    
    for (u, v), value in importance.items():
        if value >= threshold and u not in visited and v not in visited:
            # 중요한 서브그래프 생성
            subgraph = graph.subgraph([u, v])
            subgraphs.append(subgraph)
            visited.update([u, v])
    
    # 방문하지 않은 노드에 대한 서브그래프 추가
    for node in graph.nodes():
        if node not in visited:
            subgraphs.append(graph.subgraph([node]))
    
    return subgraphs

# 이미지 아이디별 장면 그래프 및 연관 규칙으로 하이퍼그래프 파티션 생성
image_hypergraph_partitions = {}
for image_id in tqdm(image_relationships.keys(), desc="Processing images"):
    relationships = image_relationships[image_id]
    graph = create_scene_graph(relationships)
    importance = calculate_importance(graph, image_association_rules[image_id])
    partitions = partition_hypergraph(graph, importance)
    image_hypergraph_partitions[image_id] = partitions


# 필요한 라이브러리와 데이터를 불러옵니다.
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

# 이미지 ID 하나에 대한 장면 그래프 생성
image_id = list(image_relationships.keys())[200]  # 첫 번째 이미지 ID
relationships = image_relationships[image_id]
graph = create_scene_graph(relationships)

# 연관 규칙을 기반으로 중요도 계산
importance = calculate_importance(graph, image_association_rules[image_id])

# 중요도를 기반으로 하이퍼그래프 파티션 생성
partitions = partition_hypergraph(graph, importance)


import networkx as nx
from community import community_louvain

# Step 1: Define the parse_association_rule function
def parse_association_rule(rule_str):
    parts = rule_str.split(" -> ")
    antecedent = tuple(parts[0].strip("()").replace("'", "").split(", "))
    consequent = tuple(parts[1].split(" : ")[0].strip("()").replace("'", "").split(", "))
    support, confidence = map(float, parts[1].split(" : ")[1].split(", "))
    return antecedent, consequent, support, confidence

# Step 2: Load the association rules using the parse_association_rule function
with open("/home/staccato/Desktop/caption/scene_graph/output/association_rules.txt", "r") as file:
    lines = file.readlines()[1:]  # Skip the header
    rules_data = [parse_association_rule(line.strip()) for line in lines if line.strip()]

association_rules = rules_data


# Step 3: Load the detected objects (just a sample for now)
with open("/home/staccato/Desktop/caption/scene_graph/output/classes_output_v2.csv", "r") as file:
    detected_objects = [line.strip().split(",")[1] for line in file.readlines()][1:]

# Step 4: Hypergraph partitioning function
def hypergraph_partitioning(image_id, relationships, association_rules, detected_objects, threshold=0.5):
    graph = nx.Graph()
    for relationship in relationships:
        subject, predicate, obj = relationship
        graph.add_edge(subject, obj, weight=1.0)

    # Adjust weights based on association rules
    for antecedent, consequent, support, confidence in association_rules:
        if confidence >= threshold:
            nodes_in_antecedent = set(graph.nodes()).intersection(antecedent)
            nodes_in_consequent = set(graph.nodes()).intersection(consequent)
            for u in nodes_in_antecedent:
                for v in nodes_in_consequent:
                    if graph.has_edge(u, v):
                        graph[u][v]['weight'] += confidence

    # Adjust weights based on detected objects
    for node in graph.nodes():
        if node in detected_objects:
            for neighbor in graph.neighbors(node):
                graph[node][neighbor]['weight'] *= 2  # Double the weight for detected objects

    # Use the Louvain method for community detection
    communities = community_louvain.best_partition(graph, weight='weight')
    return communities

# Step 5: Testing the function on a sample
image_relationships = defaultdict(list)
with open("/home/staccato/Desktop/caption/scene_graph/output/combined_relationships.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        image_id, rel_str = line.strip().split(", ", 1)
        rel = tuple(rel_str.strip("()").split(", "))
        image_relationships[int(image_id)].append(rel)

sample_image_id = list(image_relationships.keys())[200]  #smaple id 변경
sample_relationships = image_relationships[sample_image_id]
communities = hypergraph_partitioning(sample_image_id, sample_relationships, association_rules, detected_objects)

# Display the result
for node, community_id in communities.items():
    print(f"Node: {node}, Community: {community_id}")

##-------------------------------------------------------------------------------------------------------------------------##
import pandas as pd
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt

# 1. 탐지된 객체의 빈도 계산
detected_objects_df = pd.read_csv("/home/staccato/Desktop/caption/scene_graph/output/classes_output_v2.csv")
object_frequencies = detected_objects_df.iloc[:, 1:].stack().value_counts(normalize=True).to_dict()

# 빈 값 제거
object_frequencies = {k: v for k, v in object_frequencies.items() if k and not pd.isna(k)}

# 2. 그래프 생성
def create_weighted_scene_graph(image_id, image_relationships, object_frequencies, association_rules):
    relationships = image_relationships[image_id]
    graph = nx.Graph()
    edge_labels = {}
    for relationship in relationships:
        subject, predicate, obj = relationship
        graph.add_node(subject, weight=object_frequencies.get(subject, 1))
        graph.add_node(obj, weight=object_frequencies.get(obj, 1))
        rule_weight = next((conf for antecedent, consequent, _, conf in association_rules if subject in antecedent and obj in consequent), 1)
        graph.add_edge(subject, obj, weight=rule_weight)
        edge_labels[(subject, obj)] = predicate
    return graph, edge_labels

def partition_and_visualize(image_id, image_relationships, object_frequencies, association_rules):
    # 가중치가 부여된 그래프 생성
    graph, edge_labels = create_weighted_scene_graph(image_id, image_relationships, object_frequencies, association_rules)

    # 커뮤니티 탐지 실행
    partition = community_louvain.best_partition(graph, weight='weight')

    # 결과 출력
    print("Nodes and their community IDs:")
    for node, community_id in partition.items():
        print(f"Node: {node}, Community: {community_id}")

    print("\nEdges and their labels:")
    for edge, label in edge_labels.items():
        print(f"Edge: {edge}, Label: {label}")

    # 그래프 시각화
    colors = [partition[node] for node in graph.nodes()]
    pos = nx.spring_layout(graph, k=0.5, iterations=50)  # Adjust layout parameters for better visualization
    plt.figure(figsize=(12,8))
    nx.draw(graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.RdYlBu, node_size=300, font_size=12)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    plt.show()

# 특정 이미지 ID에 대해 함수 실행
sample_image_id = list(image_relationships.keys())[200]
partition_and_visualize(sample_image_id, image_relationships, object_frequencies, association_rules)

#----------------------------------------------------------------------------------------------------------------
import pandas as pd
import networkx as nx
from collections import defaultdict
from networkx.algorithms import community

# 데이터 파일 경로 설정
relationships_path = "/home/staccato/Desktop/caption/scene_graph/output/combined_relationships.txt"
detected_objects_path = "/home/staccato/Desktop/caption/scene_graph/output/classes_output_v2.csv"
association_rules_path = "/home/staccato/Desktop/caption/scene_graph/output/association_rules.txt"

# 1. 이미지와 관계를 매핑하기 위해 데이터를 불러옵니다.
with open(relationships_path, "r") as file:
    lines = file.readlines()

# 이미지 ID와 관계를 매핑
image_relationships = defaultdict(list)
for line in lines:
    image_id, rel_str = line.strip().split(", ", 1)
    rel = tuple(rel_str.strip("()").split(", "))
    image_relationships[int(image_id)].append(rel)

# 2. 객체 탐지 결과를 불러옵니다.
detected_objects_df = pd.read_csv(detected_objects_path)
detected_objects = detected_objects_df.iloc[:, 1:].stack().value_counts(normalize=True).to_dict()

# 빈 값 제거
detected_objects = {k: v for k, v in detected_objects.items() if k and not pd.isna(k)}

# 3. 연관규칙 데이터 불러오기
with open(association_rules_path, "r") as file:
    lines = file.readlines()

association_rules = {}
for line in lines[1:]:
    antecedent, rest = line.strip().split(" -> ")
    consequent, metrics = rest.split(" : ")
    _, confidence = metrics.split(", ")
    confidence = float(confidence)
    antecedent = tuple(antecedent.strip("()").split(", "))
    consequent = tuple(consequent.strip("()").split(", "))
    association_rules[(antecedent, consequent)] = confidence

# # 샘플 이미지 ID 선택
sample_image_id = list(image_relationships.keys())[200]  #이미지
sample_relationships = image_relationships[sample_image_id]

# 4. 해당 이미지의 장면 그래프를 생성합니다.
graph = nx.Graph()
for relationship in sample_relationships:
    subject, predicate, obj = relationship
    
    # 모든 이름을 소문자로 변환
    subject = subject.lower()
    obj = obj.lower()
    predicate = predicate.lower()

    # 노드 추가 (객체 탐지 결과의 빈도수를 가중치로 사용)
    graph.add_node(subject, weight=detected_objects.get(subject, 1))
    graph.add_node(obj, weight=detected_objects.get(obj, 1))
    
    # 연관규칙에서 신뢰도 찾기
    confidence = association_rules.get((subject, obj), 0)  # 만약 (subject, obj) 키에 해당하는 값이 association_rules 딕셔너리에 존재한다면 그 값을 반환
    graph.add_edge(subject, obj, weight=confidence, label=predicate)  # 'label' 속성 추가


# 5. 그래프를 두 파티션으로 분할합니다.
communities_generator = community.girvan_newman(graph)
top_level_communities = next(communities_generator)

# 결과 출력
for idx, partition in enumerate(top_level_communities, 1):
    print(f"Partition {idx}:", partition)

#-----------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as community


# 여기서 save_path를 실제 파일을 저장할 경로로 변경해야 합니다.
save_path = "/home/staccato/Desktop/caption/scene_graph/output/after_graph"

def draw_graph_with_labels(graph, pos, partitions, node_colors, image_id, save_path):
    plt.figure(figsize=(12, 8))
    # 노드와 엣지 그리기
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, font_size=20, node_size=500, edge_color="gray")
    edge_labels = {(u, v): graph[u][v]['label'] for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=20, bbox=dict(alpha=0))
    plt.title(f"Graph Partitioning for Image ID: {image_id}")
    plt.savefig(f"{save_path}/graph_{image_id}.png")  # 파일로 저장
    plt.close()  # 그래프 닫기

# 노드별 커뮤니티 ID를 매핑하여 색상 지정
color_map = {   # 추가로 더 많은 색상을 지정할 수 있다.
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "purple", 
    5: "grey"   
}

colors = []
for node in graph.nodes():

            
    for idx, partition in enumerate(top_level_communities):
        if node in partition:
            colors.append(color_map[idx])
            break
        

#pos = nx.spring_layout(graph)
# draw_graph_with_labels(graph, pos, top_level_communities, colors)

def remove_edges_between_partitions(G, partitions):
    """
    파티션 간의 엣지를 그래프에서 제거합니다.
    """
    # 모든 파티션 쌍에 대해
    for i in range(len(partitions)):
        for j in range(i+1, len(partitions)):
            # 파티션 i와 j 간의 모든 가능한 노드 쌍에 대해
            for node_i in partitions[i]:
                for node_j in partitions[j]:
                    # 엣지가 존재하면 제거합니다.
                    if G.has_edge(node_i, node_j):
                        G.remove_edge(node_i, node_j)
    return G

# 파티션 간의 엣지 제거
graph = remove_edges_between_partitions(graph, top_level_communities)
pos = nx.spring_layout(graph)
# 업데이트된 그래프를 다시 시각화합니다.
draw_graph_with_labels(graph, pos, top_level_communities, colors, image_id, save_path)


def save_graph_data(graph, image_id, save_path):
    # 그래프 데이터를 JSON 형식으로 저장
    data = nx.node_link_data(graph)
    with open(f"{save_path}/graph_data_{image_id}.json", 'w') as f:
        json.dump(data, f)
        
        
def create_scene_graph_for_image_id(image_id, image_relationships, detected_objects, association_rules):
    relationships = image_relationships[image_id]
    graph = nx.Graph()
    for relationship in relationships:
        subject, predicate, obj = relationship
        subject = subject.lower()
        obj = obj.lower()
        predicate = predicate.lower()

        graph.add_node(subject, weight=detected_objects.get(subject, 0))
        graph.add_node(obj, weight=detected_objects.get(obj, 0))

        # 연관규칙에서 신뢰도 찾기
        confidence = association_rules.get((subject, obj), 0)
        graph.add_edge(subject, obj, weight=confidence, label=predicate)
        
    return graph        

def ensure_directory_exists(path):  # 디렉토리가 있는지 확인하고 없으면 생성하는 함수
    if not os.path.exists(path):
        os.makedirs(path)


# 메인 루프에서 그래프 시각화 및 저장
for image_id in tqdm(image_relationships.keys(), desc="Processing images"):
    graph = create_scene_graph_for_image_id(image_id, image_relationships, detected_objects, association_rules)
    
    # 여기서 커뮤니티 생성기를 초기화합니다
    communities_generator = community.girvan_newman(graph)

    
    try:
        # 첫 번째 커뮤니티 분할을 얻습니다
        top_level_communities = next(communities_generator)
        node_to_community = {node: idx for idx, community in enumerate(top_level_communities) for node in community}
        
        # 파티션 간 엣지 제거
        graph = remove_edges_between_partitions(graph, top_level_communities)
        
        # 노드 색상 계산 - 메인 루프 내부로 이동
        colors = [color_map.get(node_to_community[node], "black") for node in graph.nodes()]

        # 레이아웃 계산
        pos = nx.spring_layout(graph)
        
        # 디렉토리 확인 및 생성
        ensure_directory_exists(save_path)
        graph_image_path = os.path.join(save_path, "png")
        ensure_directory_exists(graph_image_path)
        graph_data_path = os.path.join(save_path, "json")
        ensure_directory_exists(graph_data_path)

        # 그래프 이미지 그리고 저장
        draw_graph_with_labels(graph, pos, top_level_communities, colors, image_id, graph_image_path)

        # 그래프 데이터를 JSON 형식으로 저장
        save_graph_data(graph, image_id, graph_data_path)
        
    except StopIteration:
        print(f"이미지 {image_id}에 대한 추가 분할이 불가능합니다.")

#-------------------------------------------------------------

import matplotlib.pyplot as plt
import networkx as nx
import json
import os

# 이미지별로 그래프 파티션을 시각화하여 저장하는 함수
def save_graph_partitions(image_id, partitions, directory="/home/staccato/Desktop/caption/scene_graph/output/partition_graph"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    visual_directory = os.path.join(directory, "visuals")
    json_directory = os.path.join(directory, "json")
    if not os.path.exists(visual_directory):
        os.makedirs(visual_directory)
    if not os.path.exists(json_directory):
        os.makedirs(json_directory)
    
    for i, partition in enumerate(partitions):
        visual_filename = f"{image_id}_partition_{i+1}.png"
        visual_filepath = os.path.join(visual_directory, visual_filename)
        
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(partition)
        nx.draw(partition, pos, with_labels=True, node_size=700, node_color='lightblue')
        edge_labels = nx.get_edge_attributes(partition, 'predicate')
        nx.draw_networkx_edge_labels(partition, pos, edge_labels=edge_labels)
        plt.title(f"Image ID: {image_id}, Partition: {i+1}")
        plt.savefig(visual_filepath)
        plt.close()
        
        json_filename = f"{image_id}_partition_{i+1}.json"
        json_filepath = os.path.join(json_directory, json_filename)
        
        partition_data = nx.node_link_data(partition)  
        with open(json_filepath, 'w') as json_file:
            json.dump(partition_data, json_file)

for image_id in tqdm(image_relationships.keys(), desc="Saving graph partitions"):
    relationships = image_relationships[image_id]
    graph = create_scene_graph(relationships)
    importance = calculate_importance(graph, image_association_rules[image_id])
    partitions = partition_hypergraph(graph, importance)
    save_graph_partitions(image_id, partitions)

#--------------------------------------------------------------------------------------------------

import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import random
import glob


# 이미지, 노드, 엣지 정보 출력
print(f"Image ID: {sample_image_id}")
print("Nodes:", graph.nodes())
print("Edges:", graph.edges(data=True))

# 파라미터 카운트 함수 정의
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = "cuda" if torch.cuda.is_available() else "cpu"

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-cased').to(device).eval()

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()


def after_generate_caption_with_bert(graph):
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


def after_generate_caption_with_gpt2(graph):
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


# 그래프를 사용하여 캡션 생성
bert_caption = after_generate_caption_with_bert(graph)
gpt2_caption = after_generate_caption_with_gpt2(graph)

print("-------------------------결과------------------------------------")
print("After Relation Mining BERT generated caption:", bert_caption)
print(f"BERT total trainable parameters: {count_parameters(bert_model):,}")
print("After Relation Mining GPT-2 generated captions with Beam Search:")
print(f"{gpt2_caption}")
print(f"GPT-2 total trainable parameters: {count_parameters(gpt2_model):,}")


# 파일 저장 함수 정의
def save_caption_and_parameters(image_id, bert_caption, gpt2_caption, bert_params, gpt2_params, filepath):
    with open(filepath, 'w') as file:
        file.write(f"Image ID: {image_id}\n")
        file.write(f"Bert Caption: {bert_caption}\n")
        #file.write(f"Bert Trainable Parameters: {bert_params}\n")
        file.write(f"GPT-2 Caption: {gpt2_caption}\n")
        #file.write(f"GPT-2 Trainable Parameters: {gpt2_params}\n")

# 이미지 ID별로 캡션 생성 및 저장
for image_id, partitions in tqdm(image_hypergraph_partitions.items(), desc="Generating captions"):
    for partition in partitions:
        bert_caption = after_generate_caption_with_bert(partition)
        gpt2_caption = after_generate_caption_with_gpt2(partition)
        bert_params = count_parameters(bert_model)
        gpt2_params = count_parameters(gpt2_model)
        
        # 파일 경로는 실제 사용 환경에 맞게 수정
        filepath = f"/home/staccato/Desktop/caption/scene_graph/output/after_caption/{image_id}_captions.txt"
        save_caption_and_parameters(image_id, bert_caption, gpt2_caption, bert_params, gpt2_params, filepath)

print("캡션 생성 및 저장이 완료되었습니다.")