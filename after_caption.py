import json
import glob
from tqdm.auto import tqdm
import torch
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel

# JSON 파일 로딩 함수
def load_graph_data(file_path):
    with open(file_path, 'r') as file:
        graph_data = json.load(file)
    return graph_data

# 문장 생성 함수
def create_sentence_from_graph(links):
    sentences = []
    for link in links:
        sentence = f"{link['source']} {link['label']} {link['target']}."
        sentences.append(sentence)
    return " ".join(sentences)

# 모델과 토크나이저 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-cased').to(device).eval()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()

# BERT 캡션 생성 함수
def generate_caption_with_bert(sentence):
    inputs = bert_tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    mask_token_index = torch.where(inputs["input_ids"][0] == bert_tokenizer.mask_token_id)[0]
    predicted_tokens = [bert_tokenizer.decode([torch.argmax(logits[0, idx])]) for idx in mask_token_index]
    for token in predicted_tokens:
        sentence = sentence.replace("[MASK]", token, 1)
    return sentence

# GPT-2 캡션 생성 함수
def generate_caption_with_gpt2(sentence):
    inputs = gpt2_tokenizer.encode(sentence, return_tensors="pt")
    inputs = inputs.to(device)
    
    outputs = gpt2_model.generate(
        inputs, 
        max_length=35, 
        num_return_sequences=1, 
        num_beams=4, 
        no_repeat_ngram_size=2, 
        temperature=0.9, 
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 파일 저장 함수
def save_caption(image_id, bert_caption, gpt2_caption, filepath):
    with open(filepath, 'w') as file:
        file.write(f"Image ID: {image_id}\n")
        file.write(f"BERT Caption: {bert_caption}\n")
        file.write(f"GPT-2 Caption: {gpt2_caption}\n")

# 메인 로직
json_files = glob.glob('/home/staccato/Desktop/caption/scene_graph/output/after_graph/json/*.json')
for json_file in tqdm(json_files, desc="Generating captions"):
    
    # 파일 이름에서 숫자 부분만 image_id로 사용
    image_id = json_file.split('/')[-1].split('_')[2].split('.')[0]

    graph_data = load_graph_data(json_file)
    links = graph_data['links']
    
    # 그래프에서 문장 생성
    sentence = create_sentence_from_graph(links)

    if not sentence.strip():  # 문장이 비어있는지 확인
    #    print(f"No sentence was created for image ID: {image_id}. Check the graph data.")
        continue  # 다음 파일로 넘어갑니다

    #print(f"Generated sentence for image ID {image_id}: {sentence}")

    # 생성된 문장을 BERT와 GPT-2에 입력하여 캡션 생성
    bert_caption = generate_caption_with_bert(sentence)
    gpt2_caption = generate_caption_with_gpt2(sentence)

    # 파일 저장 경로 설정
    output_file_path = f'/home/staccato/Desktop/caption/scene_graph/output/caption_test/{image_id}_captions.txt'
    #print(f"Saving to file: {output_file_path}")  # 로깅 추가

    # 캡션 저장
    save_caption(image_id, bert_caption, gpt2_caption, output_file_path)
