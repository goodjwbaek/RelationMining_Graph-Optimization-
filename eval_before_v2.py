# %%
#필요한 라이브러리 설치
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List
from nltk.tokenize import sent_tokenize


from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.meteor_score import meteor_score
import os
from rouge import Rouge

from nltk.tokenize import word_tokenize
from tqdm import tqdm


# %%
import json

def load_label_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extracting captions from the 'annotations' key
    label_data = {}
    for item in data['annotations']:
        image_id = item['image_id']
        caption = item['caption']
        if image_id in label_data:
            label_data[image_id].append(caption)
        else:
            label_data[image_id] = [caption]
    
    return label_data
# Load and inspect the data
label_data = load_label_data('/home/staccato/Desktop/caption/scene_graph/coco/annotations/captions_train2017.json')


def preprocess_caption(caption):
    # 캡션을 문장으로 분할
    sentences = sent_tokenize(caption)
    # 첫 번째 문장만 반환
    return sentences[0] if sentences else caption

def load_generated_captions(directory):
    generated_captions = {}
    files = os.listdir(directory)
    if not files:
        print(f"No files found in the directory {directory}")
        return generated_captions

    for filename in files:
        with open(os.path.join(directory, filename), 'r') as file:
            lines = file.readlines()
            try:
                image_id = int(filename.split('_')[0])
                bert_caption = preprocess_caption(lines[0].split(':')[1].strip())
                gpt2_caption = preprocess_caption(lines[1].split(':')[1].strip())
                generated_captions[image_id] = [bert_caption, gpt2_caption]
            except ValueError as e:
                print(f"Error in file {filename}: {e}")
                print("File content for debugging:")
                print(lines)
                break
    return generated_captions

# 사용 예시
before_caption = load_generated_captions('/home/staccato/Desktop/caption/scene_graph/output/before_caption/')
print(f"Number of loaded captions: {len(before_caption)}")

after_caption = load_generated_captions('/home/staccato/Desktop/caption/scene_graph/output/after_caption/')
print(f"Number of loaded captions: {len(after_caption)}")


##성능평가 메트릭
rouge = Rouge()
smoothing_function = SmoothingFunction().method1

# weights 설정 (1-gram, 2-gram, 3-gram, 4-gram 각각에 대한 가중치)
weights = (0.25, 0.25, 0.25, 0.25)

def calculate_individual_performance_metrics(generated_captions, label_data):
    total_bleu_bert, total_bleu_gpt2 = 0, 0
    total_rouge_bert, total_rouge_gpt2 = 0, 0
    total_meteor_bert, total_meteor_gpt2 = 0, 0
    count = 0

    for image_id, captions in tqdm(generated_captions.items(), desc="Calculating metrics"):
        if image_id in label_data:
            references = label_data[image_id]
            bert_caption, gpt2_caption = captions

            bert_tokens = word_tokenize(bert_caption)
            gpt2_tokens = word_tokenize(gpt2_caption)

            # 누적 점수를 위한 임시 변수
            temp_bleu_bert, temp_bleu_gpt2 = 0, 0
            temp_rouge_bert, temp_rouge_gpt2 = 0, 0
            temp_meteor_bert, temp_meteor_gpt2 = 0, 0

            for reference in references:
                reference_tokens = word_tokenize(reference)

                # BLEU 점수 계산 (N-gram Weighting 적용)
                bleu_bert = sentence_bleu([reference_tokens], bert_tokens, weights=weights, smoothing_function=smoothing_function)
                bleu_gpt2 = sentence_bleu([reference_tokens], gpt2_tokens, weights=weights, smoothing_function=smoothing_function)

                # ROUGE 점수 계산
                rouge_bert = rouge.get_scores(' '.join(bert_tokens), ' '.join(reference_tokens))[0]['rouge-1']['f']
                rouge_gpt2 = rouge.get_scores(' '.join(gpt2_tokens), ' '.join(reference_tokens))[0]['rouge-1']['f']

                # METEOR 점수 계산
                meteor_bert = single_meteor_score(reference_tokens, bert_tokens)
                meteor_gpt2 = single_meteor_score(reference_tokens, gpt2_tokens)

                # 각 레퍼런스에 대한 점수 합산
                temp_bleu_bert += bleu_bert
                temp_bleu_gpt2 += bleu_gpt2
                temp_rouge_bert += rouge_bert
                temp_rouge_gpt2 += rouge_gpt2
                temp_meteor_bert += meteor_bert
                temp_meteor_gpt2 += meteor_gpt2

            # 평균 점수 계산
            total_bleu_bert += temp_bleu_bert / len(references)
            total_bleu_gpt2 += temp_bleu_gpt2 / len(references)
            total_rouge_bert += temp_rouge_bert / len(references)
            total_rouge_gpt2 += temp_rouge_gpt2 / len(references)
            total_meteor_bert += temp_meteor_bert / len(references)
            total_meteor_gpt2 += temp_meteor_gpt2 / len(references)
            count += 1

    # 최종 평균 점수 계산
    average_bleu_bert = total_bleu_bert / count if count else 0
    average_bleu_gpt2 = total_bleu_gpt2 / count if count else 0
    average_rouge_bert = total_rouge_bert / count if count else 0
    average_rouge_gpt2 = total_rouge_gpt2 / count if count else 0
    average_meteor_bert = total_meteor_bert / count if count else 0
    average_meteor_gpt2 = total_meteor_gpt2 / count if count else 0

    return (average_bleu_bert, average_rouge_bert, average_meteor_bert), (average_bleu_gpt2, average_rouge_gpt2, average_meteor_gpt2)

# 성능 지표 계산
bert_metrics, gpt2_metrics = calculate_individual_performance_metrics(before_caption, label_data)
print("BERT Metrics - BLEU: {}, ROUGE: {}, METEOR: {}".format(*bert_metrics))
print("GPT-2 Metrics - BLEU: {}, ROUGE: {}, METEOR: {}".format(*gpt2_metrics))


# # Assuming pycocoevalcap is installed
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice

# def calculate_cider_spice(after_caption, label_data):
#     cider = Cider()
#     spice = Spice()

#     # Prepare your data in the format required by the library
#     # Typically, it's a dictionary with image IDs as keys and lists of captions as values

#     # Calculate CIDEr and SPICE
#     cider_score, _ = cider.compute_score(label_data, after_caption)
#     spice_score, _ = spice.compute_score(label_data, after_caption)

#     return cider_score, spice_score

# # Just for debugging purposes
# print("First few keys in label_data: ", list(label_data.keys())[:10])
# print("First few keys in after_caption: ", list(after_caption.keys())[:10])

# # 공통 키 찾기
# common_keys = set(label_data.keys()) & set(after_caption.keys())

# # 공통 키를 사용하여 서브셋 생성
# filtered_label_data = {k: label_data[k] for k in common_keys}
# filtered_after_caption = {k: after_caption[k] for k in common_keys}

# # Ensure that the label data is in the correct format for CIDEr and SPICE
# formatted_filtered_label_data = {image_id: [caption] for image_id, captions in filtered_label_data.items() for caption in captions}

# # CIDEr and SPICE 점수 계산
# cider_score, spice_score = calculate_cider_spice(filtered_after_caption, formatted_filtered_label_data)
# print(f"CIDEr: {cider_score}, SPICE: {spice_score}")

