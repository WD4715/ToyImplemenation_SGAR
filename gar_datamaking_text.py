# import os
# import json
# import re
# from typing import Dict, List

# from tqdm import tqdm

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # ==============================
# # 0. 경로 / 설정
# # ==============================

# DATA_ROOT = "/mnt/e/research/motion_data"
# PART_NAMES = ["right_arm", "left_arm", "right_leg", "left_leg", "torso"]

# MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# DTYPE = torch.float16  # 필요하면 bfloat16 또는 float32로 변경
# BATCH_SIZE = 8         # VRAM 보고 4 / 8 / 16 정도로 조절

# # ==============================
# # 1. 캡션 전처리
# # ==============================

# def clean_caption(raw_caption: str) -> str:
#     """
#     HumanML 스타일 캡션:
#       "A person is walking forwards.#A/DET person/NOUN is/AUX walk/VERB forwards/ADV#0.0#0.0"
#     이런 형태이므로, '#' 앞 부분만 사용.
#     """
#     if "#" in raw_caption:
#         return raw_caption.split("#")[0].strip()
#     return raw_caption.strip()

# # ==============================
# # 2. Mistral 모델 로딩 (전역에서 한 번만)
# # ==============================

# print(f"[INFO] Loading Mistral model: {MISTRAL_MODEL_NAME}")
# tokenizer = AutoTokenizer.from_pretrained(
#     MISTRAL_MODEL_NAME,
#     use_fast=False,  # fast tokenizer 에러 피하기
# )

# # decoder-only + generation에서 안전한 설정
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"  # decoder-only 주의 경고 해결
# model = AutoModelForCausalLM.from_pretrained(
#     MISTRAL_MODEL_NAME,
#     torch_dtype=DTYPE,          # float16
#     device_map={"": "cuda"},    # 또는 그냥 "cuda"
#     low_cpu_mem_usage=True,
# )

# model.eval()
# print("[INFO] Mistral loaded.")

# # 공용 dummy
# DUMMY_PARTS = {
#     "right_arm": "undefined",
#     "left_arm": "undefined",
#     "right_leg": "undefined",
#     "left_leg": "undefined",
#     "torso": "undefined",
# }

# # ==============================
# # 3. 프롬프트 & JSON 파싱 유틸
# # ==============================

# def build_prompt(caption: str) -> str:
#     """
#     Mistral Instruct용 채팅 템플릿 프롬프트 생성.
#     JSON만 출력하도록 강하게 지시.
#     """
#     system_msg = (
#         "You are an assistant that ONLY outputs valid JSON.\n"
#         "You analyze a short description of a human action and write separate descriptions "
#         "for each body part in English.\n"
#         "Body parts: right_arm, left_arm, right_leg, left_leg, torso.\n"
#         "Each value must be a short natural language phrase (at most 20 words).\n"
#         "If a body part is not clearly mentioned, write \"undefined\" for that part.\n"
#         "IMPORTANT RULES:\n"
#         "1. Output ONLY a single JSON object.\n"
#         "2. DO NOT add any explanation, markdown, comments, or extra text.\n"
#         "3. DO NOT wrap the JSON in backticks.\n"
#         "4. NEVER use \"...\" as a value.\n"
#         "5. Keys must be exactly: right_arm, left_arm, right_leg, left_leg, torso.\n"
#     )

#     user_msg = (
#         f"Caption: {caption}\n\n"
#         "Return a JSON object exactly in this structure, but with real descriptions:\n"
#         "{\n"
#         '  \"right_arm\": \"...\",\n'
#         '  \"left_arm\": \"...\",\n'
#         '  \"right_leg\": \"...\",\n'
#         '  \"left_leg\": \"...\",\n'
#         '  \"torso\": \"...\"\n'
#         "}\n"
#     )

#     messages = [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_msg},
#     ]

#     # Mistral chat 템플릿으로 프롬프트 생성
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     return prompt


# def extract_json_from_text(text: str) -> Dict[str, str]:
#     """
#     모델 출력에서 JSON 부분만 잘라서 dict로 파싱.
#     - 첫 '{' 부터 마지막 '}' 까지 잘라서 시도
#     """
#     text = text.strip()
#     # 혹시라도 ```json ...``` 이런 게 섞여 있으면 제거
#     text = text.replace("```json", "```")
#     text = text.replace("```JSON", "```")

#     start = text.find("{")
#     end = text.rfind("}")

#     if start == -1 or end == -1 or end <= start:
#         raise ValueError("No JSON object found in model output.")

#     json_str = text[start : end + 1]
#     data = json.loads(json_str)
#     return data

# # ==============================
# # 4. Mistral 호출 함수 (배치 버전)
# # ==============================

# def call_llm_for_parts_batch(captions: List[str], global_start_idx: int = 0) -> List[Dict[str, str]]:
#     """
#     여러 개의 caption을 한 번에 Mistral로 처리.
#     captions: 길이 B 리스트
#     global_start_idx: 전체 데이터 기준 start index (디버그 로그용)
#     반환: 길이 B 리스트, 각 원소는 {part_name: str}
#     """
#     # 1) 프롬프트 리스트 생성
#     prompts = [build_prompt(c) for c in captions]

#     # 2) 토크나이즈 (배치)
#     inputs = tokenizer(
#         prompts,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#         padding=True,  # 배치 padding
#     ).to(model.device)

#     input_ids = inputs["input_ids"]
#     attn_mask = inputs["attention_mask"]
#     prompt_len = input_ids.shape[1]

#     # 3) 생성
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids=input_ids,
#             attention_mask=attn_mask,
#             max_new_tokens=128,   # JSON만 필요하므로 80 정도면 충분 (원하면 더 줄여도 됨)
#             temperature=0.0,     # deterministic
#             top_p=1.0,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     # 4) 프롬프트 이후 토큰만 completion으로 추출
#     gen_ids = output_ids[:, prompt_len:]
#     texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

#     results: List[Dict[str, str]] = []

#     for i, resp_text in enumerate(texts):
#         idx = global_start_idx + i

#         # 앞 몇 개는 raw output도 찍어보자 (디버그)
#         if idx < 3:
#             print("\n[RAW LLM OUTPUT]")
#             print(resp_text)
#             print("==============")

#         try:
#             part_desc = extract_json_from_text(resp_text)
#         except Exception as e:
#             print(f"[WARN] JSON parse failed (idx={idx}), use dummy. Error: {e}")
#             part_desc = dict(DUMMY_PARTS)  # copy

#         # 누락된 key 채워주기 + 타입 보정
#         clean_desc: Dict[str, str] = {}
#         for k in PART_NAMES:
#             v = part_desc.get(k, "undefined")
#             if not isinstance(v, str):
#                 v = str(v)
#             clean_desc[k] = v

#         results.append(clean_desc)

#         # 디버그용: 처음 몇 개 샘플은 파싱 결과도 찍기
#         if idx < 3:
#             print("[PARSED PARTS]")
#             print(clean_desc)
#             print("==============")

#     return results

# # ==============================
# # 5. 하나의 GAR JSON 파일을 Mistral로 채우기
# # ==============================

# def fill_gar_file_with_mistral(input_path: str, output_path: str, max_samples: int = None):
#     """
#     input_path: 현재 id + caption만 있는 gar_train.json / gar_val.json
#     output_path: Mistral로 part_descriptions 채운 새 JSON 경로
#     max_samples: 디버그용으로 앞에서 몇 개만 처리하고 싶을 때 사용 (None이면 전체)
#     """
#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     print(f"[INFO] Loaded {len(data)} samples from {input_path}")

#     if max_samples is not None:
#         data = data[:max_samples]
#         print(f"[INFO] Debug mode: only first {max_samples} samples will be processed.")

#     new_data: List[dict] = []

#     # 배치 단위로 처리
#     total = len(data)
#     for start in tqdm(range(0, total, BATCH_SIZE), desc=f"Filling {os.path.basename(input_path)}"):
#         end = min(start + BATCH_SIZE, total)
#         batch_items = data[start:end]

#         captions = [clean_caption(item.get("caption", "")) for item in batch_items]

#         # LLM 호출 (배치)
#         batch_parts = call_llm_for_parts_batch(captions, global_start_idx=start)

#         # 결과 합치기
#         for item, parts in zip(batch_items, batch_parts):
#             new_item = dict(item)
#             new_item["part_descriptions"] = {p: parts.get(p, "undefined") for p in PART_NAMES}
#             new_data.append(new_item)

#             # 처음 몇 개는 디버그 출력 (이미 batch 함수에서 raw 출력/parsed 찍어주지만 요약 느낌)
#             global_idx = len(new_data) - 1
#             if global_idx < 3:
#                 print("\n[DEBUG SAMPLE]")
#                 print("ID:       ", item.get("id", ""))
#                 print("Caption:  ", clean_caption(item.get("caption", "")))
#                 print("Parts:    ", new_item["part_descriptions"])

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(new_data, f, ensure_ascii=False, indent=2)

#     print(f"[INFO] Saved filled GAR file → {output_path}")

# # ==============================
# # 6. main
# # ==============================

# if __name__ == "__main__":
#     gar_train_in = os.path.join(DATA_ROOT, "gar_train.json")
#     gar_val_in = os.path.join(DATA_ROOT, "gar_val.json")

#     gar_train_out = os.path.join(DATA_ROOT, "gar_train_mistral.json")
#     gar_val_out = os.path.join(DATA_ROOT, "gar_val_mistral.json")

#     # 먼저 작은 max_samples로 테스트해서 JSON이 잘 나오는지 확인하고,
#     # 괜찮으면 max_samples=None으로 바꿔서 전체 돌리는 걸 추천.
#     fill_gar_file_with_mistral(gar_train_in, gar_train_out, max_samples=32)
#     fill_gar_file_with_mistral(gar_val_in, gar_val_out, max_samples=32)


import os
import json
from typing import Dict, List, Any, Optional, Tuple

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 0. 경로 / 설정
# ============================================================

DATA_ROOT = "/mnt/e/research/motion_data"
SPLIT_JOINTS_DIR = os.path.join(DATA_ROOT, "split_joints")
TEXT_DIR = os.path.join(DATA_ROOT, "texts")  # 네가 쓰던 구조 유지

PART_NAMES = ["right_arm", "left_arm", "right_leg", "left_leg", "torso"]

MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DTYPE = torch.float16
BATCH_SIZE = 8

# 생성 옵션
MAX_INPUT_LEN = 512
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False

# 파싱 실패 시 재시도 (온도 조금 올려서)
ENABLE_RETRY = True
RETRY_TEMPERATURE = 0.2
RETRY_DO_SAMPLE = True

# caption이 없으면 스킵할지/빈 문자열로 진행할지
SKIP_IF_NO_CAPTION = True

# resume용: 이미 output에 있는 id는 스킵
ENABLE_RESUME = True

# ============================================================
# 1. 캡션 전처리
# ============================================================

def clean_caption(raw_caption: Any) -> str:
    """
    HumanML 스타일 캡션:
      "A person is walking forwards.#A/DET person/NOUN ... #0.0#0.0"
    -> '#' 앞 부분만 사용
    """
    if raw_caption is None:
        return ""
    if not isinstance(raw_caption, str):
        try:
            raw_caption = str(raw_caption)
        except Exception:
            return ""
    s = raw_caption.strip()
    if "#" in s:
        s = s.split("#")[0].strip()
    return s


# ============================================================
# 2. Split / 데이터 존재 여부 스캔
# ============================================================

def read_ids_from_split_txt(split: str) -> List[str]:
    path = os.path.join(DATA_ROOT, f"{split}.txt")
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                ids.append(sid)
    return ids


def has_required_motion_files(sid: str) -> bool:
    base = os.path.join(SPLIT_JOINTS_DIR, sid)
    if not os.path.isdir(base):
        return False
    # holistic + parts
    req = ["holistic.npy"] + [f"{p}.npy" for p in PART_NAMES]
    for r in req:
        if not os.path.exists(os.path.join(base, r)):
            return False
    return True


def scan_valid_ids(split: str) -> List[str]:
    raw_ids = read_ids_from_split_txt(split)
    valid = []
    for sid in raw_ids:
        if has_required_motion_files(sid):
            valid.append(sid)
    return valid


# ============================================================
# 3. Caption 로더 (여기서 “전체 텍스트 확보”)
# ============================================================

def load_gar_json_as_id2caption(path: str) -> Dict[str, str]:
    """
    기존 gar_train.json / gar_val.json 같은 파일에서 caption을 최대한 가져오기.
    포맷:
      - list[{"id": "...", "caption": "..."}]
      - 또는 dict[id] = {...} 형태도 대응
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, str] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and "caption" in v:
                out[str(k)] = clean_caption(v.get("caption", ""))
        return out

    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and "id" in it:
                sid = str(it["id"])
                out[sid] = clean_caption(it.get("caption", ""))
    return out


def try_load_caption_from_text_dir(sid: str) -> str:
    """
    texts 폴더에서 caption을 찾는 여러 후보를 시도.
    너 환경이 확실치 않아서, 흔한 케이스들을 유연하게 지원.
    """
    # (A) texts/{id}.txt : 첫 줄을 caption으로
    cand_txt = os.path.join(TEXT_DIR, f"{sid}.txt")
    if os.path.exists(cand_txt):
        try:
            with open(cand_txt, "r", encoding="utf-8") as f:
                line = f.readline()
            return clean_caption(line)
        except Exception:
            pass

    # (B) texts/{id}.json : {"caption": "..."} 혹은 {"text": "..."}
    cand_json = os.path.join(TEXT_DIR, f"{sid}.json")
    if os.path.exists(cand_json):
        try:
            with open(cand_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                if "caption" in obj:
                    return clean_caption(obj.get("caption", ""))
                if "text" in obj:
                    return clean_caption(obj.get("text", ""))
        except Exception:
            pass

    # (C) texts/captions.json : dict[id]=caption 또는 list
    cand_caps = os.path.join(TEXT_DIR, "captions.json")
    if os.path.exists(cand_caps):
        try:
            with open(cand_caps, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and sid in obj:
                return clean_caption(obj.get(sid, ""))
            if isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict) and str(it.get("id", "")) == sid:
                        return clean_caption(it.get("caption", ""))
        except Exception:
            pass

    return ""


def build_id2caption_for_split(
    split: str,
    valid_ids: List[str],
    fallback_gar_json_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    valid_ids 전체에 대해 caption을 최대한 채운다.
    우선순위:
      1) 기존 gar json에 있는 caption
      2) texts 폴더에서 찾아오기
    """
    id2cap_from_gar = load_gar_json_as_id2caption(fallback_gar_json_path) if fallback_gar_json_path else {}

    id2cap: Dict[str, str] = {}
    missing = 0

    for sid in valid_ids:
        cap = ""
        if sid in id2cap_from_gar and id2cap_from_gar[sid]:
            cap = id2cap_from_gar[sid]
        else:
            cap = try_load_caption_from_text_dir(sid)

        cap = clean_caption(cap)
        if not cap:
            missing += 1
            if SKIP_IF_NO_CAPTION:
                continue

        id2cap[sid] = cap

    print(f"[INFO] Caption coverage ({split}): kept={len(id2cap)}, missing={missing}, valid_ids={len(valid_ids)}")
    return id2cap


# ============================================================
# 4. Mistral 로딩
# ============================================================

print(f"[INFO] Loading Mistral model: {MISTRAL_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MISTRAL_MODEL_NAME,
    use_fast=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # decoder-only 안정

model = AutoModelForCausalLM.from_pretrained(
    MISTRAL_MODEL_NAME,
    torch_dtype=DTYPE,
    device_map={"": "cuda"},
    low_cpu_mem_usage=True,
)
model.eval()
print("[INFO] Mistral loaded.")

DUMMY_PARTS = {p: "undefined" for p in PART_NAMES}


# ============================================================
# 5. 프롬프트 / JSON 파싱
# ============================================================

def build_prompt(caption: str) -> str:
    system_msg = (
        "You are an assistant that ONLY outputs valid JSON.\n"
        "You analyze a short description of a human action and write separate descriptions "
        "for each body part in English.\n"
        "Body parts: right_arm, left_arm, right_leg, left_leg, torso.\n"
        "Each value must be a short natural language phrase (at most 20 words).\n"
        "If a body part is not clearly mentioned, write \"undefined\" for that part.\n"
        "IMPORTANT RULES:\n"
        "1. Output ONLY a single JSON object.\n"
        "2. DO NOT add any explanation, markdown, comments, or extra text.\n"
        "3. DO NOT wrap the JSON in backticks.\n"
        "4. NEVER use \"...\" as a value.\n"
        "5. Keys must be exactly: right_arm, left_arm, right_leg, left_leg, torso.\n"
    )

    user_msg = (
        f"Caption: {caption}\n\n"
        "Return a JSON object exactly in this structure, but with real descriptions:\n"
        "{\n"
        "  \"right_arm\": \"...\",\n"
        "  \"left_arm\": \"...\",\n"
        "  \"right_leg\": \"...\",\n"
        "  \"left_leg\": \"...\",\n"
        "  \"torso\": \"...\"\n"
        "}\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def extract_json_from_text(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    text = text.replace("```json", "```").replace("```JSON", "```")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    json_str = text[start:end + 1]
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object.")
    return data


# ============================================================
# 6. (중요) 배치 호출: prompt_len 버그 수정 버전
# ============================================================

def _generate_batch(
    prompts: List[str],
    temperature: float,
    do_sample: bool,
) -> List[str]:
    """
    return: 각 샘플의 completion text (프롬프트 이후 부분만)
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    # 샘플별 실제 prompt 길이 (left padding 포함이므로 sum이 정답)
    prompt_lens = attn_mask.sum(dim=1).tolist()  # List[int]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=TOP_P,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 샘플별로 prompt 끝 이후만 잘라서 decode (✅ 버그 수정 핵심)
    texts: List[str] = []
    for i in range(output_ids.size(0)):
        pl = int(prompt_lens[i])
        gen_ids_i = output_ids[i, pl:]  # 각자 prompt 길이로 자름
        txt = tokenizer.decode(gen_ids_i, skip_special_tokens=True)
        texts.append(txt)
    return texts


def call_llm_for_parts_batch(captions: List[str], global_start_idx: int = 0) -> List[Dict[str, str]]:
    prompts = [build_prompt(c) for c in captions]

    texts = _generate_batch(
        prompts,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
    )

    results: List[Dict[str, str]] = []

    for i, resp_text in enumerate(texts):
        idx = global_start_idx + i

        if idx < 2:
            print("\n[RAW LLM OUTPUT]")
            print(resp_text)
            print("==============")

        part_desc: Dict[str, Any]
        ok = False
        try:
            part_desc = extract_json_from_text(resp_text)
            ok = True
        except Exception as e:
            part_desc = {}
            if ENABLE_RETRY:
                # 재시도: 온도 조금 올리고 샘플링 허용
                retry_text = _generate_batch(
                    [prompts[i]],
                    temperature=RETRY_TEMPERATURE,
                    do_sample=RETRY_DO_SAMPLE,
                )[0]
                try:
                    part_desc = extract_json_from_text(retry_text)
                    ok = True
                except Exception as e2:
                    print(f"[WARN] JSON parse failed (idx={idx}) retry failed. err1={e} err2={e2}")
                    part_desc = dict(DUMMY_PARTS)

            else:
                print(f"[WARN] JSON parse failed (idx={idx}), use dummy. Error: {e}")
                part_desc = dict(DUMMY_PARTS)

        clean_desc: Dict[str, str] = {}
        for k in PART_NAMES:
            v = part_desc.get(k, "undefined")
            if not isinstance(v, str):
                try:
                    v = str(v)
                except Exception:
                    v = "undefined"
            v = v.strip() if isinstance(v, str) else "undefined"
            if v == "":
                v = "undefined"
            clean_desc[k] = v

        results.append(clean_desc)

        if idx < 2:
            print("[PARSED PARTS]")
            print(clean_desc, "ok=", ok)
            print("==============")

    return results


# ============================================================
# 7. 전체 split에 대해 “FULL GAR JSON” 만들기
# ============================================================

def load_existing_output_ids(output_path: str) -> set:
    if not (ENABLE_RESUME and os.path.exists(output_path)):
        return set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = set()
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and "id" in it:
                    ids.add(str(it["id"]))
        return ids
    except Exception:
        return set()


def save_json_list(path: str, data: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_full_gar_for_split(
    split: str,
    output_path: str,
    fallback_gar_json_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    save_every_batches: int = 20,
):
    """
    split(train/val)에 대해:
      - split_joints에 존재하는 valid_ids 전부 대상
      - caption 확보
      - caption -> part_descriptions 생성
      - output: list[{"id","caption","part_descriptions"}]
    """
    valid_ids = scan_valid_ids(split)
    print(f"[INFO] Scanned valid ids ({split}): {len(valid_ids)}")

    id2cap = build_id2caption_for_split(split, valid_ids, fallback_gar_json_path=fallback_gar_json_path)

    # 대상 리스트
    ids = sorted(list(id2cap.keys()))
    if max_samples is not None:
        ids = ids[:max_samples]
        print(f"[INFO] Debug mode: only first {max_samples} ids will be processed.")

    # resume
    done_ids = load_existing_output_ids(output_path)
    if done_ids:
        print(f"[INFO] Resume: found {len(done_ids)} existing ids in {output_path}")

    todo_ids = [sid for sid in ids if sid not in done_ids]
    print(f"[INFO] To process ({split}): {len(todo_ids)} / {len(ids)}")

    # 기존 output 로드(있으면 이어쓰기)
    out_data: List[dict] = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, list):
                out_data = prev
        except Exception:
            out_data = []

    # 배치 처리
    batch_count = 0
    for start in tqdm(range(0, len(todo_ids), BATCH_SIZE), desc=f"Building FULL GAR ({split})"):
        end = min(start + BATCH_SIZE, len(todo_ids))
        batch_ids = todo_ids[start:end]
        captions = [id2cap[sid] for sid in batch_ids]

        parts_list = call_llm_for_parts_batch(captions, global_start_idx=start)

        for sid, cap, parts in zip(batch_ids, captions, parts_list):
            item = {
                "id": sid,
                "caption": cap,
                "part_descriptions": {p: parts.get(p, "undefined") for p in PART_NAMES},
            }
            out_data.append(item)

        batch_count += 1
        if save_every_batches > 0 and (batch_count % save_every_batches == 0):
            save_json_list(output_path, out_data)

    save_json_list(output_path, out_data)
    print(f"[INFO] Saved FULL GAR → {output_path}")
    print(f"[INFO] Total items in output: {len(out_data)}")


# ============================================================
# 8. main
# ============================================================

if __name__ == "__main__":
    # 네가 올린 기존 subset GAR json(32개짜리)도 “caption fallback”으로는 유용함
    gar_train_subset = os.path.join(DATA_ROOT, "gar_train.json")
    gar_val_subset = os.path.join(DATA_ROOT, "gar_val.json")

    # 최종 FULL output
    gar_train_full_out = os.path.join(DATA_ROOT, "texts", "gar_train_full_mistral.json")
    gar_val_full_out = os.path.join(DATA_ROOT, "texts", "gar_val_full_mistral.json")

    # 처음엔 32~128로 테스트 추천(진짜 JSON 잘 나오나 확인)
    # max_samples=None으로 두면 전량 생성
    build_full_gar_for_split(
        split="train",
        output_path=gar_train_full_out,
        fallback_gar_json_path=gar_train_subset,
        max_samples=None,
        save_every_batches=10,
    )
    build_full_gar_for_split(
        split="val",
        output_path=gar_val_full_out,
        fallback_gar_json_path=gar_val_subset,
        max_samples=None,
        save_every_batches=10,
    )
