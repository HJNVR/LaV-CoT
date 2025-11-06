
import re
import json

# format reward
def format_reward(completions, scale=0.25, num_tag_pairs=2, **kwargs):
    rewards = []
    for c in completions:
        reward = 0.0
        if '<think>' in c and '</think>' in c:
            reward += scale * 1 / num_tag_pairs
        if '<answer>' in c and '</answer>' in c:
            reward += scale * 1 / num_tag_pairs
        rewards.append(reward)
    print("Format rewards: ", rewards)
    return rewards

# count reward
def extract_json_objects(text):
    """提取文本中的 JSON 对象列表"""
    return re.findall(r'\{.*?\}', text, re.DOTALL)

def extract_obj_count(text):
    """提取文本中的 \obj{n} 数字值"""
    match = re.search(r'\\obj\{(\d+)\}', text)
    return int(match.group(1)) if match else 0

def count_json_keys(json_str):
    """计算 JSON 字符串中对象数量"""
    try:
        data = json.loads(json_str)
        return len(data)
    except json.JSONDecodeError:
        return 0

def compute_r_count(completions, gt_counts):
    """
    completions: list of strings (包含 JSON 和 \obj{})
    gt_counts: list of ints (ground truth count)
    
    返回 RCount 列表
    """
    r_counts = []
    for comp, gt in zip(completions, gt_counts, scale=0.25):
        # JSON 数组长度
        json_matches = extract_json_objects(comp)
        json_len = sum(count_json_keys(js) for js in json_matches)
        # \obj{} 数值
        obj_val = extract_obj_count(comp)
        pred_count = json_len + obj_val
        # RCount 公式
        r_count = 1 - abs(pred_count - gt) / gt
        r_counts.append(r_count * scale)
    return r_counts

# language reward
def lang_reward(completions, lang, scale=0.25, **kwargs):
    rewards = [scale if gt.lower() in r.lower() else 0.0 for r, gt in zip(completions, lang)]
    print("language_reward: ", rewards)
    return rewards

# edit_distance_reward
def edit_distance_reward(completions, answer, scale=0.25, **kwargs):
    def score(r, a):
        r = extract_final_answer(r)
        if not r:
            r = "" 
        d = Levenshtein.distance(r, a)
        raw_score = 1.0 - d / max(len(r), len(a)) if max(len(r), len(a)) else 1.0
        return raw_score * scale

    scores = [score(r, a) for r, a in zip(completions, answer)]
    print("edit_distance_reward: ", scores)
    return scores
