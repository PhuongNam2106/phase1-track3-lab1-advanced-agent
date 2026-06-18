from __future__ import annotations
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

# Load environment variables
load_dotenv()

# Check if we should use mock mode
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true" or not os.getenv("OPENAI_API_KEY")

_last_call_tokens = 0
_last_call_latency_ms = 0

def get_last_call_metrics() -> tuple[int, int]:
    """Retrieve the token count and latency of the last LLM/Mock API call."""
    return _last_call_tokens, _last_call_latency_ms

_client = None
def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    global _last_call_tokens, _last_call_latency_ms
    
    if MOCK_MODE:
        # Hardcoded mock stats
        _last_call_tokens = 320 + (attempt_id * 65) + (120 if agent_type == "reflexion" else 0)
        _last_call_latency_ms = 160 + (attempt_id * 40) + (90 if agent_type == "reflexion" else 0)
        
        if example.qid not in FIRST_ATTEMPT_WRONG:
            return example.gold_answer
        if agent_type == "react":
            return FIRST_ATTEMPT_WRONG[example.qid]
        if attempt_id == 1 and not reflection_memory:
            return FIRST_ATTEMPT_WRONG[example.qid]
        return example.gold_answer

    # Real LLM Call using gpt-4o-mini
    client = get_openai_client()
    
    context_str = ""
    for chunk in example.context:
        context_str += f"Title: {chunk.title}\nText: {chunk.text}\n\n"
        
    reflection_str = ""
    if reflection_memory:
        reflection_str = "\nFeedback/Reflection from previous failed attempt(s):\n"
        for i, ref in enumerate(reflection_memory, 1):
            reflection_str += f"{i}. {ref}\n"
            
    user_content = (
        f"Context:\n{context_str}\n"
        f"Question: {example.question}\n"
        f"{reflection_str}\n"
        f"Please think step-by-step and write down your reasoning. At the end, output your final answer clearly in the format: 'Final Answer: [Your Answer]'"
    )
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user_content}
        ],
        temperature=0.0
    )
    latency = int((time.perf_counter() - start_time) * 1000)
    tokens = response.usage.total_tokens if response.usage else 0
    
    _last_call_tokens = tokens
    _last_call_latency_ms = latency
    
    content = response.choices[0].message.content or ""
    
    # Try to extract the final answer
    if "Final Answer:" in content:
        final_answer = content.split("Final Answer:")[-1].strip()
    else:
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if lines:
            final_answer = lines[-1].replace("Final Answer:", "").strip()
        else:
            final_answer = content.strip()
            
    if final_answer.startswith("[") and final_answer.endswith("]"):
        final_answer = final_answer[1:-1].strip()
    return final_answer

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    global _last_call_tokens, _last_call_latency_ms
    
    if MOCK_MODE:
        _last_call_tokens = 0
        _last_call_latency_ms = 0
        
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        if normalize_answer(answer) == "london":
            return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

    # Real LLM Call using gpt-4o-mini
    client = get_openai_client()
    user_content = (
        f"Question: {example.question}\n"
        f"Ground-truth (gold) answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n"
    )
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    latency = int((time.perf_counter() - start_time) * 1000)
    tokens = response.usage.total_tokens if response.usage else 0
    
    _last_call_tokens = tokens
    _last_call_latency_ms = latency
    
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        score = int(data.get("score", 0))
        reason = data.get("reason", "No reason provided by Evaluator.")
        missing_evidence = data.get("missing_evidence", [])
        if isinstance(missing_evidence, str):
            missing_evidence = [missing_evidence]
        spurious_claims = data.get("spurious_claims", [])
        if isinstance(spurious_claims, str):
            spurious_claims = [spurious_claims]
            
        return JudgeResult(
            score=score,
            reason=reason,
            missing_evidence=missing_evidence,
            spurious_claims=spurious_claims
        )
    except Exception as e:
        is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        return JudgeResult(
            score=1 if is_correct else 0,
            reason=f"Failed to parse Evaluator JSON output: {str(e)}. Raw output: {content}",
            missing_evidence=[],
            spurious_claims=[]
        )

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    global _last_call_tokens, _last_call_latency_ms
    
    if MOCK_MODE:
        _last_call_tokens = 0
        _last_call_latency_ms = 0
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)

    # Real LLM Call using gpt-4o-mini
    client = get_openai_client()
    user_content = (
        f"Question: {example.question}\n"
        f"Evaluation Result (why it failed): {judge.reason}\n"
    )
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    latency = int((time.perf_counter() - start_time) * 1000)
    tokens = response.usage.total_tokens if response.usage else 0
    
    _last_call_tokens = tokens
    _last_call_latency_ms = latency
    
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=data.get("failure_reason", judge.reason),
            lesson=data.get("lesson", "Be more careful when answering multi-hop questions."),
            next_strategy=data.get("next_strategy", "Double check the second hop context.")
        )
    except Exception as e:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson=f"Failed to parse Reflector JSON output: {str(e)}. Raw output: {content}",
            next_strategy="Carefully check each hop in the context."
        )
