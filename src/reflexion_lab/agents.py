from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector, get_last_call_metrics
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            actor_tokens, actor_latency = get_last_call_metrics()
            
            judge = evaluator(example, answer)
            evaluator_tokens, evaluator_latency = get_last_call_metrics()
            
            reflector_tokens, reflector_latency = 0, 0
            
            # TODO: Học viên triển khai logic Reflexion tại đây
            # 1. Kiểm tra nếu agent_type là 'reflexion' và chưa hết số lần attempt
            # 2. Gọi hàm reflector để lấy nội dung reflection
            # 3. Cập nhật reflection_memory để Actor dùng cho lần sau
            reflection_obj = None
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts and judge.score != 1:
                reflection_obj = reflector(example, attempt_id, judge)
                reflector_tokens, reflector_latency = get_last_call_metrics()
                reflections.append(reflection_obj)
                reflection_memory.append(
                    f"Attempt {attempt_id} failed.\n"
                    f"Failure reason: {reflection_obj.failure_reason}\n"
                    f"Lesson learned: {reflection_obj.lesson}\n"
                    f"Next strategy: {reflection_obj.next_strategy}"
                )
            
            token_estimate = actor_tokens + evaluator_tokens + reflector_tokens
            latency_ms = actor_latency + evaluator_latency + reflector_latency
            
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection_obj,
                token_estimate=token_estimate,
                latency_ms=latency_ms
            )
            
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
