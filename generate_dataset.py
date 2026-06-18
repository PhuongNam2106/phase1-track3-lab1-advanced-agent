import json

# Generate 50 synthetic QA examples based on the template
examples = []
for i in range(1, 51):
    qid = f"hp_synth_{i}"
    # Alternate difficulties
    difficulty = "easy" if i % 3 == 0 else ("medium" if i % 3 == 1 else "hard")
    
    # We can use slightly varying questions
    question = f"What river flows through the city where Ada Lovelace clone {i} was born?"
    gold_answer = "River Thames"
    
    # Use standard context compatible with mock_runtime if needed,
    # or just generic text.
    context = [
        {
            "title": f"Ada Lovelace clone {i}",
            "text": f"Ada Lovelace clone {i} was born in London, England."
        },
        {
            "title": "London",
            "text": "London is crossed by the River Thames."
        }
    ]
    
    examples.append({
        "qid": qid,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": gold_answer,
        "context": context
    })

# Save to data/hotpot_synthetic.json
with open("data/hotpot_synthetic.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print("Generated 50 synthetic examples in data/hotpot_synthetic.json")
