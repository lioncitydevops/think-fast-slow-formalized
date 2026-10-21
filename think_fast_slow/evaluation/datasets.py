import random
from typing import List, Tuple, Dict, Any
from ..core.types import Query
from ..core.constants import TASK_BOOLQ, TASK_GSM8K, TASK_CRT

def generate_benchmark_datasets() -> Dict[str, Tuple[List[Query], List[Query]]]:
    """
    Generate evaluation benchmarks conforming to Section 8:
    - T1: BoolQ (yes/no natural questions, Jev strength)
    - T2: GSM8K (recast as 4-option choice with pre-fixed distractors, Jev weakness)
    - T3: CRT-style lure items (intuitive lure is wrong, tests WYSIATI and inverse scaling)
    Returns {task_name: (calibration_30_percent, test_70_percent)}
    """
    random.seed(42)

    # --- T1: BoolQ Items (Yes / No) ---
    t1_templates = [
        ("Is a tomato botanically a fruit?", "Yes", ["Yes", "No"]),
        ("Does the United States have an official national language?", "No", ["Yes", "No"]),
        ("Can penguins fly in the air?", "No", ["Yes", "No"]),
        ("Is water composed of hydrogen and oxygen?", "Yes", ["Yes", "No"]),
        ("Did Shakespeare write Hamlet?", "Yes", ["Yes", "No"]),
        ("Is the Sahara the largest desert on Earth?", "No", ["Yes", "No"]), # Antarctic is largest
        ("Do sharks have bones made of cartilage?", "Yes", ["Yes", "No"]),
        ("Is Mount Everest the tallest mountain above sea level?", "Yes", ["Yes", "No"]),
        ("Can sound travel through a vacuum in outer space?", "No", ["Yes", "No"]),
        ("Does an octopus have three hearts?", "Yes", ["Yes", "No"]),
    ]

    t1_queries = []
    for idx in range(120):
        tmpl, gold, opts = t1_templates[idx % len(t1_templates)]
        t1_queries.append(
            Query(
                id=f"boolq_{idx:03d}",
                text=f"Passage for item {idx}: {tmpl}",
                question="Is the statement true?",
                options=opts,
                gold_answer=gold,
                task=TASK_BOOLQ,
            )
        )

    # --- T2: GSM8K Items (4-choice math with distractors) ---
    t2_templates = [
        ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
         "72", ["72", "48", "24", "96"]),
        ("Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much did she earn?",
         "$60", ["$60", "$50", "$72", "$120"]),
        ("Betty picked 16 apples. A quarter of them were rotten. How many good apples does she have?",
         "12", ["12", "4", "16", "8"]),
        ("James bought 3 packs of stickers with 15 stickers each. He gave 10 stickers to his brother. How many does he have left?",
         "35", ["35", "45", "25", "30"]),
        ("A bakery makes 40 loaves of bread in the morning and 30 in the afternoon. If they sell 55 loaves, how many are left?",
         "15", ["15", "25", "70", "10"]),
    ]

    t2_queries = []
    for idx in range(100):
        problem, gold, opts = t2_templates[idx % len(t2_templates)]
        t2_queries.append(
            Query(
                id=f"gsm8k_{idx:03d}",
                text=f"Math Problem: {problem}",
                question="Select the correct numerical answer:",
                options=opts,
                gold_answer=gold,
                task=TASK_GSM8K,
                metadata={"is_math": True},
            )
        )

    # --- T3: CRT-style Lures & Adversarial Items ---
    t3_items = [
        # Bat and ball isomorph
        ("A notebook and an eraser cost $1.10 in total. The notebook costs $1.00 more than the eraser. How much does the eraser cost?",
         "$0.05", ["$0.10", "$0.05", "$1.00", "$0.55"]),
        # Lily pads isomorph
        ("In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
         "47 days", ["24 days", "47 days", "12 days", "48 days"]),
        # Machine widgets isomorph
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
         "5 minutes", ["100 minutes", "5 minutes", "1 minute", "20 minutes"]),
        # Hospital births isomorph
        ("A town has two hospitals: a large one with 45 births a day and a small one with 15. About 50% of babies are boys. Over a year, which hospital recorded more days on which over 60% of babies were boys?",
         "The small hospital", ["The large hospital", "The small hospital", "About the same", "Neither"]),
        # Reverse syllogism
        ("All roses are flowers. Some flowers fade quickly. Therefore, do some roses necessarily fade quickly?",
         "No", ["Yes", "No"]),
    ]

    t3_queries = []
    for idx in range(80):
        prompt, gold, opts = t3_items[idx % len(t3_items)]
        # First option is the intuitive lure answer
        lure_ans = opts[0] if opts[0] != gold else opts[1]
        t3_queries.append(
            Query(
                id=f"crt_{idx:03d}",
                text=f"Cognitive Reasoning Puzzle: {prompt}",
                question="Choose the logically correct solution:",
                options=opts,
                gold_answer=gold,
                task=TASK_CRT,
                metadata={"is_lure": True, "intuitive_lure": lure_ans},
            )
        )

    def split_30_70(dataset: List[Query]) -> Tuple[List[Query], List[Query]]:
        n_cal = int(len(dataset) * 0.30)
        cal = dataset[:n_cal]
        test = dataset[n_cal:]
        return cal, test

    return {
        TASK_BOOLQ: split_30_70(t1_queries),
        TASK_GSM8K: split_30_70(t2_queries),
        TASK_CRT: split_30_70(t3_queries),
    }
