"""
Golden Dataset for RAG Testing
Comprehensive test cases covering multiple scenarios
"""

from rag_pipeline import rag_chain
import json
from datetime import datetime

# Golden test dataset
golden_dataset = [
    # ========== ANSWERABLE - SIMPLE FACTS (8 questions) ==========
    {
        "id": 1,
        "question": "What is the refund policy?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["30 days", "refund", "purchase"]
    },
    {
        "id": 2,
        "question": "How many days of annual leave do employees get?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["20 days", "annual leave"]
    },
    {
        "id": 3,
        "question": "What happens in week 1 of onboarding?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["culture", "tools", "team"]
    },
    {
        "id": 4,
        "question": "Can I get a refund on a digital product?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["non-refundable", "digital", "downloaded"]
    },
    {
        "id": 5,
        "question": "How do I request a refund?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["email", "support@company.com", "order number"]
    },
    {
        "id": 6,
        "question": "How long does refund processing take?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["5 business days"]
    },
    {
        "id": 7,
        "question": "What email should I contact for refunds?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["support@company.com"]
    },
    {
        "id": 8,
        "question": "Are digital products refundable after download?",
        "category": "simple_fact",
        "expected_type": "answerable",
        "keywords": ["non-refundable", "downloaded"]
    },
    
    # ========== ANSWERABLE - REPHRASED (4 questions) ==========
    {
        "id": 9,
        "question": "What is the time limit for getting my money back?",
        "category": "rephrased",
        "expected_type": "answerable",
        "keywords": ["30 days"]
    },
    {
        "id": 10,
        "question": "How many vacation days do full-time staff receive?",
        "category": "rephrased",
        "expected_type": "answerable",
        "keywords": ["20 days"]
    },
    {
        "id": 11,
        "question": "What's covered in the first week for new hires?",
        "category": "rephrased",
        "expected_type": "answerable",
        "keywords": ["culture", "tools", "team"]
    },
    {
        "id": 12,
        "question": "If I downloaded a digital item, can I return it?",
        "category": "rephrased",
        "expected_type": "answerable",
        "keywords": ["non-refundable"]
    },
    
    # ========== ANSWERABLE - MULTI-HOP (3 questions) ==========
    {
        "id": 13,
        "question": "If I buy a digital product and change my mind, what should I do?",
        "category": "multi_hop",
        "expected_type": "answerable",
        "keywords": ["non-refundable", "downloaded"]
    },
    {
        "id": 14,
        "question": "I want a refund - how do I do it and how long will it take?",
        "category": "multi_hop",
        "expected_type": "answerable",
        "keywords": ["email", "support@company.com", "5 business days"]
    },
    {
        "id": 15,
        "question": "What happens during onboarding and how many leave days do I get?",
        "category": "multi_hop",
        "expected_type": "answerable",
        "keywords": ["onboarding", "20 days", "annual leave"]
    },
    
    # ========== UNANSWERABLE - NOT IN DOCS (5 questions) ==========
    {
        "id": 16,
        "question": "What is the CEO's name?",
        "category": "unanswerable",
        "expected_type": "unanswerable",
        "keywords": ["don't know", "not available", "not in"]
    },
    {
        "id": 17,
        "question": "What is our stock price?",
        "category": "unanswerable",
        "expected_type": "unanswerable",
        "keywords": ["don't know", "not available", "not in"]
    },
    {
        "id": 18,
        "question": "How many employees work here?",
        "category": "unanswerable",
        "expected_type": "unanswerable",
        "keywords": ["don't know", "not available", "not in"]
    },
    {
        "id": 19,
        "question": "What is the company's annual revenue?",
        "category": "unanswerable",
        "expected_type": "unanswerable",
        "keywords": ["don't know", "not available", "not in"]
    },
    {
        "id": 20,
        "question": "Who is the head of HR?",
        "category": "unanswerable",
        "expected_type": "unanswerable",
        "keywords": ["don't know", "not available", "not in"]
    },
]

def evaluate_answer(question_data, answer):
    """
    Evaluate if answer is correct based on expected type and keywords
    """
    expected_type = question_data["expected_type"]
    keywords = question_data["keywords"]
    answer_lower = answer.lower()
    
    if expected_type == "answerable":
        # Check if answer contains expected keywords
        keyword_found = any(kw.lower() in answer_lower for kw in keywords)
        # Check if it didn't abstain
        abstained = any(phrase in answer_lower for phrase in [
            "don't know", "not in", "not available", "cannot find"
        ])
        
        if abstained:
            return "FAIL", "Incorrectly abstained on answerable question"
        elif keyword_found:
            return "PASS", "Correct answer with expected content"
        else:
            return "PARTIAL", "Answered but missing expected keywords"
    
    else:  # unanswerable
        abstained = any(phrase in answer_lower for phrase in [
            "don't know", "not in", "not available", "cannot find", 
            "no information", "not mentioned"
        ])
        
        if abstained:
            return "PASS", "Correctly abstained"
        else:
            return "FAIL", "Hallucinated answer for unanswerable question"

def run_evaluation():
    """
    Run complete evaluation and generate report
    """
    print("="*60)
    print("GOLDEN DATASET EVALUATION")
    print(f"Total questions: {len(golden_dataset)}")
    print("="*60 + "\n")
    
    results = []
    category_stats = {}
    
    for test_case in golden_dataset:
        qid = test_case["id"]
        question = test_case["question"]
        category = test_case["category"]
        
        print(f"\n[{qid}/20] {question}")
        print(f"Category: {category} | Expected: {test_case['expected_type']}")
        
        # Get answer from RAG
        answer = rag_chain.invoke(question)
        
        # Evaluate
        status, reason = evaluate_answer(test_case, answer)
        
        # Store result
        result = {
            "id": qid,
            "question": question,
            "category": category,
            "expected": test_case["expected_type"],
            "answer": answer,
            "status": status,
            "reason": reason
        }
        results.append(result)
        
        # Update category stats
        if category not in category_stats:
            category_stats[category] = {"total": 0, "pass": 0, "fail": 0, "partial": 0}
        category_stats[category]["total"] += 1
        if status == "PASS":
            category_stats[category]["pass"] += 1
        elif status == "FAIL":
            category_stats[category]["fail"] += 1
        else:
            category_stats[category]["partial"] += 1
        
        # Print result
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_symbol} {status}: {reason}")
        print(f"Answer: {answer[:100]}...")
        print("-"*60)
    
    # Calculate overall metrics
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions: {total}")
    print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"⚠️  Partial: {partial} ({partial/total*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"\nOverall Accuracy: {(passed + partial*0.5)/total*100:.1f}%")
    
    print("\n" + "="*60)
    print("CATEGORY BREAKDOWN")
    print("="*60)
    for category, stats in category_stats.items():
        accuracy = (stats["pass"] / stats["total"]) * 100
        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  ✅ Pass: {stats['pass']}/{stats['total']} ({accuracy:.1f}%)")
        if stats["partial"] > 0:
            print(f"  ⚠️  Partial: {stats['partial']}/{stats['total']}")
        if stats["fail"] > 0:
            print(f"  ❌ Fail: {stats['fail']}/{stats['total']}")
    
    # Save results to file
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "partial": partial,
        "overall_accuracy": f"{(passed + partial*0.5)/total*100:.1f}%",
        "category_stats": category_stats,
        "detailed_results": results
    }
    
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Full report saved to: evaluation_report.json")
    print("="*60)
    
    return results

if __name__ == "__main__":
    run_evaluation()