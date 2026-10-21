import argparse
import sys
import json
from .core.types import Query
from .system1.jev_client import JevClient
from .system2.qwen_client import QwenClient
from .patterns.controller import DualProcessController
from .patterns import (
    create_p1_cascade,
    create_p2_budgeted,
    create_p3_probe,
    create_p4_verifier,
    create_p5_reconcile,
)
from .evaluation.harness import EvaluationHarness
from .evaluation.replications import replicate_table_3, replicate_table_5
from .domains.applications import ALL_DOMAINS

def main():
    parser = argparse.ArgumentParser(
        description="Thinking, Fast and Slow, Formalized: Jev (System 1) + Qwen3-8B (System 2) CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a query through the dual-process controller")
    run_parser.add_argument("--text", type=str, required=True, help="Input text state x")
    run_parser.add_argument("--question", type=str, required=True, help="Question instruction")
    run_parser.add_argument("--options", type=str, required=True, help="Comma-separated option choices")
    run_parser.add_argument("--pattern", type=str, default="P3", choices=["P1", "P2", "P3", "P4", "P5"], help="Pattern P1-P5")
    run_parser.add_argument("--tau", type=float, default=0.88, help="Certified act threshold tau")
    run_parser.add_argument("--mock", action="store_true", help="Use calibrated mock mode instead of live APIs")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run Section 8 Pre-Registered Evaluation Protocol")
    eval_parser.add_argument("--mock", action="store_true", default=True, help="Use mock mode for fast benchmark")
    eval_parser.add_argument("--samples", type=int, default=20, help="Max samples per task split")

    # Command: replicate
    rep_parser = subparsers.add_parser("replicate", help="Replicate Table 6 and Table 9 from the paper")
    rep_parser.add_argument("--table", type=str, default="all", choices=["6", "9", "3", "5", "all"])

    # Command: server
    srv_parser = subparsers.add_parser("server", help="Launch interactive Web Dashboard")
    srv_parser.add_argument("--host", type=str, default="127.0.0.1")
    srv_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "run":
        options = [opt.strip() for opt in args.options.split(",")]
        query = Query(
            id="cli_query",
            text=args.text,
            question=args.question,
            options=options,
        )
        jev = JevClient(use_mock=args.mock)
        qwen = QwenClient(use_mock=args.mock)

        if args.pattern == "P1":
            controller = create_p1_cascade(jev, qwen, tau=args.tau)
        elif args.pattern == "P2":
            controller = create_p2_budgeted(jev, qwen, tau=args.tau)
        elif args.pattern == "P3":
            controller = create_p3_probe(jev, qwen, tau=args.tau)
        elif args.pattern == "P4":
            controller = create_p4_verifier(jev, qwen, tau=args.tau)
        else:
            controller = create_p5_reconcile(jev, qwen, tau=args.tau)

        result = controller.execute(query)
        print("\n" + "=" * 60)
        print(f"DUAL-PROCESS DECISION (Pattern {args.pattern})")
        print("=" * 60)
        print(f"Action Released : {result.action}")
        print(f"Decision Band   : {result.band.value}")
        print(f"Execution Mode  : {result.mode}")
        print(f"Total Tokens    : {result.tokens_spent}")
        print(f"Latency         : {result.total_latency:.2f} s")
        print("-" * 60)
        print(f"System 1 (Jev)  : Choice = {result.jev.action} (Confidence = {result.jev.confidence:.2f})")
        print(f"Distribution    : {result.jev.probabilities}")
        if result.qwen:
            print(f"System 2 (Qwen) : Choice = {result.qwen.action} (Mode = {result.qwen.mode})")
            if result.qwen.reasoning_trace:
                print(f"Thinking Trace  : {result.qwen.reasoning_trace[:200]}...")
        if result.conformal_set:
            print(f"Conformal Set   : {result.conformal_set}")
        print("=" * 60 + "\n")

    elif args.command == "evaluate":
        harness = EvaluationHarness(use_mock=args.mock)
        print(f"Running Pre-Registered Evaluation Suite (mock={args.mock}, samples={args.samples})...")
        results = harness.run_full_suite(max_samples_per_task=args.samples)

        print("\n" + "=" * 70)
        print("PRE-REGISTERED HYPOTHESES VERIFICATION (Table 6)")
        print("=" * 70)
        for h in results["hypotheses"]:
            status_symbol = "[PASS]" if not h["falsified"] else "[FAIL]"
            print(f"{status_symbol} {h['id']}: {h['description']}")
            print(f"       Metric: {h['metric_value']}")
        print("\n" + "=" * 70)
        print("MATHEMATICAL IDENTITIES VERIFICATION")
        print("=" * 70)
        esc = results["escalation_identity"]
        print(f"Prop 5.1 Escalation Identity : G(Z) = M_Z*(W2-W1)")
        print(f"  Holds to Machine Precision : {esc['identity_holds']} (Diff = {esc['diff_machine_precision']})")
        print(f"  Disagreement Mass M_Z      : {esc['M_Z']:.3f} (observable without labels!)")
        print(f"  Win Margin (W2 - W1)       : {esc['win_margin']:.3f}")
        decomp = results["paired_decomposition"]
        print(f"\nProp 5.4 Paired Decomposition: G_inf - G_blind = sum P(T)*diff")
        print(f"  Holds to Machine Precision : {decomp['identity_holds']} (Diff = {decomp['diff_machine_precision']})")
        print(f"  Stable Contribution is 0   : {decomp['stable_contribution_zero']}")
        print(f"  Transition Shares          : Anchor={decomp['shares']['anchor']:.2f}, Reactance={decomp['shares']['reactance']:.2f}, Drift={decomp['shares']['drift']:.2f}")
        print("=" * 70 + "\n")

    elif args.command == "replicate":
        if args.table in ["6", "3", "all"]:
            t6 = replicate_table_3()
            print("\n" + "=" * 65)
            print("TABLE 6 REPLICATION: Optimal Budgets Under Five Return Curves")
            print("=" * 65)
            print(f"{'Curve':<15} {'Act c>=':<10} {'Give up c<=':<14} {'Peak B*':<10} {'Min B*>0':<10} {'Kept':<6}")
            print("-" * 65)
            for row in t6:
                print(f"{row['curve']:<15} {row['act_c']:<10.2f} {row['give_up_c']:<14.2f} {row['peak_B']:<10} {row['min_B_pos']:<10} {int(row['kept']*100)}%")
            print("=" * 65)

        if args.table in ["9", "5", "all"]:
            t9 = replicate_table_5(n_samples=10000)
            print("\n" + "=" * 75)
            print("TABLE 9 REPLICATION: Worked Example Under Three Prices of Thought")
            print("=" * 75)
            print(f"{'Policy':<25} {'Act':<8} {'Deliberate':<12} {'Give up':<10} {'Tokens':<8} {'Accuracy':<8}")
            print("-" * 75)
            for row in t9:
                print(f"{row['policy']:<25} {row['act_share']:<8} {row['deliberate_share']:<12} {row['give_up_share']:<10} {row['tokens']:<8} {row['accuracy']:<8.3f}")
            print("=" * 75 + "\n")

    elif args.command == "server":
        import uvicorn
        from .web.server import app
        print(f"Starting Think Fast Slow Web Dashboard on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
