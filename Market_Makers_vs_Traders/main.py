"""
main.py — CLI entry point for the Glosten-Milgrom simulator.
Run: python main.py
"""

import argparse
from src.simulation import Simulation, BatchExperiment


def run_single(args):
    print(f"\n{'='*60}")
    print("  GLOSTEN-MILGROM MARKET MICROSTRUCTURE SIMULATOR")
    print(f"{'='*60}")
    print(f"  V_L={args.v_low}  V_H={args.v_high}  μ={args.mu}  rounds={args.rounds}")
    print(f"  Informed strategy : {args.informed_strategy}")
    print(f"  Noise strategy    : {args.noise_strategy}")
    print(f"{'='*60}\n")

    sim = Simulation(
        v_low             = args.v_low,
        v_high            = args.v_high,
        mu                = args.mu,
        informed_strategy = args.informed_strategy,
        noise_strategy    = args.noise_strategy,
        n_rounds          = args.rounds,
        seed              = args.seed,
    )
    sim.run()
    s = sim.summary()

    print(f"  True asset value  : {s['true_value']:.2f}")
    print(f"  Final MM belief   : P(V_H) = {s['final_belief_high']:.4f}")
    print(f"  Final mid price   : {s['final_mid']:.4f}")
    print(f"  Avg spread        : {s['avg_spread']:.4f}")
    print(f"  Final spread      : {s['final_spread']:.4f}\n")
    print(f"  {'Player':<22} {'Total P&L':>10}  {'Trades':>8}")
    print(f"  {'-'*44}")
    print(f"  {'Market Maker':<22} {s['mm_total_pnl']:>10.4f}")
    print(f"  {'Informed Trader':<22} {s['it_total_pnl']:>10.4f}  {s['n_informed_trades']:>8}")
    print(f"  {'Noise Trader':<22} {s['nt_total_pnl']:>10.4f}  {s['n_noise_trades']:>8}\n")

    d = s['spread_decomp']
    print(f"  Spread decomposition (at avg belief, not terminal):")
    print(f"    Total spread      : {d['total_spread']:.4f}")
    print(f"    Adverse selection : {d['adverse_selection']:.4f}")
    print(f"    Order processing  : {d['order_processing']:.4f}")
    print(f"    {d.get('note','')}\n")
    print(f"  ── Theoretical note ──────────────────────────────────")
    print(f"  The zero-profit condition E[V|buy]=ask, E[V|sell]=bid")
    print(f"  holds at EVERY quote (verified analytically).")
    print(f"  MM P&L in a single run reflects realised V (40 or 60),")
    print(f"  not the prior mixture. Averaged across many seeds,")
    print(f"  MM converges to ~0. This is correct GM behaviour.\n")

    if args.verbose:
        df = sim.trades_df()
        print(df[["round","trader_type","action","price","spread","belief_high",
                   "mm_pnl","it_pnl","nt_pnl"]].tail(20).to_string(index=False))


def run_sweep(args):
    import numpy as np
    print("\nRunning μ sweep...")
    results = BatchExperiment.sweep_mu(
        mu_values = list(np.linspace(0.0, 0.9, 10)),
        n_rounds  = 200,
        n_seeds   = 10,
    )
    print("\n  mu    avg_spread   it_pnl   nt_pnl")
    print("  " + "-"*40)
    for _, row in results.iterrows():
        print(f"  {row['mu']:.2f}   {row['avg_spread']:>8.4f}   {row['it_pnl']:>7.2f}   {row['nt_pnl']:>7.2f}")


def run_compare(args):
    print("\nComparing informed trader strategies...")
    results = BatchExperiment.strategy_comparison(n_rounds=100, n_seeds=20)
    print("\n  strategy      it_pnl   mm_pnl   nt_pnl   avg_spread   n_trades")
    print("  " + "-"*65)
    for _, row in results.iterrows():
        print(f"  {row['strategy']:<12}  {row['it_pnl']:>7.2f}  {row['mm_pnl']:>7.2f}  "
              f"{row['nt_pnl']:>7.2f}  {row['avg_spread']:>10.4f}  {row['n_it_trades']:>8.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glosten-Milgrom Market Microstructure Simulator")
    sub = parser.add_subparsers(dest="command")

    # Single simulation
    p_single = sub.add_parser("run", help="Run a single simulation")
    p_single.add_argument("--v-low",  type=float, default=40.0)
    p_single.add_argument("--v-high", type=float, default=60.0)
    p_single.add_argument("--mu",     type=float, default=0.3,
                          help="P(informed trader)")
    p_single.add_argument("--rounds", type=int,   default=100)
    p_single.add_argument("--informed-strategy", default="aggressive",
                          choices=["aggressive","mixed","patient"])
    p_single.add_argument("--noise-strategy",    default="random",
                          choices=["random","momentum","contrarian"])
    p_single.add_argument("--seed",    type=int,  default=42)
    p_single.add_argument("--verbose", action="store_true")

    # Mu sweep
    sub.add_parser("sweep", help="Sweep over μ values")

    # Strategy comparison
    sub.add_parser("compare", help="Compare informed trader strategies")

    args = parser.parse_args()

    if args.command == "run":
        run_single(args)
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "compare":
        run_compare(args)
    else:
        # Default: run a standard simulation
        import sys
        sys.argv = ["main.py", "run"]
        args = parser.parse_args()
        run_single(args)