from llm.evaluator import run_benchmark

if __name__ == "__main__":
    # Test 1 — nos histoires manuelles
    # print("=== Manual stories ===")
    # run_benchmark(
    #     max_stories   = 10,
    #     use_tomi      = False,
    #     delay_seconds = 0.5,
    #     save_path     = "results_manual.json"
    # )

    # Test 2 — ToMi dataset réel
    print("\n=== ToMi dataset ===")
    run_benchmark(
        max_stories   = 50,
        use_tomi      = True,
        delay_seconds = 0.5,
        save_path     = "results_tomi.json"
    )