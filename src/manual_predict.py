"""
Manual prediction tool. No internet, no live API needed.
Just type in the batsman, bowler, over number, and match situation, and get
a prediction with commentary insights - fully offline, using your trained models.

Run:
    python3 src/manual_predict.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import NextOverPredictor


def ask(prompt, cast=str, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    if val == "" and default is not None:
        return default
    try:
        return cast(val)
    except ValueError:
        print("Invalid input, try again.")
        return ask(prompt, cast, default)


def main():
    print("=" * 50)
    print("Next-Over Predictor — Manual Entry (fully offline)")
    print("=" * 50)
    predictor = NextOverPredictor()

    while True:
        print("\nEnter the details for the upcoming over:")
        batsman = ask("Batsman name (exact spelling as in your data)")
        bowler = ask("Bowler name (exact spelling as in your data)")
        over_num = ask("Over number (1-20)", int, 1)
        score_before = ask("Team score before this over", int, 0)
        wkts_down = ask("Wickets down before this over", int, 0)
        balls_faced = ask("Balls faced so far by this batsman in this innings", int, 0)
        pitch_type = ask(
            "Pitch type (batting_paradise / balanced / slow_turner / seaming_track)",
            str,
            "balanced",
        )

        try:
            result = predictor.predict(
                batsman=batsman,
                bowler=bowler,
                over_num=over_num,
                score_before=score_before,
                wkts_down=wkts_down,
                balls_faced_by_batsman=balls_faced,
                pitch_type=pitch_type,
            )
        except Exception as e:
            print(f"\nSomething went wrong: {e}")
            print(
                "Double-check the names match exactly how they appear in your training data."
            )
            continue

        print("\n" + "-" * 50)
        print(f"PREDICTION: {batsman} facing {bowler}, over {over_num}")
        print("-" * 50)
        print(
            f"Expected runs this over: {result['expected_runs']} (range {result['expected_range']})"
        )
        print(f"Wicket probability: {result['wicket_probability']}%")

        conf = result["data_confidence"]
        if conf["limited_data"]:
            print(f"\n⚠ {conf['note']}")

        print("\nCommentary insights:")
        for line in result["commentary_lines"]:
            print(f"  - {line}")
        print("-" * 50)

        again = input("\nPredict another over? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
