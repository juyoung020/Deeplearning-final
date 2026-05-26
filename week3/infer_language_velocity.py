from __future__ import annotations

import argparse
import json

from language_velocity_model import (
    LanguageVelocityPredictor,
    default_velocity_examples,
    format_velocity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Go2 velocity from a locomotion text command.")
    parser.add_argument("--checkpoint", default="checkpoints/language_velocity_mlp.pt", help="Trained MLP checkpoint.")
    parser.add_argument("--text", default=None, help='Command text, e.g. "move forward 50cm".')
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--interactive", action="store_true", help="Prompt repeatedly for command text.")
    parser.add_argument("--all", action="store_true", help="Run inference on all 11 default commands.")
    parser.add_argument("--output", default=None, help="Save predictions to JSON file (used with --all).")
    args = parser.parse_args()

    predictor = LanguageVelocityPredictor(args.checkpoint, device=args.device)

    if args.all:
        examples = default_velocity_examples()
        texts = [e.text for e in examples]
        predictions = predictor.predict_batch(texts)
        rows = []
        print(f"{'Command':<25} {'Target':<30} {'Predicted':<30}")
        print("-" * 85)
        for example, pred in zip(examples, predictions):
            target_str = format_velocity(example.target)
            pred_str = format_velocity(pred)
            print(f"{example.text:<25} {target_str:<30} {pred_str:<30}")
            rows.append({"text": example.text, "target": list(example.target), "prediction": list(pred)})
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            print(f"\n[INFO] Saved to {args.output}")
        return

    if args.interactive or args.text is None:
        print('Enter command text. Examples: "move forward 25cm", "turn left 30 degree", "stop".')
        print('Type "quit" to exit.')
        while True:
            try:
                text = input("Command: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit"}:
                break
            velocity = predictor.predict(text)
            print(f"predicted_velocity={format_velocity(velocity)}")
        return

    velocity = predictor.predict(args.text)
    print(f"text={args.text!r}")
    print(f"predicted_velocity={format_velocity(velocity)}")


if __name__ == "__main__":
    main()
