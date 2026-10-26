"""Print a short example. Run: python demo.py"""
from pprint import pprint

from pipeline import FeedbackPipeline

# Four normalized force measurements per sample; these are made-up values.
SIMULATED_FORCES = [
    [0.0, 0.0, 0.0, 0.0],
    [0.2, 0.8, 0.5, 0.1],
    [0.6, 0.8, 0.7, 0.5],
    [0.0, 0.0, 0.0, 0.0],
]


def main() -> None:
    pipeline = FeedbackPipeline()
    for step, forces in enumerate(SIMULATED_FORCES):
        result = pipeline.step(forces, mode="all")
        print(f"\nStep {step} | normalized inputs: {forces}")
        pprint(result, sort_dicts=False)


if __name__ == "__main__":
    main()
