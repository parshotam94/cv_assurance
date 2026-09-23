"""
Dedicated Backdoor Trigger Pattern Generator
"""
from experiments.generate_poisoned_data import generate_poisoned_experiment_dataset

def main():
    res = generate_poisoned_experiment_dataset()
    print("Backdoor data generated:", res["attacks"])

if __name__ == "__main__":
    main()
