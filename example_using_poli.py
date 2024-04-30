import json

from discrete_mixed_bo.run_one_replication import run_one_replication

from poli.repository import AlbuterolSimilarityBlackBox


if __name__ == "__main__":
    f = AlbuterolSimilarityBlackBox(string_representation="SELFIES")
    with open(
        "/Users/sjt972/Projects/high_dimensional_bo_benchmark/data/small_molecule_datasets/processed/zinc250k_alphabet_stoi.json"
    ) as fp:
        alphabet = json.load(fp)

    alphabet = list(alphabet.keys())
    run_one_replication(
        seed=0,
        label="pr__ei",
        iterations=10,
        function_name="poli",
        batch_size=1,
        mc_samples=5,
        n_initial_points=10,
        problem_kwargs={
            "black_box": f,
            "sequence_length": 64,
            "alphabet": alphabet,
        },
        save_callback=lambda t: t,
    )
