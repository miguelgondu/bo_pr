import json

from discrete_mixed_bo.run_one_replication import run_one_replication

from poli.repository import AlbuterolSimilarityBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver


class ProbabilisticReparametrizationSolver(AbstractSolver):
    def __init__(
        self,
        black_box,
        x0,
        y0,
        seed: int,
        max_iter: int,
        batch_size: int = 1,
        mc_samples: int = ...,
        n_initial_points: int = ...,
        sequence_length: int | None = ...,
        alphabet: list[str] | None = None,
    ):
        super().__init__(black_box, x0, y0)
        self.seed = seed
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.n_initial_points = n_initial_points

        sequence_length_ = sequence_length or self.black_box.info.max_sequence_length
        if sequence_length_ is None or sequence_length_ == float("inf"):
            raise ValueError("Sequence length must be provided.")
        self.sequence_length = sequence_length_

        alphabet_ = alphabet or self.black_box.info.alphabet
        if alphabet_ is None:
            raise ValueError("Alphabet must be provided.")

    def solve(self, max_iter: int):
        run_one_replication(
            seed=self.seed,
            label="pr__ei",
            iterations=max_iter,
            function_name="poli",
            batch_size=self.batch_size,
            mc_samples=self.mc_samples,
            n_initial_points=self.n_initial_points,
            problem_kwargs={
                "black_box": self.black_box,
                "sequence_length": self.sequence_length,
                "alphabet": self.alphabet,
            },
            save_callback=lambda t: t,
        )


if __name__ == "__main__":
    f = AlbuterolSimilarityBlackBox(string_representation="SELFIES")
    with open(
        "/Users/sjt972/Projects/high_dimensional_bo_benchmark/data/small_molecule_datasets/processed/zinc250k_alphabet_stoi.json"
    ) as fp:
        alphabet = json.load(fp)

    alphabet = list(alphabet.keys())
