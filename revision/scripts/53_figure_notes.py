from __future__ import annotations

import sys
from pathlib import Path

from _common import load_config

# stem -> (status, comment, script, axes, what to look for)
NOTES: dict[str, tuple[str, str, str, str, str]] = {
    "revision_scdataset_fig6_drug": (
        "REPORTED", "R2 major 3", "20_train_convergence.py",
        "x: optimiser step (minibatch 4,096). y: training cross-entropy, drug task, MLP. "
        "Six arms. Red dotted verticals mark the 12 plate boundaries.",
        "This is scDataset's Figure 6, reproduced with annbatch added, restricted to the "
        "drug task because that is the discriminative one. Look for: the thick green dashed "
        "annbatch curve lying **on** the thick black uniform-random reference for the whole "
        "epoch. That is the claim. Then that scDataset at block 1024 (dark blue dashed) sits "
        "clearly above both, that sequential streaming (brown) and streaming with a "
        "16,384-row buffer (grey dashed) are violently erratic, and that the buffer barely "
        "helps. Note the excursions run in **both** directions: streaming's minimum loss is "
        "0.496 against annbatch's 2.125, because a sequential minibatch is essentially one "
        "compound and predicting it is trivial. What distinguishes the arms is the stability "
        "of the trajectory, not its lowest point."),
    "revision_fig6_chunk_sweep": (
        "REPORTED", "R3 comment 1", "20_train_convergence.py",
        "Four panels, one per chunk / block size (16, 128, 512, 1024). x: optimiser step. "
        "y: training cross-entropy, drug, MLP. The uniform-random reference is repeated in "
        "every panel.",
        "The direct answer to 'how much larger can chunk sizes be made while preserving "
        "comparable mini-batch diversity?'. Look for: pre-shuffled annbatch (dark green) "
        "sitting on the reference in **all four** panels, while un-pre-shuffled annbatch "
        "(light green) and scDataset (blue) pull away as the chunk grows and are "
        "unmistakable by 128. The quantitative version is `figure6_chunk_summary()`: "
        "deviation from the rolling median is 0.002-0.003 for pre-shuffled annbatch at every "
        "chunk size against 0.006 -> 0.042 without pre-shuffling."),
    "revision_scdataset_fig6_drug": (
        "REPORTED", "R2 major 3", "20_train_convergence.py",
        "x: optimiser step (minibatch 4,096). y: training cross-entropy, drug, MLP. "
        "Six arms, line width tracking importance. Red dashed verticals are the 12 plate "
        "boundaries.",
        "scDataset's Figure 6 reproduced with annbatch added, drug only because that is the "
        "discriminative task. Look for: thick green dashed annbatch lying on thick black "
        "uniform random for the whole epoch. Then scDataset at block 1024 (dark blue) sitting "
        "clearly above both, and streaming (brown) plus streaming-with-a-buffer (grey dashed) "
        "erratic in both directions. Streaming's *minimum* loss is 0.496 against annbatch's "
        "2.125, because a sequential minibatch is one compound and predicting it is trivial. "
        "What separates the arms is the stability of the trajectory, not its lowest point."),
    "revision_diversity_drug": (
        "supporting", "R1-2, R3-3, R3-4, R3-5", "10_batch_diversity.py",
        "x: chunk / block size, log scale. y: mean minibatch drug-label entropy in bits. "
        "Dashed rule = the ceiling (uniform random sampling, 8.42 bits); dotted rule = the "
        "floor (sequential streaming, 0.04 bits).",
        "Measured over all 94,129,984 rows from the index stream alone, without reading X, "
        "so it is exact at full scale rather than extrapolated. Look for: pre-shuffled "
        "annbatch flat on the ceiling at every chunk size, and the two unshuffled arms "
        "falling away together. scDataset is drawn as a large hollow square with "
        "un-pre-shuffled annbatch's circle nested inside it because the two agree to within "
        "0.01 bits, because they are the same algorithm at matched buffer, which is the point, and "
        "one would otherwise hide the other."),
    "revision_diversity_plate": (
        "supporting", "R1-2, R3-3", "10_batch_diversity.py",
        "As above, but the label is the plate of origin (14 plates, ceiling 3.67 bits).",
        "The plate label is the outermost on-disk block structure, so this is the cleanest "
        "view of what chunked reading costs when the label is perfectly correlated with "
        "position. Same reading as the drug panel; included because the drug result is easier "
        "to dismiss as a property of one label."),
    "revision_ondisk_structure_drug": (
        "supporting", "R3-1, R3-3", "05_ondisk_structure.py",
        "Distribution of run lengths (consecutive rows sharing a drug label) per plate.",
        "The mechanism behind the entropy curves. Look for: runs of 50,000-230,000 rows in 12 "
        "of the 14 plates. At minibatch 4,096 that is a change of drug composition every 12-56 "
        "steps, which is why a sequential reader meets distribution shift continuously rather "
        "than only at the 12 plate transitions, and why the loss spikes in Figure 6 are not "
        "confined to plate boundaries."),
    "revision_preshuffle_quality": (
        "supporting", "R3 comment 3", "06_preshuffle_quality.py",
        "x: the pre-shuffler's own `shuffle_chunk_size`. y: resulting minibatch entropy.",
        "R3-3 asks for guidance on two parameters, not one: `chunk_size` at load time and "
        "'the proportion of data to load by the pre-shuffler'. This is the second. Look for: "
        "entropy stays at the ceiling until `dataset_size / shuffle_chunk_size` falls below "
        "~2,000, which is the rule we give: the pre-shuffler's read granularity can be "
        "raised by three orders of magnitude before minibatch diversity suffers."),
    "revision_throughput_chunk_size": (
        "supporting", "R3 comment 1", "30_throughput_sweeps.py",
        "x: chunk / block size, log. y: samples/s, at a constant 16,384-row buffer, on a node "
        "held with --exclusive. Three repeats.",
        "The speed half of R3-1's trade-off, and the graph I.G. asked for ('speed vs chunk "
        "size, this will flatten off at some point'). Look for: a 17.5-fold gain from chunk 1 to "
        "4,096, flattening past ~1,024, with annbatch's default of 512 just below the knee. "
        "Caveat to state when quoting absolute values: annbatch's throughput varies 10-30% "
        "between identical repeats on this filesystem and --exclusive does not remove it."),
    "revision_dataset_size_real_loaders": (
        "supporting", "R2 major 5", "31_dataset_size_real_loaders.py",
        "x: training-set size, 2.1M to 94.1M cells, log. y: samples/s, log. Four real loaders "
        "on the formats they ship with, solid = cold cache and dashed = warm. Error bars are "
        "the spread over repeats.",
        "The panel F.F. specified ('x-axis dataset size by shard, y-axis loading speed') and "
        "the answer to whether amortisation weakens with scale. Look for: annbatch flat "
        "(1.04x over a 45-fold increase) while both h5ad-backed loaders lose 3-4x, and that "
        "the decline tracks the **storage format** rather than the loader: two independent "
        "implementations on h5ad both fall, two on Zarr both hold up. The error bars are not "
        "decoration: annbatch's own spread is 10-30%, so read its row as 'flat, no resolvable "
        "slope', not as a precise ratio."),
    "revision_wgs_memmap_baselines": (
        "supporting", "R2 minor 3", "40_wgs_memmap_baselines.py",
        "Four memmap access patterns plus annbatch, samples/s, on one node under a 64 GB "
        "cgroup against a 562 GiB array.",
        "R2 minor 3 asks for the setup to be 'fully described', so all four patterns are "
        "reported rather than one: the published implementation (39 samples/s), the same "
        "access pattern in-process without the IPC round trip (60), block-shuffled to match "
        "annbatch's sampler (280), and pure sequential streaming (495). Look for: annbatch at "
        "1,493, and the fact that the fair like-for-like comparison is the block-shuffled row "
        "(5.3x), not the published one (38.3x). The array is 9.4x the memory allowance, so at "
        "most ~11% can be cached; had it fit in RAM the comparison would say nothing."),
    "revision_wgs_ondisk_structure": (
        "supporting", "R2 major 6", "09_wgs_ondisk_structure.py",
        "Achievable minibatch diversity on the WGS cohort against chunk size, per MAF regime.",
        "Every chunk size sits on the log2(128) = 7-bit ceiling. The reason is an erratum we "
        "found: the 500,000-individual cohort is a 157-fold replication of 3,202 real "
        "individuals concatenated in order, so any 128 contiguous rows are 128 distinct "
        "individuals and the unshuffled order is already uncorrelated with sample identity. "
        "Consequence: for this benchmark the case for pre-shuffling rests on throughput, not "
        "diversity. Keep the erratum; see FULL_PACKAGE section 0 on whether to report the "
        "experiment."),
    "revision_scdataset_fig6_mlp": (
        "full notebook", "R2 major 3", "20_train_convergence.py",
        "As the drug panel, but all four tasks, MLP.",
        "The complete four-task reproduction. Cell line collapses immediately under **every** "
        "strategy including streaming, which is the negative control: the models train and "
        "the pipeline is correct, so the drug collapse is specifically the label correlated "
        "with on-disk order. Cramped at four panels, which is why the drug-only variant is "
        "the one reported."),
    "revision_scdataset_fig6_linear": (
        "full notebook", "R2 major 3", "20_train_convergence.py",
        "The same, for the linear classifier.",
        "Confirms the pattern is not an artefact of model capacity. scDataset's own paper "
        "makes the same point with two architectures."),
    "revision_validation_loss_drug_iid": (
        "supplementary", "R2 major 3", "20_train_convergence.py",
        "x: training samples seen, log. y: validation cross-entropy on the i.i.d. hold-out, "
        "**log scale**. Seven headline arms.",
        "The counterpart to Figure 6 and the reason held-out metrics are the arbiter. Look "
        "for: streaming's validation loss **rising** over training while its training loss "
        "dips low. That is the catastrophic-forgetting signature. Log y is necessary; on a linear "
        "axis the streaming arms compress everything else into the bottom fifth."),
    "revision_validation_f1_drug_iid": (
        "supplementary", "R2 major 3", "20_train_convergence.py",
        "As above, y: validation macro-F1 on drug.",
        "Part of the macro-F1 material that goes in the supplement rather than the response; "
        "see FULL_PACKAGE section 10.1."),
    "revision_final_macro_f1_iid": (
        "supplementary", "R2 major 3", "20_train_convergence.py",
        "Final macro-F1 after one epoch, all four tasks and both architectures, mean +- s.d. "
        "over three seeds.",
        "scDataset's Figure 5 equivalent. Reported as one supplementary table rather than a "
        "headline figure. Pre-shuffled annbatch 0.4717 +- 0.0009 on drug against 0.4731 +- "
        "0.0006 for uniform random, 0.28% relative, argued as non-inferiority within a 1% margin."),
    "revision_final_macro_f1_iid_bs64": (
        "supplementary", "R2 major 3", "20_train_convergence.py",
        "The same at minibatch 64, scDataset's own protocol.",
        "Rules out the objection that our larger minibatch changed the outcome. One seed per "
        "arm; the sweep above carries three."),
    "revision_validation_f1_drug_iid_bs64": (
        "supplementary", "R2 major 3", "20_train_convergence.py",
        "Validation macro-F1 against step at minibatch 64.",
        "Same purpose as the panel above."),
}

ORDER = ["REPORTED", "supporting", "supplementary", "full notebook", "SUPERSEDED"]


def main() -> int:
    root = Path(load_config()["repo"]) / "revision"
    on_disk = {f.stem for f in (root / "figures").glob("*.png")}
    missing = sorted(on_disk - set(NOTES))
    stale = sorted(set(NOTES) - on_disk)

    L = ["# What each figure shows\n",
         "Generated by `scripts/53_figure_notes.py`. One entry per file in",
         "`revision/figures/`, with the reviewer comment it answers and what to look",
         "for in it. `REPORTED` marks the two panels that go into the response.\n",
         "Both notebooks draw these from `results/*.json` at execution time:",
         "`notebooks/revision_reported.ipynb` for the reported set,",
         "`notebooks/revision_figures.ipynb` for all of it.\n"]

    for status in ORDER:
        items = sorted(k for k, v in NOTES.items() if v[0] == status and k in on_disk)
        if not items:
            continue
        L.append(f"## {status}\n")
        for stem in items:
            _, comment, script, axes, look = NOTES[stem]
            L.append(f"### `{stem}`\n")
            L.append(f"**Answers:** {comment} &nbsp;|&nbsp; **Produced by:** `{script}`\n")
            L.append(f"**Axes.** {axes}\n")
            L.append(f"**What to look for.** {look}\n")

    if missing:
        L.append("## Figures with no entry here\n")
        L.extend(f"- `{m}`" for m in missing)
        L.append("")

    out = root / "docs" / "FIGURES.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(NOTES)} entries, {len(on_disk)} figures on disk)")
    if missing:
        print(f"WARNING: {len(missing)} figure(s) with no note: {missing}")
    if stale:
        print(f"note: {len(stale)} note(s) for figures not on disk: {stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
