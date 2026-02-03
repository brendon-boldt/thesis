from functools import wraps
from pathlib import Path
from typing import Any, Callable, TextIO
import inspect

import scipy.stats
from matplotlib import pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np

sns.set_theme("paper", "darkgrid", font="serif", font_scale=0.8)

C = pl.col
RNG = np.random.default_rng()


def with_logfile(func) -> Callable:
    @wraps(func)
    def inner(*args, **kwargs) -> Any:
        name = inspect.signature(func).bind_partial(*args).arguments.get("name")
        namestr = ("-" + name) if name else ""
        logpath = Path(f"generated/log/{func.__name__}{namestr}")
        logpath.parents[0].mkdir(exist_ok=True, parents=True)
        with logpath.open("w") as fo:
            return func(*args, **kwargs, logfile=fo)

    return inner


def READ(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = (
        df.filter(
            ~(C("sender") == "s_nn")
        )
        .with_columns(
            split=C("split").replace("train", "Train").replace("test", "Test"),
        )
        .with_columns((C("sender") + "\n" + C("receiver")).alias("pair"))
        .with_columns(
            (C("sender") + "->" + C("receiver"))
            .replace(
                {
                    "s_nd->r_nd": "Online",
                    "s_ns->r_nd": "Offline Sen.",
                    "s_pb->r_nd": "PB Sender",
                    "s_nd->r_ns": "Offline Rec.",
                    "s_nd->r_pb": "PB Receiver",
                }
            )
            .alias("pair")
        )
    )
    return df


def SAVE(path: str) -> None:
    p = Path(path)
    p.parents[0].mkdir(exist_ok=True, parents=True)
    # margin = 0.15
    # plt.subplots_adjust(left=0.15, right=1-margin, top=1-margin, bottom=margin)
    # plt.margins(x=0, y=0)
    plt.savefig(p)
    if p.name.endswith(".pdf"):
        plt.savefig(p.with_suffix(".png"))
    plt.close()


def value_key(custom_vals: list[str]) -> Callable[[str], str]:
    def inner(x: str) -> str:
        y: object = x
        try:
            y = float(custom_vals.index(x))
        except:
            pass
        try:
            y = float(x)
        except:
            pass
        if isinstance(y, float):
            return f"{y:030f}"
        return str(y)

    return inner


@with_logfile
def env_ab(name: str, width: float, height: float, params: list[str], *, logfile: TextIO) -> None:
    df = READ("assets/env-ab-metrics.parquet")
    # fmt: off
    custom_vals = [
        "rnn", "lstm", "gru",
        "", "s", "ss", "sss", "r", "rr", "rrr", "sr", "rs",
    ]
    # fmt: on

    df = (
        df.filter(
            (C("metric") == "acc_exact")
            & (C("split") == "Test")
            & (C("param_name").is_in(params))
        )
        .sort(C("param_value").map_elements(value_key(custom_vals)))
        .sort(C("param_name").map_elements(params.index, return_dtype=int))
        .with_columns(
            param_name=C("param_name").replace(
                {
                    "n_values": "$n$ values",
                    "n_attributes": "$n$ attributes",
                    "test_prop": "Test proportion",
                    "vocab_size": "Vocabulary size",
                    "max_len": "Max message length",
                    "training_schedule": "Reset schedule",
                    "straight_through": "Gumbel-Softmax straight through",
                    "reset_optimizer": "Reset optimizer parameters",
                    "loss_func": "Loss function",
                    "sender_n_embedding_layers": "$n$ embedding layers, sender",
                    "receiver_n_embedding_layers": "$n$ embedding layers, receiver",
                    "sender_n_rnn_layers": "$n$ RNN layers, sender",
                    "receiver_n_rnn_layers": "$n$ RNN layers, receiver",
                    "cell_type": "RNN cell type",
                }
            ),
            param_value=C("param_value").replace(
                {
                    "s": "S",
                    "ss": "SS",
                    "sss": "SSS",
                    "r": "R",
                    "rr": "RR",
                    "rrr": "RRR",
                    "sr": "SR",
                    "rs": "RS",
                    "ce": "Cross-entropy",
                    "mse": "Mean squared error",
                    "rnn": "Elman",
                    "lstm": "LSTM",
                    "gru": "GRU",
                    "": "None",
                    # "0": "False",
                    # "1": "True",
                }
            ),
        )
        .rename(
            {
                "value": "Test Accuracy",
                "pair": "Agents",
                # "param_name": "Parameter",
            }
        )
    )

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        cols = ['param_key', 'param_value', 'Agents']
        summary = df.group_by(cols).median().sort(cols).select(cols + ["Test Accuracy"])
        print(summary, file=logfile)

    aspect = width / height
    g = sns.catplot(
        df.to_pandas(),
        hue="Agents",
        x="param_value",
        y="Test Accuracy",
        kind="boxen",
        col="param_name",
        col_wrap=1 if name == "main" else 2,
        sharex=False,
        # dodge=True,
        # height=height,
        # aspect=aspect,
        # order=order,
        legend=True,
        legend_out=name != "main",
        height=height,
        aspect=aspect,
    )
    g.set_titles(template="{col_name}")
    g.set_xlabels("")
    # plt.ylabel("Accuracy")
    # plt.xlabel(None)
    if name == "main":
        plt.tight_layout()
    SAVE(f"generated/env-ab/{name}.pdf")


@with_logfile
def ag_ab(name: str, width: float, height: float, variants: list[str], *, logfile: TextIO) -> None:
    df = READ("assets/ag-ab-metrics.parquet")
    df = df.filter(
        (C("metric") == "acc_exact")
        & (C("split") == "Test")
        & (C("variant_name").is_in(variants + ["base"]))
    )

    base_value_map = {
        "invsize": "∞*",
        "maxngram": "∞*",
        "maxsemcomps": "∞*",
        "searchbest": 1,
        "stripeos": 1,
        "weight": "MI*",
        "order": "Affinity*",
        "repeatm": 1,
        "ablatem": 1,
        "ablatef": 0,
        "method": "Greedy*",
        "idempotent": 0,
    }
    # fmt: off
    custom_vals = [
        "mi", "npmi", "jp", "pmim", "app",
        "greedy", "search", "ip",
        "affinity", "insertion", "shuffle",
        # "No", "No*", "Yes", "Yes*",
    ]
    # fmt: on

    def rename_values(d: dict) -> str:
        key = d['variant_name'], d['variant_value']
        return {
            ("searchbest", "0"): "No",
            ("searchbest", "1"): "Yes*",
            ("stripeos", "0"): "No",
            ("stripeos", "1"): "Yes*",
            ("repeatm", "0"): "No",
            ("repeatm", "1"): "Yes*",
            ("ablatem", "0"): "No",
            ("ablatem", "1"): "Yes*",
            ("idempotent", "0"): "No*",
            ("idempotent", "1"): "Yes",
            ("ablatef", "0"): "No*",
            ("ablatef", "1"): "Yes",
        }.get(key, key[1])

    bvdf = pl.from_records(
        [(k, str(base_value_map[k])) for k in variants], orient="row"
    )

    cjbvdf = (
        df.filter(C("variant_name") == "base")
        .join(bvdf, how="cross")
        .with_columns(variant_name=C("column_0"), variant_value=C("column_1"))
        .drop("column_0", "column_1")
    )

    # print(pl.concat([df, cjbvdf], how="vertical").filter(~C("variant_name").is_in(["base", ""]))["variant_name"].unique())
    df = (
        pl.concat([df, cjbvdf], how="vertical")
        .filter(~C("variant_name").is_in(["base", ""]))
        .sort(C("variant_value").map_elements(value_key(custom_vals)))
        .sort(C("variant_name").map_elements(variants.index, return_dtype=int))
        .rename({"value": "Test Accuracy", "pair": "Pair"})
        .with_columns(
            variant_name=C("variant_name").replace(
                {
                    "stripeos": "Strip EoS token",
                    "searchbest": "Search best (CSAR)",
                    "weight": "Weight method (CSAR)",
                    "maxngram": "Max form length (CSAR)",
                    "maxsemcomps": "Max meaning size (CSAR)",
                    "invsize": "Inventory size (CSAR)",
                    "method": "Sender algorithm",
                    "repeatm": "Repeat morpheme, sender",
                    "ablatem": "Ablate meaning",
                    "ablatef": "Ablate form",
                    "idempotent": "Idempotent form",
                    "order": "Form order",
                }
            ),
            variant_value=pl.struct("variant_name", "variant_value").map_elements(rename_values, return_dtype=pl.String).replace({
                "npmi": "NPMI",
                "jp": "JP",
                "pmim": "PMIM",
                "app": "App",
                "search": "Search",
                "ip": "IP",
                "insertion": "Insertion",
                "shuffle": "Shuffle",
            })
        )
    )

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        cols = ['variant_name', 'variant_value', 'Pair']
        summary = df.group_by(cols).median().sort(cols).select(cols + ["Test Accuracy"])
        print(summary, file=logfile)

    aspect = width / height

    g = sns.catplot(
        df.to_pandas(),
        hue="Pair",
        x="variant_value",
        y="Test Accuracy",
        sharex=False,
        sharey=name != "main",
        # col_wrap=3 if len(variants) == 3 else 2,
        col_wrap=1 if name == "main" else 2,
        col="variant_name",
        kind="boxen",
        height=height,
        aspect=aspect,
        legend=True,
        legend_out=name != "main",
    )
    g.set_titles(template="{col_name}")
    g.set_xlabels("")
    if name == "main":
        plt.tight_layout()
        plt.legend(loc="lower right")
    SAVE(f"generated/ag-ab/{name}.pdf")


def get_strat_idxs(bins: int, vals: np.ndarray) -> np.ndarray:
    lo, hi = vals.min(), vals.max()
    ls = np.linspace(lo, hi, num=bins, endpoint=False)[1:]
    strat_idxs = (ls[None] <= vals[:, None]).sum(-1)
    counts = np.unique(strat_idxs, return_counts=True)[1]
    # weights = (1 / bins / counts)[strat_idxs]
    samps = counts.min()
    return np.array([
        RNG.choice(len(vals), samps, replace=False, p=(strat_idxs == si).astype(float) / count)
        for si, count in enumerate(counts)
    ]).flatten()



@with_logfile
def accuracy_scatter(name: str, width: float, height: float, *, logfile: TextIO) -> None:
    full_name = name
    name = name.removeprefix("app_")
    df = READ("assets/env-ab-metrics.parquet")
    df = df.filter(
        ~C("param_name").is_in(["n_values", "n_attributes", "test_prop"])
        & (
            (
                (C("metric") == "acc_exact")
                & (C("split") == "Test")
                & C("pair").is_in(["PB Sender", "PB Receiver"])
            )
            | (C("metric") == name)
        )
    ).select(["param_name", "trial_number", "pair", "metric", "value"])
    df = df.filter(C("metric") == "acc_exact").join(
        df.filter(C("metric") == name), on=["param_name", "trial_number"]
    )
    N_BINS = 5
    df = df[get_strat_idxs(N_BINS, np.array(df['value_right']))]

    r2s = []
    for pair in df["pair"].unique():
        df_ = df.filter(C("pair") == pair)
        res = scipy.stats.pearsonr(df_["value_right"], df_["value"])
        r2 = res.statistic**2
        r2s.append(r2)
        print(f"{pair}: r={res.statistic:.3f} R²={r2:.3f}", file=logfile)
    print(f"mean: R²={sum(r2s)/len(r2s):.3f}", file=logfile)

    palette = sns.color_palette("deep")
    jitter = (np.random.default_rng().random((2, len(df))) - 0.5) * 0.08
    if name in ["inventory_mi", "inventory_mi_norm", "inventory_npmi"]:
        maxval = np.inf
    else:
        maxval = 1
    df = df.with_columns(
        color=C("pair").replace_strict(
            {"PB Sender": palette[0], "PB Receiver": palette[1]},
            return_dtype=pl.List(pl.Float64),
        ),
        value=(C("value") + jitter[0]).clip(0, 1),
        value_right=(C("value_right") + jitter[1]).clip(0, maxval),
    )
    df = df.sample(fraction=1.0, shuffle=True)

    plt.scatter(df["value_right"], df["value"], s=10, c=df["color"], alpha=0.4)
    plt.xlabel(None)
    plt.ylabel("Test Accuracy")
    plt.gcf().set_size_inches(width, height)
    plt.tight_layout()

    SAVE(f"generated/acc-scatter/{full_name}.pdf")

def inventory_table(exp_name: str) -> None:
    def float_format(x: float) -> str:
        exp = int(np.floor(np.log10(x)))
        mantissa = f"{x:.2e}".split("e")[0]
        # return f"${mantissa}\\cdot10^{{{exp}}}$"
        return f"${mantissa}\\,\\text{{e}}{{-}}{abs(exp)}$"
    tstr = (
        pl
        .read_parquet(f"assets/inventory-{exp_name}.parquet")
        [:60]
        .with_columns(
            weight=-C("weight"),
            form=C("form").str.replace('"', "", n=-1),
        )
        .rename({
            "prevalence": "Proportion",
            "weight": "Weight",
            "meaning": "Meaning",
            "form": "Form",
        })
        .to_pandas()
        .to_latex(
            index=False,
            escape=False,
            # float_format="${:.2e}$".format
            float_format=float_format,
            longtable=False,
        )
    )
    tstr = f"{{\\small\n{tstr}\n}}"
    out_path = Path(f"generated/inventory/{exp_name}.tex")
    out_path.parents[0].mkdir(exist_ok=True, parents=True)
    out_path.write_text(tstr)

def get_inventory_metric_data() -> pl.DataFrame:
    return (
        pl
        .read_parquet(f"assets/linguistic_metrics.parquet")
        .with_columns(
            name=C("name").replace({
                "inventory_size": "Inventory Size",
                "vocabulary_size": "Vocabulary Size",
                "inventory_entropy": "Inventory Entropy",
                "meaning_size": "Meaning Size",
                "form_length": "Form Length",
                "morpheme_bijectivity": "Morpheme Bijectivity",
                "forms_per_meaning": "Forms per Meaning",
                "meanings_per_form": "Meanings per Form",
                "synonymy_entropy": "Synonymy Entropy",
                "polysemy_entropy": "Polysemy Entropy",
            }),
        )
    )

def inventory_metrics_hist(dim: tuple[float, float]) -> None:
    df = get_inventory_metric_data()

    grid = sns.displot(
        height=dim[1],
        aspect=dim[0] / dim[1],
        data=df.to_pandas(),
        x="value",
        col="name",
        col_wrap=3,
        stat="density",
        kind="hist",
        element="step",
        alpha=0.5,
        linewidth=0,
        common_bins=False,
        facet_kws=dict(
            sharex=False,
            sharey=False,
        ),
    )
    grid.set_titles(template="{col_name}")
    grid.set_xlabels("")

    SAVE(f"generated/inventory-metrics/all.pdf")

def inventory_metrics_pair(dim: tuple[float, float], metrics: list[str]) -> None:
    df = get_inventory_metric_data()
    df = df.pivot("name", values="value")

    sns.pairplot(
        height=dim[1],
        aspect=dim[0] / dim[1],
        data=df.to_pandas(),
        kind="hist",
        vars=metrics,
        plot_kws=dict(
            # fill=True,
        ),
        diag_kws=dict(
            alpha=0.5,
            element="step",
            linewidth=0,
        ),
    )
    SAVE(f"generated/inventory-metrics/pair.pdf")
