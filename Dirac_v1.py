#!/usr/bin/env python3


"""
Paul Adrien Maurice Dirac

Matematicka adaptacija diskretnog Dirakovog operatora:
    D(q, p) = (B p, B.T q)

q opisuje celu kombinaciju u odnosu na istorijska stanja.
p opisuje promenu tog opisa na granama grafa.

Koeficijenti dinamike uce se iskljucivo iz prethodnih kola.
Skor kandidata nije verovatnoca dobitka.
"""


import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


DEFAULTS = [
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_2973.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_plus_1729.csv",
]

TRAIN = 192
VALIDATION = 48
TEST = 64
REFIT = 16

# Broj cvorova, ridge regularizacija, tezina greske toka.
CONFIGS = (
    (24, 0.03, 0.25),
    (24, 0.30, 1.00),
    (40, 0.03, 0.25),
    (40, 0.30, 1.00),
)

PAIR = np.asarray(
    list(combinations(range(7), 2)),
    dtype=np.int32,
)


def read_csv(path):
    result = []

    with Path(path).open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line, cells in enumerate(csv.reader(handle), 1):
            if not cells or all(not x.strip() for x in cells):
                continue

            try:
                row = tuple(
                    sorted(int(x.strip()) for x in cells)
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: neispravan ceo broj."
                ) from exc

            if (
                len(row) != 7
                or len(set(row)) != 7
                or not 1 <= row[0] <= row[-1] <= 39
            ):
                raise ValueError(
                    f"{path}, red {line}: potrebno je "
                    "7 razlicitih brojeva 1..39."
                )

            result.append(row)

    minimum = TRAIN + VALIDATION + TEST + 2

    if len(result) < minimum:
        raise ValueError(
            f"{path}: potrebno najmanje {minimum} kola; "
            f"ima {len(result)}."
        )

    return np.asarray(result, dtype=np.int32)


def profile(rows):
    """
    Opis pojedinacne cele kombinacije.

    Sedam tacaka odredjuju njenu empirijsku distribuciju.
    Dodatno se opisuje distribucija svih 21 parnih rastojanja.
    Nema tabela istorijskih frekvencija pojedinacnih brojeva.
    """
    a = np.atleast_2d(rows).astype(np.float64)

    distances = (
        a[:, PAIR[:, 1]]
        - a[:, PAIR[:, 0]]
    )

    cdf = np.stack(
        [
            (distances <= boundary).mean(axis=1)
            for boundary in (2, 5, 10, 20)
        ],
        axis=1,
    )

    span = (a[:, -1:] - a[:, :1]) / 38.0

    return np.concatenate(
        (a / 39.0, cdf, span),
        axis=1,
    )


def kernel(a, b, bandwidth):
    """Racionalna slicnost celih kombinacija."""
    distances = np.maximum(
        (a * a).sum(axis=1)[:, None]
        + (b * b).sum(axis=1)[None, :]
        - 2 * a @ b.T,
        0.0,
    ) / a.shape[1]

    return 1.0 / (1.0 + distances / bandwidth)


def incidence(anchors):
    """Normalizovana orijentisana matrica incidencije grafa."""
    count = len(anchors)

    distances = (
        (anchors[:, None, :] - anchors[None, :, :]) ** 2
    ).mean(axis=2)

    np.fill_diagonal(distances, np.inf)

    edges = set()

    for i in range(count):
        neighbors = np.argsort(
            distances[i],
            kind="stable",
        )[:3]

        for j in neighbors:
            edges.add(
                tuple(sorted((i, int(j))))
            )

    edges = sorted(edges)
    degree = np.zeros(count)

    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    b = np.zeros((count, len(edges)))

    for edge, (i, j) in enumerate(edges):
        b[i, edge] = -1.0 / np.sqrt(degree[i])
        b[j, edge] = 1.0 / np.sqrt(degree[j])

    return b


class ComponentRidge:
    """
    Zasebna mala regresija za svaku komponentu.

    x: vreme × komponenta × ulaz
    y: vreme × komponenta
    """

    def __init__(self, x, y, penalty):
        self.xmean = x.mean(axis=0)
        self.xscale = np.maximum(
            x.std(axis=0),
            1e-6,
        )
        self.ymean = y.mean(axis=0)

        standardized = (
            x - self.xmean
        ) / self.xscale

        centered = y - self.ymean

        gram = np.einsum(
            "tni,tnj->nij",
            standardized,
            standardized,
        ) / len(standardized)

        rhs = np.einsum(
            "tni,tn->ni",
            standardized,
            centered,
        ) / len(standardized)

        identity = np.eye(x.shape[2])[None, :, :]

        self.coef = np.linalg.solve(
            gram + penalty * identity,
            rhs[..., None],
        )[..., 0]

    def predict(self, x):
        standardized = (
            x - self.xmean
        ) / self.xscale

        return self.ymean + np.einsum(
            "ni,ni->n",
            standardized,
            self.coef,
        )


class DiracModel:
    def __init__(
        self,
        history,
        config,
        state_only=False,
    ):
        count, penalty, self.flow_weight = config
        self.state_only = state_only

        # history sadrzi samo poznata kola.
        features = profile(history)
        start = max(2, len(history) - TRAIN)

        fit_features = features[start - 2:]

        self.center = np.median(
            fit_features,
            axis=0,
        )

        q25, q75 = np.quantile(
            fit_features,
            [0.25, 0.75],
            axis=0,
        )

        self.scale = np.maximum(
            q75 - q25,
            0.03,
        )

        standardized = (
            features - self.center
        ) / self.scale

        # Deterministicki izbor istorijskih cvorova.
        indexes = np.linspace(
            start - 2,
            len(history) - 1,
            count,
        ).astype(int)

        self.anchors = standardized[indexes]

        distances = (
            (
                self.anchors[:, None]
                - self.anchors[None, :]
            ) ** 2
        ).mean(axis=2)

        positive = distances[
            np.triu_indices(count, 1)
        ]
        positive = positive[positive > 1e-12]

        self.bandwidth = (
            float(np.median(positive))
            if len(positive)
            else 1.0
        )

        self.b = incidence(self.anchors)

        q = kernel(
            standardized,
            self.anchors,
            self.bandwidth,
        )

        # Za cilj kolo t:
        # current = stanje kola t-1
        # previous = stanje kola t-2
        # target = stanje stvarnog kola t
        current = q[start - 1:-1]
        previous = q[start - 2:-2]
        target = q[start:]

        delta = current - previous
        p = delta @ self.b

        # Vektori su zapisani kao redovi.
        divergence = p @ self.b.T
        gradient = current @ self.b

        if state_only:
            node_inputs = current[..., None]
        else:
            node_inputs = np.stack(
                (current, delta, divergence),
                axis=2,
            )

        self.node_model = ComponentRidge(
            node_inputs,
            target,
            penalty,
        )

        self.node_scale = np.maximum(
            target.std(axis=0),
            0.02,
        )

        if not state_only:
            edge_inputs = np.stack(
                (p, -gradient),
                axis=2,
            )

            edge_target = (
                target - current
            ) @ self.b

            self.edge_model = ComponentRidge(
                edge_inputs,
                edge_target,
                penalty,
            )

            self.edge_scale = np.maximum(
                edge_target.std(axis=0),
                0.02,
            )

    def embed(self, rows):
        standardized = (
            profile(rows) - self.center
        ) / self.scale

        return kernel(
            standardized,
            self.anchors,
            self.bandwidth,
        )

    def forecast(self, history):
        previous, current = self.embed(
            history[-2:]
        )

        delta = current - previous
        p = delta @ self.b

        if self.state_only:
            node_inputs = current[:, None]
        else:
            node_inputs = np.stack(
                (
                    current,
                    delta,
                    p @ self.b.T,
                ),
                axis=1,
            )

        node = self.node_model.predict(
            node_inputs
        )

        edge = None

        if not self.state_only:
            edge_inputs = np.stack(
                (
                    p,
                    -(current @ self.b),
                ),
                axis=1,
            )

            edge = self.edge_model.predict(
                edge_inputs
            )

        return current, node, edge

    def predict(self, history):
        current, node, edge = self.forecast(
            history
        )

        def costs(candidates):
            q = self.embed(candidates)

            result = (
                (
                    (q - node) / self.node_scale
                ) ** 2
            ).mean(axis=1)

            if not self.state_only:
                p = (q - current) @ self.b

                result += self.flow_weight * (
                    (
                        (p - edge) / self.edge_scale
                    ) ** 2
                ).mean(axis=1)

            if not np.isfinite(result).all():
                raise RuntimeError(
                    "Neispravan skor modela."
                )

            return result

        # Istorijske kombinacije sluze kao pocetne tacke
        # deterministicke pretrage, ne kao obavezni izlazi.
        recent = history[-96:]
        distances = costs(recent)

        seed_ids = np.argsort(
            distances,
            kind="stable",
        )[:12]

        seeds = {
            tuple(map(int, recent[i]))
            for i in seed_ids
        }

        seeds.update(
            tuple(map(int, row))
            for row in history[-8:]
        )

        def rank(candidates):
            candidates = sorted(candidates)
            values = costs(
                np.asarray(candidates)
            )

            order = sorted(
                range(len(candidates)),
                key=lambda i: (
                    float(values[i]),
                    candidates[i],
                ),
            )

            return [
                candidates[i]
                for i in order[:4]
            ]

        beam = rank(seeds)

        best_cost = float(
            costs(np.asarray(beam[:1]))[0]
        )

        # Menja se jedan broj; ogranicenja su samo
        # pravila igre: 7 razlicitih brojeva iz 1..39.
        for _ in range(6):
            candidates = set(beam)

            for row in beam:
                used = set(row)

                for removed in row:
                    others = used - {removed}

                    for added in range(1, 40):
                        if added not in used:
                            candidates.add(
                                tuple(
                                    sorted(
                                        others | {added}
                                    )
                                )
                            )

            new_beam = rank(candidates)

            new_cost = float(
                costs(
                    np.asarray(new_beam[:1])
                )[0]
            )

            if new_cost >= best_cost - 1e-12:
                break

            beam = new_beam
            best_cost = new_cost

        prediction = beam[0]

        assert len(prediction) == len(set(prediction)) == 7
        assert 1 <= prediction[0] <= prediction[-1] <= 39

        return prediction


def evaluate(
    rows,
    start,
    stop,
    config,
    state_only=False,
):
    predictions = []
    values = []

    for t in range(start, stop):
        if (t - start) % REFIT == 0:
            model = DiracModel(
                rows[:t],
                config,
                state_only,
            )

        prediction = model.predict(rows[:t])

        predictions.append(prediction)

        values.append(
            len(
                set(prediction)
                & set(map(int, rows[t]))
            )
        )

    return np.asarray(values), predictions


def choose(
    rows,
    start,
    stop,
    state_only=False,
):
    records = []
    label = (
        "Samo stanje"
        if state_only
        else "Dirac"
    )

    for config in CONFIGS:
        values, _ = evaluate(
            rows,
            start,
            stop,
            config,
            state_only,
        )

        mean = float(values.mean())
        records.append(mean)

        print(
            f"Validacija {label} {config}: "
            f"{mean:.4f}",
            flush=True,
        )

    # Kod jednakog rezultata prednost ima
    # ranije navedena konfiguracija.
    return CONFIGS[int(np.argmax(records))]


def longest_run(row):
    best = current = 1

    for a, b in zip(row, row[1:]):
        current = (
            current + 1
            if b == a + 1
            else 1
        )
        best = max(best, current)

    return best


def self_check():
    rows = np.asarray(
        [
            sorted({
                (3 * i + 5 * j) % 39 + 1
                for j in range(7)
            })
            for i in range(240)
        ],
        dtype=np.int32,
    )

    b = incidence(profile(rows[:24]))

    d = np.block([
        [
            np.zeros((b.shape[0], b.shape[0])),
            b,
        ],
        [
            b.T,
            np.zeros((b.shape[1], b.shape[1])),
        ],
    ])

    assert np.allclose(d, d.T)

    square = d @ d

    assert np.allclose(
        square[:len(b), :len(b)],
        b @ b.T,
    )

    assert np.allclose(
        square[len(b):, len(b):],
        b.T @ b,
    )

    model = DiracModel(
        rows[:210],
        CONFIGS[0],
    )

    altered = rows.copy()
    altered[210:] = np.arange(1, 8)

    other = DiracModel(
        altered[:210],
        CONFIGS[0],
    )

    assert np.array_equal(
        model.b,
        other.b,
    )

    assert (
        model.predict(rows[:210])
        == other.predict(altered[:210])
    )

    assert (
        model.predict(rows[:210])
        == model.predict(rows[:210])
    )


def run(path):
    rows = read_csv(path)
    n = len(rows)

    test_start = n - TEST
    val_start = test_start - VALIDATION

    print(
        f"\n{Path(path).name}: {n} kola",
        flush=True,
    )

    config = choose(
        rows,
        val_start,
        test_start,
    )

    baseline_config = choose(
        rows,
        val_start,
        test_start,
        state_only=True,
    )

    values, predictions = evaluate(
        rows,
        test_start,
        n,
        config,
    )

    state_values, _ = evaluate(
        rows,
        test_start,
        n,
        baseline_config,
        state_only=True,
    )

    previous = np.asarray([
        len(
            set(map(int, rows[t - 1]))
            & set(map(int, rows[t]))
        )
        for t in range(test_start, n)
    ])

    historic = sum(
        any(
            tuple(map(int, row)) == prediction
            for row in rows[:t]
        )
        for t, prediction in zip(
            range(test_start, n),
            predictions,
        )
    )

    print(f"Konfiguracija: {config}")

    print(
        f"TEST Dirac: {values.mean():.4f}"
    )

    print(
        "TEST samo stanje: "
        f"{state_values.mean():.4f}"
    )

    print(
        "Kontrola: prethodno izvuceno kolo: "
        f"{previous.mean():.4f}"
    )

    print(
        "Pogodaka 0..7:",
        np.bincount(
            values,
            minlength=8,
        ).tolist(),
    )

    print(
        f"Razlicitih predloga: "
        f"{len(set(predictions))}/{TEST}; "
        f"istorijskih: {historic}/{TEST}"
    )

    print(
        "Predloga sa nizom 5+:",
        sum(
            longest_run(p) >= 5
            for p in predictions
        ),
    )

    print(
        "Predloga potpuno u jednoj polovini:",
        sum(
            p[-1] <= 19 or p[0] >= 20
            for p in predictions
        ),
    )

    model = DiracModel(rows, config)
    prediction = model.predict(rows)

    assert prediction == model.predict(rows)

    print(
        f"NEXT red {n + 1}: "
        + ",".join(map(str, prediction))
    )

    print(
        "Postoji u istoriji:",
        any(
            tuple(map(int, row)) == prediction
            for row in rows
        ),
    )

    print(
        f"Najduzi niz: {longest_run(prediction)}; "
        f"1..19: {sum(x <= 19 for x in prediction)}"
    )

    if values.mean() <= state_values.mean():
        print(
            "Sprega nije nadmasila varijantu "
            "koja koristi samo stanje."
        )

    if values.mean() <= previous.mean():
        print(
            "Model nije nadmasio kontrolu "
            "prethodnog kola."
        )

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dirac V1 za Loto 7/39; "
            "potreban samo numpy."
        )
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="CSV bez zaglavlja, 7 brojeva po redu.",
    )

    args = parser.parse_args()

    self_check()

    print(
        "Dirac V1: diskretni operator stanja/promene; "
        "skor nije verovatnoca dobitka.",
        flush=True,
    )

    results = [
        (path, run(path))
        for path in (args.csv or DEFAULTS)
    ]

    # Jednak izlaz se prijavljuje, bez vestacke izmene.
    for i, (path, prediction) in enumerate(results):
        for other_path, other in results[:i]:
            if prediction == other:
                print(
                    f"Isti NEXT: {Path(path).name} / "
                    f"{Path(other_path).name}: "
                    f"{prediction}"
                )


if __name__ == "__main__":
    main()



"""
Dirac V1: diskretni operator stanja/promene; skor nije verovatnoca dobitka.


loto7_4702_k82.csv: 4702 kola
Validacija Dirac (24, 0.03, 0.25): 1.1667
Validacija Dirac (24, 0.3, 1.0): 1.2500
Validacija Dirac (40, 0.03, 0.25): 1.1875
Validacija Dirac (40, 0.3, 1.0): 1.3958
Validacija Samo stanje (24, 0.03, 0.25): 1.1250
Validacija Samo stanje (24, 0.3, 1.0): 1.1042
Validacija Samo stanje (40, 0.03, 0.25): 1.1667
Validacija Samo stanje (40, 0.3, 1.0): 1.1042
Konfiguracija: (40, 0.3, 1.0)
TEST Dirac: 1.2812
TEST samo stanje: 1.3906
Kontrola: prethodno izvuceno kolo: 1.1719
Pogodaka 0..7: [13, 28, 16, 6, 1, 0, 0, 0]
Razlicitih predloga: 49/64; istorijskih: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT red 4703: 3,4,22,24,25,31,32
Postoji u istoriji: False
Najduzi niz: 2; 1..19: 2
Sprega nije nadmasila varijantu koja koristi samo stanje.



loto7_4702_k82_loto_2973.csv: 2973 kola
Validacija Dirac (24, 0.03, 0.25): 1.4167
Validacija Dirac (24, 0.3, 1.0): 1.0208
Validacija Dirac (40, 0.03, 0.25): 1.2500
Validacija Dirac (40, 0.3, 1.0): 1.3542
Validacija Samo stanje (24, 0.03, 0.25): 1.4375
Validacija Samo stanje (24, 0.3, 1.0): 1.3542
Validacija Samo stanje (40, 0.03, 0.25): 1.2083
Validacija Samo stanje (40, 0.3, 1.0): 1.2500
Konfiguracija: (24, 0.03, 0.25)
TEST Dirac: 1.3594
TEST samo stanje: 1.3125
Kontrola: prethodno izvuceno kolo: 1.3281
Pogodaka 0..7: [12, 28, 16, 5, 3, 0, 0, 0]
Razlicitih predloga: 41/64; istorijskih: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT red 2974: 3,5,23,24,25,29,36
Postoji u istoriji: False
Najduzi niz: 3; 1..19: 2



loto7_4702_k82_loto_plus_1729.csv: 1729 kola
Validacija Dirac (24, 0.03, 0.25): 1.1458
Validacija Dirac (24, 0.3, 1.0): 1.1667
Validacija Dirac (40, 0.03, 0.25): 1.3125
Validacija Dirac (40, 0.3, 1.0): 1.4167
Validacija Samo stanje (24, 0.03, 0.25): 1.3125
Validacija Samo stanje (24, 0.3, 1.0): 1.2292
Validacija Samo stanje (40, 0.03, 0.25): 1.4167
Validacija Samo stanje (40, 0.3, 1.0): 1.4583
Konfiguracija: (40, 0.3, 1.0)
TEST Dirac: 1.2031
TEST samo stanje: 1.2500
Kontrola: prethodno izvuceno kolo: 1.2656
Pogodaka 0..7: [17, 24, 16, 7, 0, 0, 0, 0]
Razlicitih predloga: 39/64; istorijskih: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT red 1730: 6,7,8,25,26,34,35
Postoji u istoriji: False
Najduzi niz: 3; 1..19: 3
Sprega nije nadmasila varijantu koja koristi samo stanje.
Model nije nadmasio kontrolu prethodnog kola.
"""



"""
Iz Dirakove jednačine zajedno modelovati stanje distribucije i smer njenog menjanja.
Dirak je koristio matričnu jednačinu prvog reda sa međusobno povezanim komponentama stanja. 
To je matematička osnova inspiracije. Dirakov originalni rad, 1928.

- Stanje: opis rasporeda svih sedam brojeva i njihovih međusobnih odnosa.
- Promena: kako se taj opis promenio kroz prethodna kola.
- Povezana dinamika: model uči kako prethodno stanje utiče na promenu i kako promena utiče na naredno stanje.

Važna razlika: dve istorijske situacije mogu imati sličan trenutni raspored, ali različit prethodni tok. Model bi ih razlikovao, umesto da im automatski dodeli isti NEXT.
Za konkretnu matematičku konstrukciju koristiti diskretni Dirakov operator na grafu, koji povezuje opis stanja sa opisom promena između povezanih stanja. 
Proveravamo baš doprinos povezivanja „stanje + promena“. 
Ključna provera bila bi poređenje sa istim modelom koji koristi samo stanje. 


Provera na poslednja 64 kola, nakon izbora parametara na prethodnih 48:

CSV	                Dirac	Samo stanje	Prethodno kolo
Zajednički 4702	    1,2812	1,3906	    1,1719
Loto 2973	        1,3594	1,3125	    1,3281
Plus 1729	        1,2031	1,2500	    1,2656


Sprega stanja i promene pomogla je samo na Loto CSV-u. 
Na druga dva nije nadmašila varijantu koja koristi samo stanje.
Nema random postupaka, nametnute normalne ili uniformne raspodele ni ručnih filtera kombinacije. 
U testu nije bilo nizova od pet uzastopnih brojeva niti kombinacija potpuno u jednoj polovini.
"""
