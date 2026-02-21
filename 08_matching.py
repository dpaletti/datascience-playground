import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Matching
    We want to estimate the effect of having a premium or better cut on price
    """)
    return


@app.cell
def _():
    outcome = "price"
    cut = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    treatment = "is_cut_premium_or_better"
    return outcome, treatment


@app.cell
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import polars as pl
    from sklearn.preprocessing import MinMaxScaler
    import statsmodels.formula.api as smf
    from sklearn.linear_model import LogisticRegression
    import plotly.express as px
    import plotly.io as pio
    import numpy as np


    pio.templates.default = "plotly_white"
    return LogisticRegression, MinMaxScaler, np, pl, px, smf


@app.cell
def _(mo):
    mo.md(r"""
    ## Data reading
    """)
    return


@app.cell
def _(pl):
    diamonds = (
        pl.read_csv("./data/diamonds.csv")
        .with_columns(
            is_cut_premium_or_better=pl.col("cut")
            .is_in({"Premium", "Ideal"})
            .cast(pl.UInt8)
        )
        .drop(["", "x", "y", "z", "cut"])
    )
    return (diamonds,)


@app.cell
def _(MinMaxScaler, diamonds, pl):
    scaler = MinMaxScaler().set_output(transform="polars").fit(diamonds["depth", "table"])
    diamonds_scaled = pl.concat(
        [diamonds.drop("depth", "table"), scaler.transform(diamonds["depth", "table"])],
        how="horizontal",
    )
    return (diamonds_scaled,)


@app.cell
def _(diamonds_scaled, pl):
    diamonds_dummies = pl.concat(
        [
            diamonds_scaled.drop("color", "clarity"),
            diamonds_scaled["color"].to_dummies(drop_first=True),
            diamonds_scaled["clarity"].to_dummies(drop_first=True),
        ],
        how="horizontal",
    )
    diamonds_dummies
    return (diamonds_dummies,)


@app.cell
def _(diamonds_dummies, outcome, pl, treatment):
    treated = diamonds_dummies.filter(pl.col(treatment) == 1)
    untreated = diamonds_dummies.filter(pl.col(treatment) == 0)
    features = diamonds_dummies.drop([treatment, outcome]).columns
    return (features,)


@app.cell
def _(mo):
    mo.md(r"""
    ## OLS-based effect
    We are able to detect a mean 2% (+/- 0.3%) price increase, seems quite low. 223 is the flat increment
    """)
    return


@app.cell
def _(diamonds, smf):
    price_regression = smf.ols(
        "np.log(price) ~ is_cut_premium_or_better + C(clarity)*C(color) + carat", data=diamonds.to_pandas()
    ).fit()
    price_regression.summary().tables[1]
    return


@app.cell
def _(diamonds, smf):
    price_regression_flat = smf.ols(
        "price ~ is_cut_premium_or_better + C(clarity)*C(color) + carat", data=diamonds.to_pandas()
    ).fit()
    price_regression_flat.summary().tables[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matching
    With matching keeping all the variables as confounders the increment in price is ~94 for Premium and Ideal cuts. Even lower.
    """)
    return


@app.cell
def _(diamonds_dummies, features):
    from dowhy import CausalModel

    model = CausalModel(
        data=diamonds_dummies.to_pandas(),
        treatment="is_cut_premium_or_better",
        outcome="price",
        common_causes=features
    )

    identified_estimand = model.identify_effect()
    return CausalModel, identified_estimand, model


@app.cell
def _(identified_estimand, model):
    estimate_k1 = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.distance_matching",
        target_units="att",                   # Average Treatment effect on the Treated
        method_params={
            "num_matches_per_unit": 1,        # 1-nearest-neighbour matching
            "distance_metric": "minkowski",   # Euclidean (p=2 default)
        },
    )
    return (estimate_k1,)


@app.cell
def _(estimate_k1):
    print(f"ATT estimate (k=1): {estimate_k1.value:,.0f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Propensity score
    We remove 'depth' because it greatly diminished positivity.
    We get a 79$ (+/- 15$) effect for premium and Ideal cuts.
    """)
    return


@app.cell
def _(features):
    propensity_features = [f for f in features if f not in {"depth"}]
    return (propensity_features,)


@app.cell
def _(LogisticRegression, diamonds_dummies, propensity_features, treatment):
    propensity_score_model = LogisticRegression(C=1e6).fit(diamonds_dummies[propensity_features], diamonds_dummies[treatment])
    return (propensity_score_model,)


@app.cell
def _(diamonds_dummies, propensity_features, propensity_score_model):
    diamonds_with_propensity = diamonds_dummies.with_columns(propensity_score=propensity_score_model.predict_proba(diamonds_dummies[propensity_features])[:, 1])
    return (diamonds_with_propensity,)


@app.cell
def _(diamonds_with_propensity, pl, treatment):
    weight_t = 1/diamonds_with_propensity.filter(pl.col(treatment) == 1)["propensity_score"]
    weight_nt = (1/(1-diamonds_with_propensity.filter(pl.col(treatment) == 1)["propensity_score"])).map_elements(lambda x: min(x, 20.0))
    print("Original Sample Size", diamonds_with_propensity.shape[0])
    print("Treated Population Sample Size", sum(weight_t))
    print("Untreated Population Sample Size", sum(weight_nt))
    return


@app.cell
def _(diamonds_with_propensity, px, treatment):
    fig = px.histogram(
        diamonds_with_propensity,   # Plotly Express needs pandas
        x="propensity_score",
        color=treatment,
        barmode="overlay",         # "overlay" or "stack" or "group"
        opacity=0.7,
        nbins=150,
        title="Positivity check",
    )
    fig.show()
    return


@app.cell
def _(diamonds_with_propensity, outcome, treatment):
    weight = ((diamonds_with_propensity[treatment]-diamonds_with_propensity["propensity_score"]) /
              (diamonds_with_propensity["propensity_score"]*(1-diamonds_with_propensity["propensity_score"])))

    ate = (weight * diamonds_with_propensity[outcome]).mean()

    print("ATE", ate)
    return


@app.cell
def _(
    LogisticRegression,
    diamonds_dummies,
    outcome,
    propensity_features,
    treatment,
):
    def run_ps(df, X, T, y, max_iter=1000, C=1e6):
        # estimate the propensity score
        ps = LogisticRegression(C=C, max_iter=max_iter).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    
        weight = (df[T]-ps) / (ps*(1-ps)) # define the weights
        return (weight * df[y]).mean() # compute the ATE

    # run 10 bootstrap samples (a real value should be much higher)
    bootstrap_samples = 10
    ates = [run_ps(diamonds_dummies.sample(fraction=1, with_replacement=True), propensity_features, treatment, outcome) for _ in range(bootstrap_samples)]
    return (ates,)


@app.cell
def _(ates, np):
    print(f"ATE: {np.mean(ates)}")
    print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matching with Propensity
    Going back to an estimate which is far higher, 149$ with a p-value of 0.06 (slightly larger than 0.05 which is what we were looking for)
    This implementation uses logistic regression by default and computes confidence intervals with bootstrapping (not clear if they are valid or not).
    """)
    return


@app.cell
def _(CausalModel, diamonds_dummies, propensity_features):
    mp_model = CausalModel(
        data=diamonds_dummies.to_pandas(),
        treatment="is_cut_premium_or_better",
        outcome="price",
        common_causes=propensity_features  # already excludes "depth"
    )

    mp_identified_estimand = mp_model.identify_effect(proceed_when_unidentifiable=True)

    mp_estimate_psm = mp_model.estimate_effect(
        mp_identified_estimand,
        method_name="backdoor.propensity_score_matching",
        target_units="ate",
        method_params={"num_matches_per_unit": 1},
        confidence_intervals=True,
    )

    print(f"ATE (propensity score matching): {mp_estimate_psm.value:,.2f}")
    return mp_estimate_psm, mp_identified_estimand


@app.cell
def _(model, mp_estimate_psm, mp_identified_estimand):
    # not clear if this ok for matching estimator
    refute = model.refute_estimate(
        mp_identified_estimand,
        mp_estimate_psm,
        method_name="bootstrap_refuter",
        num_simulations=20
    )
    print(refute)

    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
