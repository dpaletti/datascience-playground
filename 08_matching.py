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
    from dowhy.causal_estimators.distance_matching_estimator import DistanceMatchingEstimator
    from dowhy.causal_identifier import IdentifiedEstimand


    return MinMaxScaler, pl, smf


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

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    print("Common causes seen by DoWhy:", model.get_common_causes())

    return identified_estimand, model


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


if __name__ == "__main__":
    app.run()
