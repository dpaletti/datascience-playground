import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import statsmodels.formula.api as smf
    import polars as pl
    import altair as alt

    return alt, pl, smf


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading
    """)
    return


@app.cell
def _(pl):
    lead_scoring_df = pl.read_csv("data/Lead Scoring.csv")
    return (lead_scoring_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Regression on random data
    """)
    return


@app.cell
def _(lead_scoring_df, pl):
    cleaned_lead_scoring_df = lead_scoring_df.with_columns(
        pl.col("Do Not Email")
        .map_elements(lambda x: 1 if x == "Yes" else 0)
        .alias("do_not_email")
    )
    return (cleaned_lead_scoring_df,)


@app.cell
def _():
    outcome = "Converted"
    treatment = "do_not_email"
    return outcome, treatment


@app.cell
def _(cleaned_lead_scoring_df, outcome, smf, treatment):
    regression_model = smf.ols(
        f"{outcome} ~ {treatment}", data=cleaned_lead_scoring_df.to_pandas()
    ).fit()
    return (regression_model,)


@app.cell
def _(regression_model):
    regression_model.summary().tables[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ATE=-0.24 wit 95% CI [-0.280, -0.208] and P-value ~ 0
    We can be 95% sure that the true average treatment effect of do_not_email falls between -0.280 and -0.208
    """)
    return


@app.cell
def _(cleaned_lead_scoring_df, pl):
    # The intercept coefficient is simply the observed sample mean for no-treatment
    # E[Y|T=0]
    sample_mean_not_treated = cleaned_lead_scoring_df.filter(
        ~pl.col("do_not_email").cast(pl.Boolean)
    )["Converted"].mean()

    grouped_mean = cleaned_lead_scoring_df.group_by("Do Not Email").agg(
        pl.col("Converted").mean()
    )

    # E[Y|T=1] - E[Y|T=0]
    sample_difference_in_means = (
        grouped_mean.filter(pl.col("Do Not Email") == "Yes")["Converted"][0]
        - grouped_mean.filter(pl.col("Do Not Email") == "No")["Converted"][0]
    )
    return sample_difference_in_means, sample_mean_not_treated


@app.cell
def _(sample_difference_in_means, sample_mean_not_treated):
    # Y is the observed outcome
    print(f"E[Y|T=0] (= intercept coef): {sample_mean_not_treated}")
    print(f"E[Y|T=1] - E[Y|T=0] (= treatment coef): {sample_difference_in_means}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regression on non-random data
    """)
    return


@app.cell
def _(lead_scoring_df, pl):
    non_random_lead_scoring_df = lead_scoring_df.select(
        *[
            pl.col(col).alias(col.lower().replace(" ", "_"))
            for col in lead_scoring_df.columns
        ]
    )
    return (non_random_lead_scoring_df,)


@app.cell
def _(non_random_lead_scoring_df, smf):
    # ols on binary outcomes gives additive effects
    # if the outcome is non-binary and does not contain zeros one can do 'np.log(y) ~ x' to get additive effects
    non_random_rm = smf.ols(
        "converted ~ total_time_spent_on_website",
        data=non_random_lead_scoring_df.to_pandas(),
    ).fit()
    return (non_random_rm,)


@app.cell
def _(non_random_rm):
    non_random_rm.summary().tables[1]
    return


@app.cell
def _(alt, non_random_lead_scoring_df, pl):
    import marimo as mo

    mo.ui.altair_chart(
        alt.Chart(
            non_random_lead_scoring_df.sort(by="total_time_spent_on_website")
            .with_columns(
                pl.col("total_time_spent_on_website"),
                pl.col("converted").rolling_mean(window_size=100),
            )
            .sample(2000)
        )
        .mark_line()
        .encode(x="total_time_spent_on_website", y="converted")
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    Here the finding seem rather contradicting (in part because this data is quite fake). We are going now to treat this data as non-random. The regression is representative only of one feature but we have many others, now we can add them,
    """)
    return


@app.cell
def _(non_random_lead_scoring_df, smf):
    control_variables = ["page_views_per_visit", "totalvisits"]
    full_non_random_rm = smf.ols(
        f"converted ~ total_time_spent_on_website + {' + '.join(control_variables)}",
        data=non_random_lead_scoring_df.to_pandas(),
    ).fit()
    return (full_non_random_rm,)


@app.cell
def _(full_non_random_rm):
    full_non_random_rm.summary().tables[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    here we were able to account not only for the treatment but also for the cofounding variable which should influence both the treatment and the outcome. If we assume a random controlled trial this treatment is random and so cannot cause/be caused by anything. The regression then, plays the part of breaking the dependence between outcome and cofounding variables by computing the treatment effect keeping them fixed.
    """)
    return


if __name__ == "__main__":
    app.run()
