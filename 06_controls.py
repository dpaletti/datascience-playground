import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Controls
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    import statsmodels.formula.api as smf

    alt.data_transformers.enable("vegafusion")
    return alt, pl, smf


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading
    """)
    return


@app.cell
def _(pl):
    lead_scoring_df = pl.read_csv("./data/Lead Scoring.csv")
    return (lead_scoring_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Good controls
    """)
    return


@app.cell
def _(lead_scoring_df, pl):
    prepared_df = lead_scoring_df.with_columns(
        pl.col("Total Time Spent on Website").alias("total_time_spent_on_website"),
        pl.col("Lead Origin").alias("lead_origin"),
    ).with_columns(
        lead_scoring_df["Lead Origin"].to_dummies(drop_first=True),
        email=(
            ~(
                pl.col("Do Not Email")
                .replace({"No": 0, "Yes": 1})
                .cast(pl.UInt8)
                .cast(pl.Boolean)
            )
        ).cast(pl.UInt8),
    )
    prepared_df
    return (prepared_df,)


@app.cell
def _():
    outcome = "total_time_spent_on_website"
    treatment = "email"
    control = "lead_origin"
    return control, outcome, treatment


@app.cell
def _(mo):
    mo.md(r"""
    On the plot above we see that `outcome` has high variance wrt `treatment` this means that it is insufficient on explaining `outcome`
    """)
    return


@app.cell
def _(alt, outcome, prepared_df, treatment):
    alt.Chart(prepared_df).mark_circle().encode(x=treatment, y=outcome)
    return


@app.cell
def _(alt, outcome, pl, prepared_df, treatment):
    alt.Chart(
        prepared_df.filter(pl.col("Lead Origin_Landing Page Submission") == 1)
    ).mark_circle().encode(x=treatment, y=outcome)
    return


@app.cell
def _(outcome, prepared_df, smf, treatment):
    treatment_only_model = smf.ols(
        f"{outcome} ~ {treatment}", data=prepared_df.to_pandas()
    ).fit()
    treatment_only_model.summary().tables[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    Below we show that adding a categorical control helps in identifying the effect of `treatment` on `outcome`
    """)
    return


@app.cell
def _(control, outcome, prepared_df, smf, treatment):
    control_model = smf.ols(
        f"{outcome} ~ {treatment} + C({control})", data=prepared_df.to_pandas()
    ).fit()
    control_model.summary().tables[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    Adding too many categories loses performance probably country does not influence much
    """)
    return


@app.cell
def _(control, outcome, prepared_df, smf, treatment):
    control_large_model = smf.ols(
        f"{outcome} ~ {treatment} + C({control}) + C(Country)", data=prepared_df.to_pandas()
    ).fit()
    control_large_model.summary().tables[1]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
