import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 03. Stats review
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
    import marimo as mo

    return mo, pl


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
    ## Analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### standard deviation
    """)
    return


@app.cell
def _(lead_scoring_df):
    lead_scoring_df["Converted"].std()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### confidence intervals
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
