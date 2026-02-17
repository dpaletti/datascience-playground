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
    from scipy.stats import norm
    import numpy as np
    import altair as alt

    return alt, norm, np, pl


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
def _(alt, norm, np, pl):
    # generate normal distribution data given inverse CDF, called percent point function
    # the CDF is a function s.t. CDF_X(x) = Pr(X <= x)
    # the PPF is a function s.t. PPF_(pr) = x, given a probability it returns the realisation of the distribution
    # if the inverse does not exist the smaller possible value at the given probability is selected to create a generalised PPF
    # finally we compute the probability density function given the outcomes, that is the probability with which each outcome happens
    mean = 0
    std_dev = 1
    custom_norm = norm(loc=mean, scale=std_dev)
    normal_outcomes = np.linspace(custom_norm.ppf(0.001), custom_norm.ppf(0.999), num=100)
    normal_pdf = norm.pdf(normal_outcomes)
    normal_distribution_df = pl.DataFrame({"outcomes":normal_outcomes, "pdf": normal_pdf})
    alt.Chart(normal_distribution_df).mark_bar().encode(x="outcomes", y="pdf")
    return custom_norm, mean, std_dev


@app.cell
def _(mo):
    mo.md(r"""
    Here we build a 95% confidence interval for the true mean

    This means (strictly speaking) that if we were to repeat this experiment 100 times 95 out of 100 the mean would fall here

    In practice we would say that this is the interval in which we have 95% probability to find the mean
    """)
    return


@app.cell
def _(mean, std_dev):
    ci_95 = (mean - 2 * std_dev, mean + 2 *std_dev)
    ci_95
    return


@app.cell
def _(mo):
    mo.md(r"""
    To create a custom interval we simply need to select the correct `z` value.

    In fact, the two is approximately what comes out the inverse CDF at 0.95, but if we want we can go higher at 0.995
    """)
    return


@app.cell
def _(custom_norm):
    z = custom_norm.ppf(0.995)
    z
    return (z,)


@app.cell
def _(mean, std_dev, z):
    ci_z= (mean - z * std_dev, mean + z *std_dev)
    ci_z
    return


@app.cell
def _(mo):
    mo.md(r"""
    #TODO: start from hypothesis testing
    """)
    return


if __name__ == "__main__":
    app.run()
