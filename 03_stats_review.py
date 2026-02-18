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
    normal_distribution_df = pl.DataFrame({"outcomes": normal_outcomes, "pdf": normal_pdf})
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
    ci_95 = (mean - 2 * std_dev, mean + 2 * std_dev)
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
    ci_z = (mean - z * std_dev, mean + z * std_dev)
    ci_z
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hypothesis testing
    Test if the mean of the total number of visits is statistically different
    """)
    return


@app.cell
def _(lead_scoring_df, pl):
    converted_df = lead_scoring_df.filter(pl.col("Converted").cast(pl.Boolean))
    not_converted_df = lead_scoring_df.filter(~pl.col("Converted").cast(pl.Boolean))
    return converted_df, not_converted_df


@app.cell
def _(converted_df, not_converted_df, np):
    mean_diff: float = (
        converted_df["TotalVisits"].mean() - not_converted_df["TotalVisits"].mean()  # type: ignore
    )
    standard_error_difference: float = np.sqrt(
        converted_df["TotalVisits"].var() / len(converted_df)  # type: ignore
        + not_converted_df["TotalVisits"].var() / len(not_converted_df)  # type: ignore
    )
    ci = (
        mean_diff - 1.96 * standard_error_difference,
        mean_diff + 1.96 * standard_error_difference,
    )
    return ci, mean_diff, standard_error_difference


@app.cell
def _(ci):
    ci
    return


@app.cell
def _(mo):
    mo.md(r"""
    Computing this 95% CI we can safely say that we are 95% confident that the true difference between the converted and non-converted group total visits falls between 0.08 e 0.51
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    When testing specific hypothesis we go to z statistics, in particular here we want to test the null hypothesis that the difference between the two means is 0
    """)
    return


@app.cell
def _(mean_diff: float, standard_error_difference: float):
    z = mean_diff / standard_error_difference
    z
    return (z,)


@app.cell
def _(mo):
    mo.md(r"""
    Here we can see that this is at more than 2 standard deviations from the mean (0) which means that we are observing a significant difference, that is we can reject the hypothesis that the two means are equal wih 95% confidence. From the z-value is easy to get the p-value, that is the probability of observing data given the null hypothesis is true. To estimate it from the z-statistics we use the survival function which for z on the left of the mean is CDF(x) while on the right is 1 - CDF(x).
    """)
    return


@app.cell
def _(norm, z):
    norm.sf(z)
    return


if __name__ == "__main__":
    app.run()
