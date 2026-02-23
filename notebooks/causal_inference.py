import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Causal inference
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This notebook contains practice code for [Causal inference for the brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html).

    I am using [Hillstrom's dataset](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) for email campaigns to assess the methods.

    This dataset contains 64,000 customers who last purchased within twelve months. The customers were involved in an e-mail test.
    - 1/3 were randomly chosen to receive an e-mail campaign featuring Mens merchandise.
    - 1/3 were randomly chosen to receive an e-mail campaign featuring Womens merchandise.
    - 1/3 were randomly chosen to not receive an e-mail campaign.
    During a period of two weeks following the e-mail campaign, results were tracked.

    Historical customer attributes:
    - recency: months since last purchase.
    - history_segment (7 segments): categorization of dollars spent in the past year.
    - history: actual dollar value spent in the past year.
    - mens (1/0 indicator): 1 = customer purchased mens merchandise in the past year.
    - womens (1/0 indicator): 1 = customer purchased Womens merchandise in the past year.
    - zip_code (1/0 indicator): classifies zip code as Urban, Surburban, or Rural.
    - newbie (1/0 indicator): 1 = New customer in the past twelve months.
    - channel ("Phone", "Web", "Multichannel"): describes the channels the customer purchased from in the past year.
    - **segment** (Mens E-Mail, Womens E-Mail, No E-Mail): the email that the customer received

    **Overarching question**: Was the e-mail campaign succesfull?

    **Questions**:
    - Which e-mail campaign performed the best, the Mens version, or the Womens version?
    - How much incremental sales per customer did the Mens version of the e-mail campaign drive? How much incremental sales per customer did the Womens version of the e-mail campaign drive?
    - If you could only send an e-mail campaign to the best 10,000 customers, which customers would receive the e-mail campaign? Why?
    - If you had to eliminate 10,000 customers from receiving an e-mail campaign, which customers would you suppress from the campaign? Why?
    - Did the Mens version of the e-mail campaign perform different than the Womens version of the e-mail campaign, across various customer segments?
    - Did the campaigns perform different when measured across different metrics, like Visitors, Conversion, and Total Spend?
    - Did you observe any anomalies, or odd findings?
    - Which audience would you target the Mens version to, and the Womens version to, given the results of the test? What data do you have to support your recommendation?
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
    import polars.selectors as cs
    import marimo as mo

    alt.data_transformers.enable("vegafusion")
    return alt, cs, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading
    """)
    return


@app.cell
def _(pl):
    # compiled from https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
    schema = {
        "channel": pl.Enum(["Phone", "Web", "Multichannel"]),
        "conversion": pl.UInt8,  # 0, 1 indicator
        "history": pl.Float32,
        "history_segment": pl.Enum(
            [
                "1) $0 - $100",
                "2) $100 - $200",
                "3) $200 - $350",
                "4) $350 - $500",
                "5) $500 - $750",
                "6) $750 - $1,000",
                "7) $1,000 +",
            ]
        ),
        "mens": pl.UInt8,  # 0, 1 indicator
        "newbie": pl.UInt8,  # 0, 1 indicator
        "recency": pl.UInt64,
        "segment": pl.Enum(categories=["Womens E-Mail", "No E-Mail", "Mens E-Mail"]),
        "spend": pl.Float32,
        "visit": pl.UInt8,  # 0, 1 indicator
        "womens": pl.UInt8,  # 0, 1 indicator
        "zip_code": pl.Enum(["Urban", "Rural", "Surburban"]),
    }
    return (schema,)


@app.cell
def _(pl, schema):
    raw_data = pl.read_csv(
        "../data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        schema_overrides=schema,
    )
    return (raw_data,)


@app.cell
def _(raw_data):
    raw_data
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Preliminary Analysis
    """)
    return


@app.cell
def _(alt, cs, mo, pl, raw_data):
    bar_size = 20
    bar_padding = 0.5
    plot_width = 200
    plots_per_row = 3

    plotting_data = raw_data.with_columns(cs.by_dtype(pl.Enum).cast(pl.String))
    charts = []
    for col in plotting_data.columns:
        charts.append(
            alt.Chart(plotting_data)
            .mark_bar(size=bar_size)
            .encode(
                x=alt.X(
                    col,
                    axis=alt.Axis(labelAngle=-45),
                    scale=alt.Scale(paddingInner=0.4),
                ),
                y="count()",
            )
            .properties(width=plot_width)
        )

    mo.ui.altair_chart(alt.concat(*charts, columns=plots_per_row))
    return


if __name__ == "__main__":
    app.run()
