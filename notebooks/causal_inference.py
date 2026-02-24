import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Causal inference
    """)
    return


@app.cell(hide_code=True)
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

    Attributes describing customer behaviour after the e-mail campaign:
    - visit (1/0 indicator): customer visited website in the following two weeks
    - conversion (1/0 indicator): customer purchased merchandise in the following two weeks
    - spend (1/0 indicator): actual dollar spent in the following two weeks

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    import altair as alt
    import polars.selectors as cs
    import marimo as mo
    import statsmodels.formula.api as smf

    alt.data_transformers.enable("vegafusion")
    return alt, cs, mo, pl, smf


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preliminary Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - recency: bimodal distribution, customers tend to come back after 9, 10 months
    - history_segment: power law, there are more customers with low spending than customers whith high spending (the intervals have left-side closed)
    - history: very concentrated on low spending, most of the customers (95%) spend less than $561 and 75% of the customers are under $325, this is quite clear from the segment representation while from the history itself is not very clear as the maximum is at $3.4k (which is not to be considered an outlier but just a rare case)
    - mens, womens: balanced representation of men and women merchandise spending
    - zip code: slight under-representation of rural areas but in general prettu balanced
    - newbie: balanced representation of new and old (more than 1y maturity) customers
    - channel: slight under representation of multi-channel customers
    - segment: treatment is uniform among mens and womens, we also have untreated

    As expected after the campaign we have a noticeable amount of visits (not necessarily driven by the campaign), and as we would expect a small subset of these visits finalized a buy (conversion).

    **Preliminary cleaning**:
    - we bin the spend column with the same bins of history segment
    - we clean the bins so that history and spend segment bins are the same and explicit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    History has 90% of the values represented under 561 but then values shoot up to 3345
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Spend has 99% of the values at 0. Among the remaining ones we see that 59% is between 0 and 100, 24% between 100 and 200, 10% between 200 and 350 and a residual 5% between 350 and 500. As we would expect, spend is 0 when conversion is 0. With respect to the usual spending we see a bit less spending, this is usually to be expected.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell
def _(pl, raw_data):
    history_segment_mapping = {
        "1) $0 - $100": "(0, 100)",
        "2) $100 - $200": "[100, 200)",
        "3) $200 - $350": "[200, 350)",
        "4) $350 - $500": "[350, 500)",
        "5) $500 - $750": "[500, 750)",
        "6) $750 - $1,000": "[750, 1000)",
        "7) $1,000 +": "1000+",
    }
    prepared_data = raw_data.with_columns(
        spend_segment=pl.when(pl.col("spend") != 0)
        .then(pl.col("spend").cut([0.1, 100, 200, 350, 500], left_closed=True))
        .otherwise(pl.lit("=0"))
        .replace({"[0.1, 100)": "(0, 100)"})
        .cast(pl.Enum(["=0", "(0, 100)", "[100, 200)", "[200, 350)", "[350, 500)"])),
        history_segment=pl.col("history_segment")
        .cast(pl.String)
        .replace(history_segment_mapping)
        .cast(pl.Enum(list(history_segment_mapping.values()))),
    )
    prepared_data
    return (prepared_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graphical causal model

    Treatment (segment) is uniform so segment has no causes.

    **Treatment effect assumption**: there is some relation between receiving the email and subsequent visit, conversion and spend

    [segment] -> visit/conversion/spend

    This will be tested later down the line because there other competing variables we should use as controls.

    **Heterogeneous gender effect assumption**: receiving a mens/womes e-mail having previously bought mens/womens items should have higher effect on conversion, spend and visit

    [segment] • [mens] • [women] -> visit/conversion/spend


    **Heterogeneous history segment effect assumption**: having spent more in the past year leads to a higher spend when coming back and a higher probability of coming back, we can for now try to work with segments and ignore actual spend, we will use it to check the association with zip_code

    [segment] • [history_segment] -> visit/conversion/spend

    **(not detected) Zip code effect**: zip code may easily be related to history_segment and channel, as the place you live may be related to both spending power (also in this shop) and how well internet works

    [zip_code] -> [history_segment]
    [zip_code] -> [channel]

    Running OLS (see below) for these relations we do not find particular relationships between these two variables

    **newbie effect**: we expect new customers to be coming preferably from the web and in general have a lower spending

    [newbie] -> [channel]
    - channel = "Multichannel", coef=-1.4144 +/- 0.26 with p-value ~0 and intercept -1.5: new customers tend to buy both from internet and phone (reasonable, data comes from 2006)

    [newbie] -> [history_segment]
    - we are able to detect a reliable negative effect for history segments 1 to 3 that pushes newbie into the 0 category the one from (0, 100). From here we can preliminary hypothesize that newbies tend to spend less

    **heterogeneous recency effect**: months since last purchase for sure is expected to correlate with visit/conversion/spend

    [segment] • [recency] -> visit/conversion/spend

    **correlations**:
    To probe for other relationships we ran a correlation matrix and we found that someone who buys male products does not buy women ones and viceversa. In particular 10% of the customers buy both, all the others buy either male or female. This leads to **restructuring that column in 3-category columns product_gender (male, female, both)**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Testing zip code effect on history_segment and channel (no evidence)
    """)
    return


@app.cell
def _(pl, prepared_data, smf):
    history_zip_code = smf.mnlogit(
        "history_segment_int ~ C(zip_code)",
        data=prepared_data.with_columns(
            history_segment_int=pl.col("history_segment").to_physical()
        ).to_pandas(),
    ).fit()
    history_zip_code.summary().tables[1]
    return


@app.cell
def _(pl, prepared_data, smf):
    channel_zip_code = smf.mnlogit(
        "channel_int ~ C(zip_code)",
        data=prepared_data.with_columns(
            channel_int=pl.col("channel").to_physical()
        ).to_pandas(),
    ).fit()
    channel_zip_code.summary().tables[1]
    return


@app.cell
def _(pl, prepared_data, smf):
    zip_code_channel_history = smf.mnlogit(
        "zip_code_int ~ C(channel) + C(history_segment)",
        data=prepared_data.with_columns(
            zip_code_int=pl.col("zip_code").to_physical()
        ).to_pandas(),
    ).fit()
    zip_code_channel_history.summary().tables[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Testing newbie effect on channel and history_segment
    """)
    return


@app.cell
def _(pl, prepared_data):
    prepared_data.with_columns(channel_int=pl.col("channel").to_physical())[
        "channel", "channel_int"
    ].unique()
    return


@app.cell
def _(pl, prepared_data, smf):
    channel_newbie = smf.mnlogit(
        "channel_int ~ newbie",
        data=prepared_data.with_columns(
            channel_int=pl.col("channel").to_physical()
        ).to_pandas(),
    ).fit()

    channel_newbie.summary().tables[1]
    return


@app.cell
def _(pl, prepared_data):
    prepared_data.with_columns(
        history_segment_int=(pl.col("history_segment").to_physical() + 1).replace({7: 0})
    )["history_segment", "history_segment_int"].unique().sort("history_segment_int")
    return


@app.cell
def _(pl, prepared_data, smf):
    history_segment_newbie = smf.mnlogit(
        "history_segment_int ~ newbie",
        data=prepared_data.with_columns(
            history_segment_int=(pl.col("history_segment").to_physical())
        )
        .filter(pl.col("history_segment_int") <= 4)
        .to_pandas(),
    ).fit()

    history_segment_newbie.summary().tables[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Correlations
    """)
    return


@app.cell
def _(cs, pl, prepared_data):
    correlations = prepared_data.select(cs.numeric()).corr()
    correlations_long = correlations.with_columns(
        pl.Series("col1", correlations.columns)
    ).unpivot(index="col1", variable_name="col2")

    return (correlations_long,)


@app.cell
def _(alt, correlations_long):
    alt.layer(
        alt.Chart(correlations_long)
        .mark_rect()
        .encode(
            x=alt.X("col1:O"),
            y=alt.Y("col2:O"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        ),
        alt.Chart(correlations_long)
        .mark_text(fontSize=10)
        .encode(
            x=alt.X("col1:O"),
            y=alt.Y("col2:O"),
            text=alt.Text("value:Q", format=".2f"),
            color=alt.condition(
                "datum.value > 0.5 || datum.value < -0.5",
                alt.value("white"),
                alt.value("black"),
            ),
        ),
    ).properties(width=alt.Step(50), height=alt.Step(50))

    return


@app.cell
def _(pl, prepared_data):
    prepared_data.with_columns(
        mens_womens=pl.concat_str([pl.col("mens"), pl.col("womens")], separator=",")
    )["mens_womens"].value_counts(normalize=True)
    return


@app.cell
def _(pl, prepared_data):
    cleaned_data = prepared_data.with_columns(
        product_gender=pl.when(pl.col("mens") == 1, pl.col("womens") == 1)
        .then(pl.lit("both"))
        .otherwise(
            pl.when(pl.col("mens") == 1).then(pl.lit("woman")).otherwise(pl.lit("man"))
        )
    ).drop(["mens", "womens"])

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## OLS effect estimation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
