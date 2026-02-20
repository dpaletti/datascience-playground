import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Diamonds analysis
    Applying causal inference techniques to the diamond dataset
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
    from collections import defaultdict

    alt.data_transformers.enable("vegafusion")
    return alt, defaultdict, pl


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading
    """)
    return


@app.cell
def _(pl):
    diamonds_df = pl.read_csv("data/diamonds.csv").drop("")
    return (diamonds_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Preliminary analysis
    - carat (aka weight)
    - cut: shape given to the diamond after harvesting
    - color
    - clarity
    - depth:  z / mean(x, y)
    - table: width of the top of the diamond relative to the widest point
    - x, y, z: length, width, depth

    outcome: price
    """)
    return


@app.cell
def _():
    cut_quality_scale = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    clarity_scale = ["SI1", "SI2", "I1", "IF", "VVS1", "VVS2", "VS1", "VS2"]
    return clarity_scale, cut_quality_scale


@app.cell
def _(diamonds_df):
    diamonds_df
    return


@app.cell
def _(alt, defaultdict, diamonds_df, pl):
    colors = [
        "#008f7a",
        "#2c73d2",
        "#4ffbdf",
        "#845ec2",
        "#d65db1",
        "#ff6f91",
        "#ff9671",
        "#ffc75f",
        "#f9f871",
    ]
    charts = defaultdict(list)
    plot_width = 350
    charts_per_row = 3
    padding_categorical_charts = 0.15
    vertical_spacing = 50
    horizontal_spacing = 40

    for col_num, col in enumerate(diamonds_df.columns):
        bar_color = colors[col_num % len(colors)]
        if diamonds_df[col].dtype == pl.String:
            chart = (
                alt.Chart(diamonds_df)
                .mark_bar(color=bar_color)
                .encode(
                    alt.X(
                        col,
                        sort="-y",
                        scale=alt.Scale(paddingInner=padding_categorical_charts),
                        axis=alt.Axis(labelAngle=-360),
                    ),
                    y="count()",
                    tooltip=[alt.Tooltip(col), alt.Tooltip("count()", title="Count")],
                )
                .properties(title=col.upper(), width=plot_width)
            )
        else:
            chart = (
                alt.Chart(diamonds_df)
                .mark_bar(color=bar_color)
                .encode(
                    alt.X(col, bin=True),
                    y="count()",
                    tooltip=[
                        alt.Tooltip(col, bin=True, title=f"{col} range"),
                        alt.Tooltip("count()", title="Count"),
                    ],
                )
                .properties(title=col.upper(), width=plot_width)
            )

        charts[f"row{col_num // charts_per_row}"].append(chart)

    rows = [alt.hconcat(*charts, spacing=horizontal_spacing) for charts in charts.values()]
    alt.vconcat(*rows, spacing=vertical_spacing).configure_axis(
        titleFontSize=16, labelFontSize=14
    ).configure_title(fontSize=22)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Verify multicollinearity
    We notice that, as one would expect, X, Y Z are heavily correlated with each other.
    We remove them as we have depth and table that are reasonable proxies
    """)
    return


@app.cell
def _(alt, diamonds_df):
    cols = ["depth", "table", "x", "y", "z"]
    corr_df = (
        diamonds_df.select(cols)
        .to_pandas()  # convert to pandas (if using Polars)
        .corr()
        .abs()
        .reset_index()
        .melt(id_vars="index")
    )
    corr_df.columns = ["var1", "var2", "correlation"]

    base = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X("var1:O", title=None),
            y=alt.Y("var2:O", title=None),
            color=alt.Color("correlation:Q"),
            tooltip=["var1", "var2", alt.Tooltip("correlation:Q", format=".2f")],
        )
        .properties(title="Correlation Heatmap", width=300, height=300)
    )
    heatmap = base.mark_rect().encode(color=alt.Color("correlation:Q"))


    labels = base.mark_text(fontSize=12).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.correlation) > 0.7",  # dark squares get white text
            alt.value("white"),
            alt.value("black"),
        ),
    )

    (heatmap + labels).properties(title="Correlation Heatmap", width=300, height=300)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature interactions
    1. Higher carat diamonds do NOT seem to be cut in better ways necessarily
    2. lower quality cuts do NOT necessarily mean larger table or smaller depths
    """)
    return


@app.cell
def _(clarity_scale, cut_quality_scale, diamonds_df, pl):
    import seaborn as sns

    sns.pairplot(
        diamonds_df.with_columns(
            cut=pl.col("cut").replace(
                {cut: num for num, cut in enumerate(cut_quality_scale)}
            ).cast(pl.UInt8),
            clarity=pl.col("clarity").replace(
                {clarity: num for num, clarity in enumerate(clarity_scale)}
            ).cast(pl.UInt8)
        )
        .select(pl.all().exclude(["x", "y", "z"]))
        .to_pandas()
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
