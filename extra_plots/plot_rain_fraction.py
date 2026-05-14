# plot quantiles of rain fraction
"""Plot monthly rain-fraction quantiles for one or more processing directories.

Module-level constants below define the input locations, quantiles to compute,
plot layout choices, and cache behavior. Update those values to adapt this
script to new sites or plotting preferences without changing function logic.

Quick config guide:
- Fast cached run: keep `FORCE_RECOMPUTE_CACHE = False` and rerun. Cached
  monthly quantiles are reused when source files are unchanged.
- Fresh recompute run: set `FORCE_RECOMPUTE_CACHE = True` when you change
  quantile or cache logic; this bypasses cache reads and refreshes cache files.
- Cache location: cache files are written under `ROOT_DIR / CACHE_DIR_NAME`
  (default `_cache_monthly_quantiles`).
- Cache lifecycle: each directory either loads a fresh cache file (cache hit)
  or recomputes and writes a new one (cache miss/stale cache).
- Quantile color mapping: tune `QUANTILE_COLOR_LOW` and
  `QUANTILE_COLOR_HIGH` (range `0 <= low < high <= 1`) to control the
  blue shading; higher quantiles are mapped to darker blue.
- To change the data-processing logic while keeping the cache/load/GUI
  infrastructure, define a new `ProcessingSpec` and attach it to the config.
- To change the panel rendering while keeping figure/layout/export logic,
  define a new `PlotSpec` and attach it to the config.


Non-GUI run (default): edit the DEFAULT_* constants and flags at the top
of this file, then run as normal.

GUI run: set USE_GUI = True at the top of this file.  A small form will
pop up before any data is loaded, letting you change root directory,
locations, conversion, quantiles, and save/cache options interactively.
--------------------------

AI note:
- This script was AI-generated and then iteratively refined in-editor.
- Prompting summary: build maintainable code to compute monthly rain-fraction
  quantiles, plot one shared-axis subplot per key, reuse the same matplotlib
  figure, add small time-axis margins, support optional PNG export, and cache
  computed quantiles with a manual recompute switch.
- Further prompting summary:
  - Refined plotting layout so suptitle/legend spacing is compact and stable
    across multiple subplots.
  - Refactored cache flow to separate filename construction, freshness checks,
    cache read/write decisions, and pure quantile computation.
  - Added optional GUI-driven configuration (Tkinter) for key runtime settings
    such as root directory, locations, conversion, quantiles, and save/cache
    options.
  - Added quantile color-range controls so higher quantiles map to darker blue,
    with configurable low/high bounds and GUI validation.
  - Added configurable sample-count overlay controls (`count_scale` and
    `bar_width_days`) and synchronized secondary-axis limits across all panels
    for direct visual comparison.
  - Added configurable figure sizing (`fig_size`) via defaults and GUI.
  - Added mosaic-driven layout configuration (`site_mosaic`) with `BLANK`
    placeholders, deriving processed sites from non-blank entries.
  - Updated GUI mosaic editing to support one sub-list per line (with
    backward-compatible parsing for full list-of-lists input).
  - Added maintainability documentation throughout to clarify data flow,
    plotting logic, and cache behavior.
"""

import hashlib
import json
import pathlib
import typing
import ast
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import xarray
# imports used for typing
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib import colors as mcolors
from matplotlib import dates as mdates

import ausLib
from ausLib import data_dir # root for data

USE_GUI = True # If True use GUI to set values.
# Base folder containing one subdirectory per entry in DIRECTORIES.
ROOT_DIR = data_dir/"rain_indices"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# Default panel mosaic. Use "BLANK" to leave an empty slot.
DEFAULT_SITE_MOSAIC = [
    ["Mornington", "Cairns", "BLANK"],
    ["Grafton", "Brisbane", "Gladstone"],
    ["Canberra", "Sydney", "Newcastle"],
    ["Adelaide", "Wtakone", "Melbourne"],
]
# Default reflectivity-to-rain conversion label; used to build directory names.
CONVERSION = 'melbourne'
# Monthly quantiles to compute and plot.
QUANTILES = [0.9,  1.0]
# Min number of radar samples in a period.
MIN_SAMPLES_PER_MONTH = 2000 # complete ten-minute data would be about 7000.
FIGURE_NUM = "rain_fraction_quantiles"
# Fractional x-axis padding added at both left and right plot boundaries.
X_MARGIN_FRACTION = 0.02
# Name of the cache folder created under ROOT_DIR for computed monthly quantiles.
CACHE_DIR_NAME = "_cache_monthly_quantiles"
# If True, skip cache reads and recompute quantiles from source files.
FORCE_RECOMPUTE_CACHE = False
# If True, write the plot to SCRIPT_DIR/figures as a PNG.
SAVE_FIG = True

# Figure-level layout tuning for title/legend spacing.
TOP_MARGIN = 0.01
AXES_TO_LEGEND_GAP = 0.008
LEGEND_TO_TITLE_GAP = 0.006

# Figure size for the output plot (width, height in inches).
# Will be scaled based on number of sites if using dynamic layout.
DEFAULT_FIG_SIZE: tuple[float, float] = (12.0, 8.0)
DEFAULT_Y_LIMITS: tuple[float, float] = (0.0, 1.0)
DEFAULT_GUI_CONFIG_FILE = SCRIPT_DIR / "plot_rain_fraction_gui_config.json"

# Secondary-axis defaults for sample-count overlay.
COUNT_SCALE_DEFAULT = 1000.0
BAR_WIDTH_DAYS_DEFAULT = 80.0

# Line labels/styles used for each quantile in QUANTILES.
QUANTILE_STYLES = {
    0.9: {"label": "90%", "color":"red"},
    0.99: {"label": "99%", "color":"skyblue" },
    1.0: {"label": "Max", "color":"blue"},
}
# Blues colormap range used when auto-generating quantile colors.
# 0.0 = white, 1.0 = darkest blue; clipping avoids near-white lines.
QUANTILE_COLOR_LOW = 0.35
QUANTILE_COLOR_HIGH = 0.90


def flatten_mosaic_locations(site_mosaic: list[list[str]]) -> list[str]:
    """Return ordered non-BLANK site names from a mosaic."""
    return [site for row in site_mosaic for site in row if site != "BLANK"]


def format_mosaic_lines(site_mosaic: list[list[str]]) -> str:
    """Format a mosaic as one Python sub-list per line for GUI editing."""
    return "\n".join(repr(row) for row in site_mosaic)


@dataclass(frozen=True)
class ProcessingSpec:
    """Describe how raw files are turned into processed/cached datasets.

    This specification is used in the live code path:
    - `PlotRainFractionConfig.directories` calls `directory_name(...)`
    - `get_directory_quantiles(...)` uses `file_glob` and `token()`
    - `plot_quantiles(...)` maps panel/site names via `directory_name(...)`

    Attributes
    ----------
    name : str
        Human-readable processing name (for logging/identification), and default
        fallback token when `cache_token` is not provided.
    compute_from_files : Callable
        Function that performs the core transformation from source files to the
        processed dataset. Signature:
        `(files: list[pathlib.Path], config: PlotRainFractionConfig) -> xarray.Dataset`.
    directory_name_fn : Callable
        Function that maps a logical location/site name plus config into the
        source-directory key used for loading and plotting. Signature:
        `(location: str, config: PlotRainFractionConfig) -> str`.
    file_glob : str, optional
        Glob pattern used to list source files inside each location directory.
        Default is `"*.nc"`.
    cache_token : str | None, optional
        Optional stable token for cache-key generation. If `None`, `name` is used.
    """

    name: str
    compute_from_files: typing.Callable[[list[pathlib.Path], "PlotRainFractionConfig"], xarray.Dataset]
    directory_name_fn: typing.Callable[[str, "PlotRainFractionConfig"], str]
    file_glob: str = "*.nc"
    cache_token: typing.Optional[str] = None

    def directory_name(self, location: str, config: "PlotRainFractionConfig") -> str:
        """Build the input directory name for one location.

        Parameters
        ----------
        location : str
            Logical location/site name (e.g., `"Adelaide"`).
        config : PlotRainFractionConfig
            Runtime configuration supplying conversion and other naming context.

        Returns
        -------
        str
            Directory key used to load data for this location (for example,
            `"Adelaide_rain_melbourne"` with the default naming function).
        """
        return self.directory_name_fn(location, config)

    def token(self) -> str:
        """Return the cache token for this processing logic.

        Returns
        -------
        str
            Cache token used in cache file naming. Uses `cache_token` when set,
            otherwise falls back to `name`.
        """
        return self.cache_token or self.name


@dataclass(frozen=True)
class PlotSpec:
    """Describe how processed datasets are rendered.

    A PlotSpec decouples the data loading/caching logic from the visualization
    logic. It supplies callable functions that render panels, generate titles,
    and optionally add secondary axes. This allows multiple plotting styles to
    coexist without altering the core data pipeline.

    Attributes
    ----------
    name : str
        Human-readable name for this plotting specification (e.g.,
        "rain_fraction_quantile_timeseries"). Used for identification only.
    plot_panel_fn : Callable
        Function to render the main plot content on a given axes. Signature:
        `(ax: Axes, key: str, data: xarray.Dataset, config: PlotRainFractionConfig) -> None`.
        Called once per site subplot to draw quantile lines or other data.
    figure_title_fn : Callable
        Function to generate the figure-level title string. Signature:
        `(config: PlotRainFractionConfig) -> str`. Called once per figure.
    panel_title_fn : Callable
        Function to generate each subplot title string. Signature:
        `(key: str, data: xarray.Dataset, config: PlotRainFractionConfig) -> str`.
        Called once per site subplot.
    y_label : str, optional
        Label for the primary y-axis (default: "Rain Fraction"). Applied to
        all subplots.
    y_limits : tuple[float, float], optional
        (ymin, ymax) limits for the primary y-axis, shared across all subplots.
        Default: (0.0, 1.0) which suits fractional data.
    add_secondary_axis_fn : Callable, optional
        Optional function to add a secondary axes overlay (e.g., sample counts).
        Signature: `(ax: Axes, key: str, data: xarray.Dataset, config: PlotRainFractionConfig) -> Optional[Axes]`.
        If None, no secondary axis is created. Default: None.
    """

    name: str
    plot_panel_fn: typing.Callable[[Axes, str, xarray.Dataset, "PlotRainFractionConfig"], None]
    figure_title_fn: typing.Callable[["PlotRainFractionConfig"], str]
    panel_title_fn: typing.Callable[[str, xarray.Dataset, "PlotRainFractionConfig"], str]
    y_label: str = "Rain Fraction"
    y_limits: tuple[float, float] = DEFAULT_Y_LIMITS
    add_secondary_axis_fn: typing.Optional[
        typing.Callable[[Axes, str, xarray.Dataset, "PlotRainFractionConfig"], typing.Optional[Axes]]
    ] = None

    def plot_panel(self, ax: Axes, key: str, data: xarray.Dataset, config: "PlotRainFractionConfig") -> None:
        """Render one panel's main plot content.

        This is a thin wrapper around `plot_panel_fn` that delegates to the
        callable registered in this PlotSpec. Override or customize by providing
        a different `plot_panel_fn` at instantiation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes object on which to draw the plot.
        key : str
            Directory/site key (e.g., "Adelaide_rain_melbourne") identifying which
            dataset is being plotted. Used for legend, titles, or other labeling.
        data : xarray.Dataset
            Processed dataset containing variables like `quant` (quantile values)
            and `count_missing` (sample counts). Loaded from cache or computed on-demand.
        config : PlotRainFractionConfig
            Runtime configuration object containing quantiles, color ranges, thresholds,
            and other user preferences that may influence the plot appearance.

        Returns
        -------
        None
            Modifies `ax` in place to add plot content (lines, markers, etc.).
        """
        self.plot_panel_fn(ax, key, data, config)

    def panel_title(self, key: str, data: xarray.Dataset, config: "PlotRainFractionConfig") -> str:
        """Return the panel title."""
        return self.panel_title_fn(key, data, config)

    def figure_title(self, config: "PlotRainFractionConfig") -> str:
        """Return the figure-level title."""
        return self.figure_title_fn(config)

    def add_secondary_axis(
        self, ax: Axes, key: str, data: xarray.Dataset, config: "PlotRainFractionConfig"
    ) -> typing.Optional[Axes]:
        """Add optional secondary-axis content to a panel."""
        if self.add_secondary_axis_fn is None:
            return None
        return self.add_secondary_axis_fn(ax, key, data, config)


def rain_fraction_directory_name(location: str, config: "PlotRainFractionConfig") -> str:
    """Build the default directory name for a rain-fraction site."""
    return f"{location}_rain_{config.conversion}"

# this function does the actual computation...
def compute_quantiles_from_files(
    files: list[pathlib.Path],
    config: "PlotRainFractionConfig",
) -> xarray.Dataset:
    """Compute monthly rain-fraction quantiles & count of missing data from a list of NetCDF files.

    This is a pure computation step — it knows nothing about directories,
    caching, or paths beyond the files it is given.

    Args:
        files: Sorted list of source NetCDF files to load.
        config: Plot configuration containing quantile settings.


    Returns:
        DataArray of monthly quantiles indexed by `time` and a quantile dimension.

    Raises:
        KeyError: If `rain_fraction` is missing from the source dataset.

    """
    print(f"Reading in data from {len(files)} files: {files[0].parent.name}")
    with xarray.open_mfdataset(files) as ds:
        if "rain_fraction" not in ds:
            raise KeyError(f"rain_fraction variable not found in {files[0].parent}")
        # Mask out zeros — zero fraction means no rain was recorded at all
        # and should not bias the high quantiles.
        ts = ds.rain_fraction.where(ds.rain_fraction > 0).load()
    # Resample to monthly starts (MS) then compute the requested quantiles
    # across all time steps within each calendar month.
    resamp = ts.resample(time="QS")
    count = resamp.count()
    quant = resamp.quantile(config.quantiles)
    # have annoying case of 1 sample from subsequent year (as time must refer to end of period)
    quant = quant.where(count > 1)
    count = count.where(count > 1)
    ds_result = xarray.Dataset(dict(count_missing=count,quant=quant))


    return ds_result


def plot_quantile_panel(
    ax: Axes,
    key: str,
    site_data: xarray.Dataset,
    config: "PlotRainFractionConfig",
) -> None:
    """Plot the default monthly quantile lines for one site."""
    qdata = site_data.quant
    count_missing = site_data.count_missing.rename(dict(time="Time"))
    quantile_dim = get_quantile_dim_name(qdata)
    quantile_styles = get_quantile_styles(config)
    for quantile in config.quantiles:
        style = quantile_styles[quantile]
        qseries = qdata.sel({quantile_dim: quantile}, method="nearest", tolerance=1e-6).rename(dict(time="Time"))
        qseries = qseries.where(count_missing > config.min_samples_per_month)
        qseries.plot(ax=ax, zorder=3, **style)


def add_count_overlay(
    ax: Axes,
    key: str,
    site_data: xarray.Dataset,
    config: "PlotRainFractionConfig",
) -> Axes:
    """Add the default sample-count overlay.

    Args:
        ax: Target primary axes.
        key: Directory key for the panel.
        site_data: Dataset containing ``count_missing``.
        config: Plot configuration.

    Returns:
        The created secondary axes.
    """
    ax_count = ax.twinx()
    ax.set_facecolor("none")
    ax_count.set_facecolor("none")
    count_data = site_data.count_missing
    count_scale = config.count_scale
    count_data_scaled = count_data / count_scale
    count_times = count_data.indexes["time"].to_pydatetime()
    # Use a width in days so matplotlib accepts it consistently for datetime axes.
    bar_width_days = config.bar_width_days
    ax_count.bar(
        count_times,
        count_data_scaled.values,
        width=bar_width_days,
        color="black",
        alpha=0.3,
        zorder=1,
    )
    ax_count.axhline(config.min_samples_per_month / count_scale, color='grey', linestyle='--', linewidth=2)
    ax_count.set_ylabel(f"Sample count / {count_scale:g}", color="black")
    ax_count.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax_count.tick_params(axis="y", labelcolor="grey")
    ax_count.set_ylim(bottom=0)
    ax_count.set_xlabel("")
    return ax_count


def rain_fraction_figure_title(config: "PlotRainFractionConfig") -> str:
    """Build the default figure title."""
    return f"Rain fraction quantiles (conversion: {get_conversion_label(config.conversion)})"


def rain_fraction_panel_title(
    directory_name: str,
    site_data: xarray.Dataset,
    config: "PlotRainFractionConfig",
) -> str:
    """Build the default subplot title.

    Args:
        directory_name: Directory key used in ``tsq``.
        site_data: Processed dataset for the panel. Unused by the default implementation.
        config: Plot configuration. Unused by the default implementation.

    Returns:
        Human-readable panel title.
    """
    return format_panel_title(directory_name)


DEFAULT_PROCESSING_SPEC = ProcessingSpec(
    name="rain_fraction_monthly_quantiles",
    compute_from_files=compute_quantiles_from_files,
    directory_name_fn=rain_fraction_directory_name,
)


DEFAULT_PLOT_SPEC = PlotSpec(
    name="rain_fraction_quantile_timeseries",
    plot_panel_fn=plot_quantile_panel,
    figure_title_fn=rain_fraction_figure_title,
    panel_title_fn=rain_fraction_panel_title,
    y_label="Rain Fraction",
    y_limits=DEFAULT_Y_LIMITS,
    add_secondary_axis_fn=add_count_overlay,
)



def plot_quantiles(
    tsq: dict[str, xarray.Dataset],
    config: "PlotRainFractionConfig",
    save_fig: bool | None = None,
) -> Figure:
    """Plot monthly quantile time series with one subplot per tsq key.

    All panels share the same x and y limits so values are directly comparable
    across sites, and a single figure-level legend is used for quantile lines.

    Args:
        tsq: Mapping of key name to a DataArray with `time` and quantile
            dimensions, as produced by `load_monthly_quantiles`.
        config: Runtime configuration controlling figure layout and labels.
        save_fig: Optional override for whether to save the PNG. If omitted,
            ``config.save_fig`` is used.

    Returns:
        matplotlib figure to allow further tweaking outside code.

    Uses module-level constants:
        SCRIPT_DIR: Root for the ``figures/`` output subdirectory when saving.
    """
    if not tsq:
        raise ValueError("tsq is empty. Provide at least one quantile time series to plot.")

    if save_fig is None:
        save_fig = config.save_fig

    keys = list(tsq)

    fig, axs = ausLib.std_fig_axs(
        f'rain_fraction',
        sharex=True,
        sharey=True,
        clear=True,
        xtime=True,
        figsize=config.fig_size,
        mosaic=config.site_mosaic,
    )


    # One row per site; sharex/sharey ensures the same axis range everywhere.
    # num + clear=True reuses the same window instead of opening a new one each run.
    #expected_figsize = (12, 1+1.5 * len(keys))

    # If figure exists with wrong size, close it so we can recreate with correct dimensions
    #if plt.fignum_exists(config.figure_num):
    #    fig_obj = plt.figure(config.figure_num)
    #    current_size = tuple(fig_obj.get_size_inches())
    #    if current_size != expected_figsize:
    #        plt.close(fig_obj)

    # fig, axes = plt.subplots(
    #     len(keys),
    #     1,
    #     figsize=expected_figsize,
    #     sharex="all",
    #     sharey="all",
    #     num=config.figure_num,
    #     clear=True,
    #     layout='tight'
    # )
    # # Wrap single-axis result so the rest of the code can always iterate over axes.
    # axes = [typing.cast(Axes, axis) for axis in np.atleast_1d(axes).ravel()]

    # Compute shared x and y limits across all datasets so plots are comparable.
    keys=list(tsq.keys())
    x_min = min(tsq[key].time.min().values for key in keys)
    x_max = max(tsq[key].time.max().values for key in keys)
    y_min, y_max = config.y_limits
    x_span = x_max - x_min
    # Add a small fractional margin to each end of the time axis so data points
    # are not clipped against the frame edge.
    x_pad = x_span * config.x_margin_fraction if x_span != np.timedelta64(0, "ns") else np.timedelta64(7, "D")
    major_time_locator = mdates.YearLocator(base=10, month=1, day=1)
    minor_time_locator = mdates.YearLocator(base=1, month=1, day=1)
    major_time_formatter = mdates.DateFormatter("%Y")

    secondary_axes = []  # Track all secondary axes to apply shared range
    for site, ax in axs.items():
        key = config.processing_spec.directory_name(site, config)
        site_data = tsq[key]
        config.plot_spec.plot_panel(ax, key, site_data, config)
        ax.set_title(config.plot_spec.panel_title(key, site_data, config))
        ax.set_ylabel(config.plot_spec.y_label)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min, y_max)
        # Use calendar-aware locators so leap years are handled correctly.
        ax.xaxis.set_major_locator(major_time_locator)
        ax.xaxis.set_minor_locator(minor_time_locator)
        ax.xaxis.set_major_formatter(major_time_formatter)
        ax.grid(True, alpha=0.3)
        # Hide x-axis label and tick labels on non-edge subplots so 'Time' only
        # appears on the bottom panel, reducing clutter.
        ax.label_outer()

        ax_count = config.plot_spec.add_secondary_axis(ax, key, site_data, config)
        if ax_count is not None:
            secondary_axes.append(ax_count)
        ax_count.label_outer()

    # Apply shared range to all secondary axes based on global max count.
    if secondary_axes:
        max_count = max(tsq[key].count_missing.max().values for key in tsq.keys())
        max_count_scaled = max_count / config.count_scale
        # Add 10% headroom for visual clarity.
        for ax_count in secondary_axes:
            ax_count.set_ylim(0, max_count_scaled * 1.1)

    first_ax = next(iter(axs.values()))
    handles, labels = first_ax.get_legend_handles_labels()
    # Place the figure-level title and legend dynamically above the top subplot
    # so spacing adapts to whatever font size and DPI are in use.
    suptitle = fig.suptitle(
        config.plot_spec.figure_title(config),
        y=1.0,
        va="bottom",
    )
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(config.quantiles),
        frameon=False,
    )
    # Tighten the layout so only the measured title + legend height is reserved
    # at the top — avoids the large empty band that tight_layout would otherwise leave.
    position_top_annotations(fig, list(axs.values()), suptitle, legend)

    if save_fig:
        figure_dir = SCRIPT_DIR / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_path = figure_dir / f"{config.figure_num}.png"
        # bbox_inches="tight" crops surplus whitespace from the saved PNG;
        # pad_inches adds a small consistent border around the content.
        fig.savefig(figure_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved figure to {figure_path}")

    # On headless backends (Agg, PDF, etc.) close without trying to open a window.
    # On GUI backends show the figure, blocking only if not in interactive mode.
    show_or_close_figure(fig)

    return fig

@dataclass
class PlotRainFractionConfig:
    """User-facing configuration for rain-fraction quantile loading and plotting."""

    root_dir: pathlib.Path
    locations: list[str] = field(default_factory=list)
    conversion: str = CONVERSION
    quantiles: list[float] = field(default_factory=list)
    force_recompute_cache: bool = FORCE_RECOMPUTE_CACHE
    save_fig: bool = SAVE_FIG
    min_samples_per_month: int = MIN_SAMPLES_PER_MONTH
    figure_num: str = FIGURE_NUM
    x_margin_fraction: float = X_MARGIN_FRACTION
    cache_dir_name: str = CACHE_DIR_NAME
    quantile_color_low: float = QUANTILE_COLOR_LOW
    quantile_color_high: float = QUANTILE_COLOR_HIGH
    count_scale: float = COUNT_SCALE_DEFAULT
    bar_width_days: float = BAR_WIDTH_DAYS_DEFAULT
    fig_size: tuple[float, float] = DEFAULT_FIG_SIZE
    y_limits: tuple[float, float] = DEFAULT_Y_LIMITS
    site_mosaic: list[list[str]] = field(default_factory=list)
    processing_spec: ProcessingSpec = field(default_factory=lambda: DEFAULT_PROCESSING_SPEC)
    plot_spec: PlotSpec = field(default_factory=lambda: DEFAULT_PLOT_SPEC)

    @property
    def active_locations(self) -> list[str]:
        """Return locations sourced from the mosaic (fallback to explicit locations)."""
        if self.site_mosaic:
            return flatten_mosaic_locations(self.site_mosaic)
        return self.locations

    @property
    def directories(self) -> list[str]:
        """Return per-site directory names derived from locations and conversion."""
        return [self.processing_spec.directory_name(location, self) for location in self.active_locations]


def make_default_config() -> PlotRainFractionConfig:
    """Build a configuration object from the module's default values."""
    return PlotRainFractionConfig(
        root_dir=ROOT_DIR,
        locations=flatten_mosaic_locations(DEFAULT_SITE_MOSAIC),
        conversion=CONVERSION,
        quantiles=list(QUANTILES),
        force_recompute_cache=FORCE_RECOMPUTE_CACHE,
        save_fig=SAVE_FIG,
        count_scale=COUNT_SCALE_DEFAULT,
        bar_width_days=BAR_WIDTH_DAYS_DEFAULT,
        fig_size=DEFAULT_FIG_SIZE,
        y_limits=DEFAULT_Y_LIMITS,
        site_mosaic=[list(row) for row in DEFAULT_SITE_MOSAIC],
    )


def save_gui_config(
    config: PlotRainFractionConfig,
    file_path: pathlib.Path = DEFAULT_GUI_CONFIG_FILE,
) -> None:
    """Persist GUI-editable configuration values to a single JSON file."""
    payload = dict(
        root_dir=str(config.root_dir),
        conversion=config.conversion,
        quantiles=list(config.quantiles),
        force_recompute_cache=config.force_recompute_cache,
        save_fig=config.save_fig,
        min_samples_per_month=config.min_samples_per_month,
        quantile_color_low=config.quantile_color_low,
        quantile_color_high=config.quantile_color_high,
        count_scale=config.count_scale,
        bar_width_days=config.bar_width_days,
        fig_size=list(config.fig_size),
        y_limits=list(config.y_limits),
        site_mosaic=config.site_mosaic,
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_gui_config(
    file_path: pathlib.Path = DEFAULT_GUI_CONFIG_FILE,
) -> PlotRainFractionConfig:
    """Load saved GUI config if present; otherwise return defaults."""
    config = make_default_config()
    if not file_path.exists():
        return config

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
    except Exception as exc:
        print(f"Warning: failed to read saved GUI config {file_path}: {exc}")
        return config

    if root_dir := saved.get("root_dir"):
        config.root_dir = pathlib.Path(root_dir)
    if conversion := saved.get("conversion"):
        config.conversion = conversion
    if quantiles := saved.get("quantiles"):
        config.quantiles = list(quantiles)
    if "force_recompute_cache" in saved:
        config.force_recompute_cache = bool(saved["force_recompute_cache"])
    if "save_fig" in saved:
        config.save_fig = bool(saved["save_fig"])
    if "min_samples_per_month" in saved:
        config.min_samples_per_month = int(saved["min_samples_per_month"])
    if "quantile_color_low" in saved:
        config.quantile_color_low = float(saved["quantile_color_low"])
    if "quantile_color_high" in saved:
        config.quantile_color_high = float(saved["quantile_color_high"])
    if "count_scale" in saved:
        config.count_scale = float(saved["count_scale"])
    if "bar_width_days" in saved:
        config.bar_width_days = float(saved["bar_width_days"])
    if fig_size := saved.get("fig_size"):
        config.fig_size = (float(fig_size[0]), float(fig_size[1]))
    if y_limits := saved.get("y_limits"):
        config.y_limits = (float(y_limits[0]), float(y_limits[1]))
    if site_mosaic := saved.get("site_mosaic"):
        config.site_mosaic = site_mosaic
        config.locations = flatten_mosaic_locations(site_mosaic)

    return config


def get_quantile_styles(config: PlotRainFractionConfig) -> dict[float, dict[str, str]]:
    """Return a plotting style dict for each configured quantile.

    Colours are mapped by log(return time), where return time = 1 / (1 - q).
    Highest return time (most extreme, q→1) → darkest blue; lowest return time → lightest.
    This ensures visually even spacing even when quantiles like 0.99 and 1.0 are
    very close in probability but span orders of magnitude in return time.

    For q == 1.0 (infinite return time) the log return time is set one step beyond
    the next finite quantile so it receives a distinct, darker colour.
    """
    styles: dict[float, dict[str, str]] = {}
    numeric_quantiles = [float(value) for value in config.quantiles]

    # Compute log10(return time) for each quantile; handle q==1 specially.
    finite_log_rts = [np.log10(1.0 / (1.0 - q)) for q in numeric_quantiles if not np.isclose(q, 1.0)]
    if finite_log_rts:
        log_rt_step = (max(finite_log_rts) - min(finite_log_rts)) / max(len(finite_log_rts) - 1, 1)
    else:
        log_rt_step = 1.0
    # Assign q==1.0 a log-RT one step beyond the largest finite return time.
    log_rt_for_one = (max(finite_log_rts) if finite_log_rts else 2.0) + log_rt_step

    log_rts = [
        log_rt_for_one if np.isclose(q, 1.0) else np.log10(1.0 / (1.0 - q))
        for q in numeric_quantiles
    ]
    log_rt_min = min(log_rts)
    log_rt_max = max(log_rts)

    def quantile_to_blue(log_rt: float) -> str:
        # Highest log-RT (q=1.0 / longest return time) → cmap HIGH (darkest blue);
        # lowest log-RT (least extreme quantile) → cmap LOW (lightest blue).
        if np.isclose(log_rt_max, log_rt_min):
            scaled = 0.5
        else:
            scaled = (log_rt - log_rt_min) / (log_rt_max - log_rt_min)
        cmap_value = config.quantile_color_low + scaled * (config.quantile_color_high - config.quantile_color_low)
        rgba = plt.get_cmap("Blues")(cmap_value)
        return mcolors.to_hex(typing.cast(typing.Any, rgba))

    for quantile, log_rt in zip(numeric_quantiles, log_rts):
        matched_style = next(
            (dict(style) for value, style in QUANTILE_STYLES.items() if np.isclose(value, quantile)),
            None,
        )
        label = matched_style["label"] if matched_style and "label" in matched_style else (
            "Max" if np.isclose(quantile, 1.0) else f"{quantile * 100:g}%"
        )
        style = {"label": label, "color": quantile_to_blue(log_rt)}
        if matched_style is not None:
            # Preserve optional style keys such as linestyle, marker, etc.
            for key, value in matched_style.items():
                if key not in {"label", "color"}:
                    style[key] = value
        styles[quantile] = style
    return styles


def edit_config_via_tkinter(config: PlotRainFractionConfig) -> PlotRainFractionConfig | None:
    """Open a small Tkinter form for editing the user-facing configuration.

    Returns ``None`` if the dialog is cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as exc:
        raise RuntimeError("Tkinter is not available in this Python environment.") from exc

    result: PlotRainFractionConfig | None = None

    def parse_csv(text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]

    def parse_mosaic(text: str) -> list[list[str]]:
        text = text.strip()
        if not text:
            raise ValueError("Site mosaic must not be empty.")

        # Backward-compatible: accept either a full list-of-lists string or
        # one sub-list per line.
        if text.startswith("[["):
            parsed_rows = ast.literal_eval(text)
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            parsed_rows = [ast.literal_eval(line) for line in lines]

        if not isinstance(parsed_rows, list) or not parsed_rows:
            raise ValueError("Site mosaic must be a non-empty list of rows.")

        mosaic: list[list[str]] = []
        for row in parsed_rows:
            if not isinstance(row, list) or not row:
                raise ValueError("Each mosaic row must be a non-empty list.")
            clean_row: list[str] = []
            for cell in row:
                if not isinstance(cell, str) or not cell.strip():
                    raise ValueError("Each mosaic cell must be a non-empty string.")
                clean_row.append(cell.strip())
            mosaic.append(clean_row)
        return mosaic

    def browse_root_dir() -> None:
        selected = filedialog.askdirectory(initialdir=str(root_dir_var.get() or config.root_dir))
        if selected:
            root_dir_var.set(selected)

    def apply_config_to_form(cfg: PlotRainFractionConfig) -> None:
        root_dir_var.set(str(cfg.root_dir))
        conversion_var.set(cfg.conversion)
        quantiles_var.set(", ".join(f"{value:g}" for value in cfg.quantiles))
        min_samples_var.set(str(cfg.min_samples_per_month))
        color_low_var.set(f"{cfg.quantile_color_low:g}")
        color_high_var.set(f"{cfg.quantile_color_high:g}")
        count_scale_var.set(f"{cfg.count_scale:g}")
        bar_width_days_var.set(f"{cfg.bar_width_days:g}")
        fig_size_var.set(f"{cfg.fig_size[0]:g},{cfg.fig_size[1]:g}")
        y_limits_var.set(f"{cfg.y_limits[0]:g},{cfg.y_limits[1]:g}")
        force_recompute_var.set(cfg.force_recompute_cache)
        save_fig_var.set(cfg.save_fig)
        site_mosaic_text.delete("1.0", "end")
        site_mosaic_text.insert("1.0", format_mosaic_lines(cfg.site_mosaic or DEFAULT_SITE_MOSAIC))

    def on_reset() -> None:
        if DEFAULT_GUI_CONFIG_FILE.exists():
            DEFAULT_GUI_CONFIG_FILE.unlink()
        apply_config_to_form(make_default_config())

    def on_run() -> None:
        nonlocal result
        try:
            parsed_site_mosaic = parse_mosaic(site_mosaic_text.get("1.0", "end"))
            parsed_locations = flatten_mosaic_locations(parsed_site_mosaic)
            parsed_quantiles = [float(value) for value in parse_csv(quantiles_var.get())]
            parsed_min_samples = int(min_samples_var.get())
            parsed_color_low = float(color_low_var.get())
            parsed_color_high = float(color_high_var.get())
            parsed_count_scale = float(count_scale_var.get())
            parsed_bar_width_days = float(bar_width_days_var.get())
            # Parse fig_size as "width,height" (e.g., "12,8")
            fig_size_parts = [float(x.strip()) for x in fig_size_var.get().split(",")]
            if len(fig_size_parts) != 2:
                raise ValueError("Figure size must be in format 'width,height' (e.g., '12,8').")
            parsed_fig_size: tuple[float, float] = (fig_size_parts[0], fig_size_parts[1])
            y_limit_parts = [float(x.strip()) for x in y_limits_var.get().split(",")]
            if len(y_limit_parts) != 2:
                raise ValueError("Y limits must be in format 'ymin,ymax' (e.g., '0,1').")
            parsed_y_limits: tuple[float, float] = (y_limit_parts[0], y_limit_parts[1])
            if not parsed_locations:
                raise ValueError("Site mosaic must include at least one non-BLANK location.")
            if len(set(parsed_locations)) != len(parsed_locations):
                raise ValueError("Site mosaic contains duplicate non-BLANK site names.")
            if not parsed_quantiles:
                raise ValueError("At least one quantile is required.")
            if parsed_min_samples < 0:
                raise ValueError("Min samples per month must be non-negative.")
            if not (0.0 <= parsed_color_low < parsed_color_high <= 1.0):
                raise ValueError("Color range must satisfy 0 <= low < high <= 1.")
            if parsed_count_scale <= 0:
                raise ValueError("Count scale must be > 0.")
            if parsed_bar_width_days <= 0:
                raise ValueError("Bar width (days) must be > 0.")
            if parsed_fig_size[0] <= 0 or parsed_fig_size[1] <= 0:
                raise ValueError("Figure size dimensions must be > 0.")
            if parsed_y_limits[0] >= parsed_y_limits[1]:
                raise ValueError("Y limits must satisfy ymin < ymax.")
            result = PlotRainFractionConfig(
                root_dir=pathlib.Path(root_dir_var.get()),
                locations=parsed_locations,
                conversion=conversion_var.get().strip(),
                quantiles=parsed_quantiles,
                force_recompute_cache=force_recompute_var.get(),
                save_fig=save_fig_var.get(),
                min_samples_per_month=parsed_min_samples,
                figure_num=config.figure_num,
                x_margin_fraction=config.x_margin_fraction,
                cache_dir_name=config.cache_dir_name,
                quantile_color_low=parsed_color_low,
                quantile_color_high=parsed_color_high,
                count_scale=parsed_count_scale,
                bar_width_days=parsed_bar_width_days,
                fig_size=parsed_fig_size,
                y_limits=parsed_y_limits,
                site_mosaic=parsed_site_mosaic,
            )
            save_gui_config(result)
            window.destroy()
        except Exception as exc:  # GUI validation feedback
            messagebox.showerror("Invalid configuration", str(exc))

    def on_cancel() -> None:
        window.destroy()

    window = tk.Tk()
    window.title("Rain Fraction Plot Configuration")
    window.resizable(False, False)

    root_dir_var = tk.StringVar(value=str(config.root_dir))
    conversion_var = tk.StringVar(value=config.conversion)
    quantiles_var = tk.StringVar(value=", ".join(f"{value:g}" for value in config.quantiles))
    min_samples_var = tk.StringVar(value=str(config.min_samples_per_month))
    color_low_var = tk.StringVar(value=f"{config.quantile_color_low:g}")
    color_high_var = tk.StringVar(value=f"{config.quantile_color_high:g}")
    count_scale_var = tk.StringVar(value=f"{config.count_scale:g}")
    bar_width_days_var = tk.StringVar(value=f"{config.bar_width_days:g}")
    fig_size_var = tk.StringVar(value=f"{config.fig_size[0]:g},{config.fig_size[1]:g}")
    y_limits_var = tk.StringVar(value=f"{config.y_limits[0]:g},{config.y_limits[1]:g}")
    force_recompute_var = tk.BooleanVar(value=config.force_recompute_cache)
    save_fig_var = tk.BooleanVar(value=config.save_fig)

    fields = [
        ("Conversion", conversion_var),
        ("Quantiles (comma-separated)", quantiles_var),
        ("Min samples per month", min_samples_var),
        ("Blue color low (0-1)", color_low_var),
        ("Blue color high (0-1)", color_high_var),
        ("Count scale", count_scale_var),
        ("Count bar width (days)", bar_width_days_var),
        ("Figure size (width,height in inches)", fig_size_var),
        ("Y limits (ymin,ymax)", y_limits_var),
    ]

    # Root dir row + browse button
    row = 0
    tk.Label(window, text="Root directory").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    tk.Entry(window, textvariable=root_dir_var, width=45).grid(row=row, column=1, padx=8, pady=4)
    tk.Button(window, text="Browse", command=browse_root_dir).grid(row=row, column=2, padx=8, pady=4)

    # Site mosaic editor: one sub-list per line
    row += 1
    tk.Label(window, text="Site mosaic (one Python sub-list per line\n use 'BLANK' for empty list)").grid(
        row=row, column=0, sticky="nw", padx=8, pady=4
    )
    site_mosaic_text = tk.Text(window, width=45, height=6)
    site_mosaic_text.grid(row=row, column=1, padx=8, pady=4, sticky="we")
    initial_mosaic = config.site_mosaic if config.site_mosaic else DEFAULT_SITE_MOSAIC
    site_mosaic_text.insert("1.0", "\n".join(repr(mrow) for mrow in initial_mosaic))

    # Remaining single-line entry fields
    row += 1
    for label_text, variable in fields:
        tk.Label(window, text=label_text).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        tk.Entry(window, textvariable=variable, width=45).grid(row=row, column=1, padx=8, pady=4)
        row += 1

    tk.Label(
        window,
        text=(
            "Mosaic example lines: ['Mornington','Cairns','BLANK']"
            "   and   ['Grafton','Brisbane','Gladstone']"
        ),
        fg="gray30",
    ).grid(row=row, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 4))

    tk.Label(
        window,
        text="Higher return-time quantiles are darker blue. Valid range: 0 <= low < high <= 1.",
        fg="gray30",
    ).grid(row=row + 1, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 6))

    tk.Checkbutton(window, text="Force recompute cache", variable=force_recompute_var).grid(
        row=row + 2, column=0, columnspan=2, sticky="w", padx=8, pady=4
    )
    tk.Label(
        window,
        text=f"Data cache dir: <root_dir>/{CACHE_DIR_NAME}",
        fg="gray30",
    ).grid(row=row + 3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
    tk.Checkbutton(window, text="Save figure as PNG", variable=save_fig_var).grid(
        row=row + 4, column=0, columnspan=2, sticky="w", padx=8, pady=4
    )
    tk.Button(window, text="Reset defaults", command=on_reset).grid(
        row=row + 5, column=0, sticky="w", padx=8, pady=8
    )
    tk.Button(window, text="Run", command=on_run).grid(row=row + 5, column=1, sticky="e", padx=8, pady=8)
    tk.Button(window, text="Cancel", command=on_cancel).grid(
        row=row + 5, column=2, sticky="w", padx=8, pady=8
    )

    window.mainloop()
    return result


def run_from_config(config: PlotRainFractionConfig) -> dict[str, xarray.Dataset]:
    """Load data and produce the plot for a supplied configuration."""
    tsq = load_monthly_quantiles(config)
    plot_quantiles(tsq, config)
    return tsq




def load_monthly_quantiles(config: PlotRainFractionConfig) -> dict[str, xarray.Dataset]:
    """Load files and compute one processed dataset for each directory.

    Delegates per-directory caching to `get_directory_quantiles`. The
    processing logic itself is supplied by ``config.processing_spec``.

    Cache behavior summary
    ----------------------
    - Cache directory: `config.root_dir / config.cache_dir_name`
    - `config.force_recompute_cache=False`: try cache first.
    - `config.force_recompute_cache=True`: skip cache reads, recompute, and
      refresh cache files.

    Args:
        config: Runtime configuration containing source locations, quantiles,
            cache settings, and other user-facing options.

    Returns:
        Dictionary mapping each directory key to its processed dataset.

    Raises:
        FileNotFoundError: If a requested directory has no `.nc` files.
        KeyError: If `rain_fraction` is missing from an input dataset.

    Uses module-level constants:
        None. The configuration object supplies all user-editable values.
    """
    cache_dir = config.root_dir / config.cache_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        directory: get_directory_quantiles(config, directory, cache_dir)
        for directory in config.directories
    }



def get_directory_quantiles(
    config: PlotRainFractionConfig,
    directory: str,
    cache_dir: pathlib.Path,
) -> xarray.Dataset:
    """Return the processed dataset for one directory.

    This is the main entry point for fetching quantiles. It lists source files
    once, then chooses one path:
    - cache hit: load and return cached dataset
    - cache miss/stale cache: recompute and write a refreshed cache file
    - force recompute: always recompute, then refresh cache file

    Args:
        config: Runtime configuration containing root directory, quantiles,
            cache options, and processing spec.
        directory: Subdirectory name to process.
        cache_dir: Directory in which cache files are stored.

    Returns:
        Processed dataset for one directory.

    Raises:
        FileNotFoundError: If no source files are found in the directory.
    """
    # List source files once; every downstream step reuses this list.
    files = sorted((config.root_dir / directory).glob(config.processing_spec.file_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching {config.processing_spec.file_glob!r} found in {(config.root_dir / directory)!s}"
        )
    # construct cache_args as have two places where it is called.
    cache_args = (cache_dir, directory, files, config.quantiles, config.processing_spec.token())
    if not config.force_recompute_cache:
        # Ask the cache layer for a result or a write path.
        cache_outcome = get_cache_result(*cache_args)
        if isinstance(cache_outcome, xarray.Dataset): # TODO should really check not a pathlib.Path
            return cache_outcome  # cache hit — done
        cache_write_path = cache_outcome  # cache miss — remember where to save
    else:
        # Forced recompute: skip cache read but still write a fresh cache entry.
        print(f"Bypassing cache and recomputing monthly quantiles for {directory}")
        cache_write_path = build_cache_path(*cache_args)

    # Delegate pure computation to the active processing spec.
    result = config.processing_spec.compute_from_files(files, config)

    # Persist the result so future runs can reuse it.
    result.to_netcdf(cache_write_path)
    print(f"Cached monthly quantiles for {directory} -> {cache_write_path}")
    return result


def get_cache_result(
    cache_dir: pathlib.Path,
    directory: str,
    files: list[pathlib.Path],
    quantiles: list[float],
    processing_token: str,
) -> xarray.Dataset | pathlib.Path:
    """Return a cached DataArray on a cache hit, or the cache write path on a miss.

    The caller should check the return type:
    - ``xarray.Dataset`` → cache hit; use directly.
    - ``pathlib.Path``     → cache miss; compute a result and save it to that path.

    Args:
        cache_dir: Directory in which cache files are stored.
        directory: Name of the source subdirectory (used as a filename prefix).
        files: Sorted list of source NetCDF files for this directory.
        quantiles: Quantile levels encoded in the cache filename.
        processing_token: Token identifying the processing logic used.

    Returns:
        Loaded Dataset if the cache is fresh, otherwise the path to write to.
    """
    cache_file = build_cache_path(cache_dir, directory, files, quantiles, processing_token)
    if is_cache_fresh(cache_file, files):
        # Cache is fresh — load and return the DataArray directly.
        print(f"Loading cached monthly quantiles for {directory}")
        with xarray.open_dataset(cache_file) as cached:
            return cached.load()
    # Cache is stale or absent — return the path so the caller knows where to save.
    return cache_file


def is_cache_fresh(cache_file: pathlib.Path, files: list[pathlib.Path]) -> bool:
    """Return True if the cache file is newer than all source files.

    Args:
        cache_file: Path to the candidate cache NetCDF file.
        files: Source NetCDF files whose mtimes the cache must post-date.

    Returns:
        True if the cache exists and is more recent than every source file.
    """
    if not cache_file.exists():
        return False
    latest_source_mtime = max(path.stat().st_mtime_ns for path in files)
    return cache_file.stat().st_mtime_ns >= latest_source_mtime


def build_cache_path(
    cache_dir: pathlib.Path,
    directory: str,
    files: list[pathlib.Path],
    quantiles: list[float],
    processing_token: str,
) -> pathlib.Path:
    """Return the cache file path for a directory's processed dataset.

    The path embeds a short hash of the source files' names, sizes, and
    modification times, plus the chosen quantile values and processing token.
    This means the cache is automatically invalidated whenever source data,
    quantiles, or processing logic change.

    Args:
        cache_dir: Directory in which cache files are stored.
        directory: Name of the source subdirectory (used as a filename prefix).
        files: Sorted list of source NetCDF files for this directory.
        quantiles: Quantile levels encoded into the cache filename.
        processing_token: Token identifying the processing logic used.

    Returns:
        Path to the corresponding cache NetCDF file.

    """
    # Build a fingerprint from each file's name, size, and last-modified time.
    # Any change to a source file produces a different hash and invalidates cache.
    file_signature = "|".join(
        f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}" for path in files
    )
    signature_hash = hashlib.sha1(file_signature.encode("utf-8")).hexdigest()[:12]
    # Embed quantile values in the filename so changing QUANTILES also invalidates
    # old cache files without needing to inspect the file contents.
    quantile_token = "_".join(f"{q:g}" for q in quantiles)
    return cache_dir / f"{directory}_{processing_token}_q_{quantile_token}_{signature_hash}.nc"



def get_quantile_dim_name(qdata: xarray.DataArray) -> str:
    """Return the quantile dimension name used by a quantile DataArray.

    Args:
        qdata: DataArray produced by a quantile reduction.

    Returns:
        Name of the quantile dimension (`quantile` or `q`).

    Raises:
        ValueError: If no supported quantile dimension is present.
    """
    if "quantile" in qdata.dims:
        return "quantile"
    if "q" in qdata.dims:
        return "q"
    raise ValueError(f"Expected a quantile dimension named 'quantile' or 'q', found {qdata.dims}")


def format_panel_title(directory_name: str) -> str:
    """Return a readable subplot title from a directory key.

    Expected keys look like '<site>_rain_<source>'. If that pattern is present,
    the title includes both site and rain source; otherwise underscores are
    replaced with spaces.

    Args:
        directory_name: Directory key used in `tsq`, usually in the form
            '<site>_rain_<source>'.

    Returns:
        A human-readable panel title.
    """
    left, _, right = directory_name.partition("_rain_")
    return left


def get_conversion_label(conversion: str | list[str]) -> str:
    """Return a readable conversion label for figure-level titles.

    Args:
        conversion: Conversion setting as a single string or list of strings.

    Returns:
        Comma-separated conversion label.
    """
    if isinstance(conversion, str):
        return conversion
    return ", ".join(conversion)


def position_top_annotations(fig: Figure, axes: list[Axes], suptitle: Text, legend: Legend) -> None:
    """Place the figure title and legend close to the top subplot.

    The title and legend heights are measured after an initial draw, then the
    subplot area is tightened so only the required vertical space is reserved.

    Args:
        fig: Target matplotlib figure.
        axes: Array of subplot axes.
        suptitle: Figure-level title object.
        legend: Figure-level legend object.

    Uses module-level constants:
        AXES_TO_LEGEND_GAP: Figure-fraction gap between the top subplot and the legend.
        LEGEND_TO_TITLE_GAP: Figure-fraction gap between the legend and the suptitle.
        TOP_MARGIN: Minimum figure-fraction clearance above the suptitle.
    """
    # First pass: draw the figure so Matplotlib knows the actual pixel sizes of
    # the title and legend text at the chosen DPI and font size.
    fig.canvas.draw()
    canvas = typing.cast(typing.Any, fig.canvas)
    renderer = canvas.get_renderer()
    title_bbox = suptitle.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())

    # Calculate exactly how much vertical space title + legend + gaps need, then
    # shrink the subplot grid to leave only that space at the top.
    reserved_top = title_bbox.height + legend_bbox.height + AXES_TO_LEGEND_GAP + LEGEND_TO_TITLE_GAP + TOP_MARGIN
    fig.tight_layout(rect=(0, 0, 1, max(0.2, 1 - reserved_top)))

    # Second pass: after tight_layout has repositioned the axes, re-measure
    # the first subplot's top edge in figure-fraction coordinates.
    fig.canvas.draw()
    renderer = canvas.get_renderer()
    first_axis_bbox = axes[0].get_tightbbox(renderer=renderer).transformed(fig.transFigure.inverted())

    legend_height = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).height
    title_height = suptitle.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).height

    # Stack the legend immediately above the first subplot with a small gap.
    legend_bottom = first_axis_bbox.y1 + AXES_TO_LEGEND_GAP
    typing.cast(typing.Any, legend).set_bbox_to_anchor((0.5, legend_bottom), transform=fig.transFigure)

    # Stack the title immediately above the legend with another small gap.
    title_bottom = legend_bottom + legend_height + LEGEND_TO_TITLE_GAP
    suptitle.set_y(title_bottom)
    typing.cast(typing.Any, suptitle).set_va("bottom")

    # Safety clamp: if the title would overflow the figure top, nudge it down.
    max_title_top = 1 - TOP_MARGIN
    if title_bottom + title_height > max_title_top:
        suptitle.set_y(max_title_top - title_height)

    # Final draw to apply all repositioning before save or display.
    fig.canvas.draw()


def show_or_close_figure(fig: Figure) -> None:
    """Show the figure on interactive backends and close it otherwise.

    This keeps local interactive use convenient while avoiding hangs in
    headless or notebook-style backends that should not open GUI windows.

    Args:
        fig: Figure to display or close.
    """
    backend = plt.get_backend().lower()
    non_interactive_backends = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}
    if backend in non_interactive_backends or "inline" in backend:
        plt.close(fig)
        return

    try:
        plt.show(block=not plt.isinteractive())
    except TypeError:
        plt.show()


if __name__ == "__main__":
    # --- Quick config guide ---
    # Non-GUI run (default): edit the DEFAULT_* constants and flags at the top
    # of this file, then run as normal.
    #
    # GUI run: set USE_GUI = True at the top of this file.  A small form will
    # pop up before any data is loaded, letting you change root directory,
    # locations, conversion, quantiles, blue color range, and save/cache options.
    # --------------------------
    config = load_gui_config()
    if USE_GUI:
        selected_config = edit_config_via_tkinter(config)
        if selected_config is None:
            raise SystemExit("GUI cancelled by user.")
        config = selected_config
    tsq = run_from_config(config)
